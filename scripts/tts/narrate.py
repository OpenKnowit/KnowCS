"""Narration for the /watch/ explainers: Gemini TTS clips, one per caption, assembled into one track per episode
and language, plus the manifest the player reads (src/site/watch/narration.ts).

  python3 -I scripts/tts/narrate.py gen   knn [more ids | all] [--lang zh,yue,en]   # missing clips only
  python3 -I scripts/tts/narrate.py build knn [more ids | all] [--lang …]           # track + manifest
  python3 -I scripts/tts/narrate.py qa    knn [more ids | all] [--lang …]           # listen back, flag mistakes

Scripts: src/content/narration/<id>.json, {lang: {scene id: [one line per caption]}}; a line is a string, or
{"text": shown as the caption, "say": what is read} when the two differ. Needs GOOGLE_GEMINI_API_KEY (from .env),
ffmpeg, node (for the scene timings) and SSL_CERT_FILE=/etc/ssl/cert.pem with python.org builds.
Clips are cached by content in work/dubbing/clips/, so an edited line is the only one generated again.
Output: work/dubbing/out/audio/<id>/<lang>.json + <lang>.<hash>.m4a (or $KNOWCS_AUDIO_DIR), served at /audio/.
"""
import argparse
import array
import base64
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "src", "content", "narration")
WORK = os.path.join(ROOT, "work", "dubbing")
CLIPS = os.path.join(WORK, "clips")
QA = os.path.join(WORK, "qa")
OUT = os.environ.get("KNOWCS_AUDIO_DIR") or os.path.join(WORK, "out", "audio")

MODEL = "gemini-3.8-flash-tts"
LISTENER = "gemini-3.8-flash"
VOICES = {"zh": "Charon", "yue": "Iapetus", "en": "Charon"}
STYLES = {
    "zh": "a warm, clear university lecturer speaking standard Mandarin, unhurried",
    "yue": "a warm, clear university lecturer speaking natural Hong Kong Cantonese, unhurried",
    "en": "a warm, clear university lecturer, unhurried",
}
LANGS = ["zh", "yue", "en"]
NAMES = {"zh": "Mandarin", "yue": "Cantonese", "en": "English"}

RATE = 24000  # Gemini TTS returns 24 kHz mono 16-bit WAV
LEAD_MS = 300  # silence before the first line of a scene
GAP_MS = 450  # silence after every line
MIN_SPACING_S = 6.5  # Tier 1: 10 TTS requests a minute


def key():
    k = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not k:
        sys.exit("GOOGLE_GEMINI_API_KEY is not set (set -a; source .env; set +a)")
    return k


def post(url, body, timeout=90):
    """POST JSON with retries: the local proxy drops connections now and then, and 429 means wait"""
    for i in range(10):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"x-goog-api-key": key(), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 400 or e.code == 403:
                sys.exit(f"HTTP {e.code}: {msg}")
            if e.code == 429 and "per day" in msg.lower():
                sys.exit(f"daily quota used up: {msg}")
            print(f"    retry {i + 1}: HTTP {e.code} {msg[:160]}", flush=True)
            time.sleep(30 if e.code == 429 else 10)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            print(f"    retry {i + 1}: {e!r}"[:160], flush=True)
            time.sleep(10)
    sys.exit("gave up after 10 tries")


def find_audio(o):
    if isinstance(o, dict):
        if isinstance(o.get("data"), str) and len(o["data"]) > 1000:
            return o["data"]
        o = list(o.values())
    if isinstance(o, list):
        for v in o:
            r = find_audio(v)
            if r:
                return r
    return None


def load_script(ep):
    with open(os.path.join(SCRIPTS, f"{ep}.json"), encoding="utf-8") as f:
        return json.load(f)


def episodes(args):
    have = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SCRIPTS, "*.json")))
    if args.ids == ["all"]:
        return have
    missing = [e for e in args.ids if e not in have]
    if missing:
        sys.exit(f"no script for {missing} in {SCRIPTS}")
    return args.ids


def text_of(line):
    return line if isinstance(line, str) else line["text"]


def say_of(line):
    return line if isinstance(line, str) else line.get("say", line["text"])


def clip_path(lang, line):
    h = hashlib.sha1(f"{MODEL}|{VOICES[lang]}|{STYLES[lang]}|{say_of(line)}".encode()).hexdigest()[:16]
    return os.path.join(CLIPS, lang, f"{h}.wav")


def lines_of(script, lang):
    for scene, lines in script[lang].items():
        for k, line in enumerate(lines):
            yield scene, k, line


def read_pcm(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
    return a


def trim(a, floor=500, keep_ms=60):
    """drop the silence the model leaves at either end, keeping a short fade margin"""
    loud = [i for i in range(0, len(a), 120) if max(abs(x) for x in a[i : i + 120]) > floor]
    if not loud:
        return a
    keep = keep_ms * RATE // 1000
    return a[max(0, loud[0] - keep) : min(len(a), loud[-1] + 120 + keep)]


def clip_ms(path):
    return len(trim(read_pcm(path))) * 1000 // RATE


# ---------------------------------------------------------------- gen


def tts(text, lang):
    """one request: WAV bytes for text in the language's voice"""
    body = {
        "model": MODEL,
        "input": [{"type": "user_input", "content": [{"type": "text", "text": text, "annotations": [{"type": "speech_metadata", "style": STYLES[lang]}]}]}],
        "generation_config": {"speech_config": [{"voice": VOICES[lang]}]},
    }
    data = find_audio(post("https://generativelanguage.googleapis.com/v1beta/interactions", body))
    if not data:
        sys.exit("no audio in the response")
    return base64.b64decode(data)


def weight(text):
    """rough speaking time of a line: Chinese characters and digits count fully, Latin letters about a third"""
    w = 0.0
    for ch in text:
        if "\u3400" <= ch <= "\u9fff" or ch.isdigit():
            w += 1
        elif ch.isalpha():
            w += 0.35
        elif ch in "，。、：；？！,.;:?!—":
            w += 0.6
    return max(w, 1.0)


def line_starts(pcm, texts, lang):
    """ask the listening model where each line starts (seconds). Pauses alone cannot tell a line break from a
    sentence break, but these times (good to about half a second) can. None if the answer is unusable."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "t.wav")
        write_wav(wav, pcm)
        with open(wav, "rb") as f:
            audio = base64.b64encode(f.read()).decode()
    numbered = "\n".join(f"{k + 1}. {t}" for k, t in enumerate(texts))
    prompt = (
        f"The audio ({NAMES[lang]}) reads these {len(texts)} lines in order:\n{numbered}\n\n"
        'For each line give the time in seconds (one decimal) where its FIRST word starts. Return JSON {"starts": [{"n": 1, "start": 0.0}, ...]}.'
    )
    body = {"contents": [{"parts": [{"inlineData": {"mimeType": "audio/wav", "data": audio}}, {"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    try:
        res = post(f"https://generativelanguage.googleapis.com/v1beta/models/{LISTENER}:generateContent", body)
        st = sorted(json.loads(res["candidates"][0]["content"]["parts"][-1]["text"])["starts"], key=lambda x: x["n"])
        starts = [float(x["start"]) for x in st]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if len(starts) != len(texts) or any(b <= a for a, b in zip(starts, starts[1:])):
        return None
    return starts


def checked_starts(pcm, texts, lang, label, tries=3):
    """line times from the listener, checked against where the lines' lengths put them. The listener is usually
    right to half a second but now and then returns nonsense for a long take: ask again, and replace any time still
    far from the estimate by the estimate."""
    total = len(pcm) / RATE
    ws = [weight(t) for t in texts]
    est = [total * sum(ws[:k]) / sum(ws) for k in range(len(texts))]
    limit = max(8.0, 0.08 * total)
    best = None
    for _ in range(tries):
        st = line_starts(pcm, texts, lang)
        if st is None:
            continue
        bad = sum(abs(a - b) > limit for a, b in zip(st, est))
        if best is None or bad < best[0]:
            best = (bad, st)
        if bad == 0:
            break
    if best is None:
        print(f"  {label}: no line times from the listener, cutting by length only", flush=True)
        return None
    bad, st = best
    if bad:
        print(f"  {label}: {bad} line time(s) far from the estimate, using the estimate there", flush=True)
        st = [a if abs(a - b) <= limit else b for a, b in zip(st, est)]
        st = [max(st[: k + 1]) for k in range(len(st))]  # keep them in order
    return st


def split(pcm, texts, floor=400, min_gap_ms=120, starts=None):
    """cut one recording of several lines into one piece per line, at pauses. With `starts` (from line_starts) each
    cut goes to the pause just before the line's reported start; without, to where the line's share of the
    speaking time says it should be. Longer pauses are preferred either way."""
    n = len(texts)
    if n == 1:
        return [pcm], 0.0
    hop = RATE // 100  # 10 ms
    quiet = [max(abs(x) for x in pcm[i : i + hop]) <= floor for i in range(0, len(pcm), hop)]
    gaps, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * 10 >= min_gap_ms and i > 0 and j < len(quiet):
                gaps.append(((i + j) // 2 * hop, (j - i) * 10))
            i = j
        else:
            i += 1
    if len(gaps) < n - 1:
        raise ValueError(f"only {len(gaps)} pause(s) for {n} lines")
    ws = [weight(t) for t in texts]
    total = len(pcm)
    if starts:
        # the model reports starts a little late; the pause before a line ends just before it
        expect = [(starts[k + 1] - 0.3) * RATE for k in range(n - 1)]
    else:
        expect = [total * sum(ws[: k + 1]) / sum(ws) for k in range(n - 1)]
    longest = max(g[1] for g in gaps)
    # dynamic programming over (cut number, gap index)
    INF = float("inf")
    cost = [[INF] * len(gaps) for _ in range(n - 1)]
    back = [[-1] * len(gaps) for _ in range(n - 1)]
    for k in range(n - 1):
        for g, (pos, ms) in enumerate(gaps):
            c = ((pos - expect[k]) / RATE) ** 2 + (1 - ms / longest) * 0.3 if starts else ((pos - expect[k]) / total) ** 2 * 40 + (1 - ms / longest) * 0.5
            if k == 0:
                cost[k][g] = c
            else:
                best = min(range(g), key=lambda h: cost[k - 1][h], default=None)
                if best is not None and cost[k - 1][best] < INF:
                    cost[k][g] = cost[k - 1][best] + c
                    back[k][g] = best
    g = min(range(len(gaps)), key=lambda h: cost[n - 2][h])
    if cost[n - 2][g] == INF:
        raise ValueError("could not place the cuts")
    cuts = []
    for k in range(n - 2, -1, -1):
        cuts.append(gaps[g][0])
        g = back[k][g]
    cuts = [0] + cuts[::-1] + [total]
    pieces = [pcm[cuts[k] : cuts[k + 1]] for k in range(n)]
    # how far the pieces are from the expected shares (0 = exactly as expected)
    worst = max(abs((cuts[k + 1] - cuts[k]) / total - ws[k] / sum(ws)) / (ws[k] / sum(ws)) for k in range(n))
    return pieces, worst


def write_wav(path, pcm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path + ".part", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    os.replace(path + ".part", path)


SECONDS_PER_WEIGHT = 0.27  # measured on the KNN clips; about the same in all three languages


def save_take(lang, lines, blob, label):
    """split one recording of several lines into their clips"""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(blob)
        f.flush()
        pcm = read_pcm(f.name)
    # keep the whole take: if a cut turns out wrong it can be cut again without a new request (recut)
    write_wav(take_path(lang, lines), pcm)
    cut_take(lang, lines, pcm, label)


def take_path(lang, lines):
    key = hashlib.sha1("|".join(os.path.basename(clip_path(lang, l)) for l in lines).encode()).hexdigest()[:16]
    return os.path.join(WORK, "takes", lang, f"{key}.wav")


def cut_take(lang, lines, pcm, label):
    texts = [say_of(l) for l in lines]
    starts = checked_starts(pcm, texts, lang, label) if len(texts) > 1 else None
    pieces, worst = split(pcm, texts, starts=starts)
    for line, piece in zip(lines, pieces):
        write_wav(clip_path(lang, line), trim(piece))
    flag = "  !! check the cuts" if worst > 0.45 else ""
    print(f"  {label}: {len(pcm) / RATE:.1f}s → " + " + ".join(f"{len(trim(p)) / RATE:.1f}" for p in pieces) + f" (off by up to {worst:.0%}){flag}", flush=True)


def chunks(scenes, seconds):
    """group consecutive scenes into takes of about `seconds` of speech (a scene is never split across takes)"""
    out, cur, t = [], [], 0.0
    for scene, lines in scenes:
        est = sum(weight(say_of(l)) for l in lines) * SECONDS_PER_WEIGHT
        if cur and t + est > seconds:
            out.append(cur)
            cur, t = [], 0.0
        cur.append((scene, lines))
        t += est
    if cur:
        out.append(cur)
    return out


def cmd_recut(args):
    """cut the saved takes again (after a change to split()), without new requests"""
    for ep in episodes(args):
        script = load_script(ep)
        for lang in args.langs:
            for group in chunks(list(script[lang].items()), args.chunk):
                lines = [l for _, ls in group for l in ls]
                path = take_path(lang, lines)
                if os.path.exists(path):
                    cut_take(lang, lines, read_pcm(path), f"{ep}/{lang}/" + "+".join(sc for sc, _ in group))


def cmd_gen(args):
    """scenes whose lines are all missing are recorded together, about --chunk seconds per request, and cut at the
    pauses: far fewer requests than one per line, and the intonation runs on naturally. If a take cannot be cut,
    its scenes are recorded one by one, then line by line. A line edited later is recorded on its own."""
    last = [0.0]

    def pace():
        time.sleep(max(0.0, MIN_SPACING_S - (time.time() - last[0])))
        last[0] = time.time()

    def take(lang, group):
        lines = [l for _, ls in group for l in ls]
        label = "+".join(sc for sc, _ in group)
        if len(lines) > 1 or len(group) > 1:
            pace()
            try:
                save_take(lang, lines, tts("\n\n".join(say_of(l) for l in lines), lang), label)
                return
            except ValueError as e:
                print(f"  {label}: {e}", flush=True)
            if len(group) > 1:
                for g in group:
                    take(lang, [g])
                return
        for line in lines:
            pace()
            out = clip_path(lang, line)
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                f.write(tts(say_of(line), lang))
                f.flush()
                write_wav(out, trim(read_pcm(f.name)))
            print(f"  {label} (one line): {clip_ms(out) / 1000:.1f}s", flush=True)

    for ep in episodes(args):
        script = load_script(ep)
        for lang in args.langs:
            scenes = list(script[lang].items())
            whole = [(sc, ls) for sc, ls in scenes if all(not os.path.exists(clip_path(lang, l)) for l in ls)]
            part = [(sc, [l for l in ls if not os.path.exists(clip_path(lang, l))]) for sc, ls in scenes if (sc, ls) not in whole]
            part = [(sc, ls) for sc, ls in part if ls]
            groups = chunks(whole, args.chunk) if args.chunk > 0 else [[w] for w in whole]
            print(f"{ep}/{lang}: {len(groups)} take(s) for {len(whole)} scene(s), {sum(len(ls) for _, ls in part)} single line(s)", flush=True)
            for g in groups:
                take(lang, g)
            for sc, ls in part:
                for l in ls:
                    take(lang, [(sc, [l])])


# ---------------------------------------------------------------- build


def timings():
    r = subprocess.run(["node", os.path.join(ROOT, "scripts", "tts", "timings.mjs")], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def plan(scenes, script_lang, lang):
    """per scene, the length of every caption's segment: lead + line + gap, never shorter than the original segment
    (so no animation runs faster than designed). The player warps each segment onto its original stretch of the scene."""
    out = []
    for s in scenes:
        lines = script_lang.get(s["id"])
        if lines is None or len(lines) != len(s["cues"]):
            sys.exit(f"{lang}/{s['id']}: script has {None if lines is None else len(lines)} line(s), the scene has {len(s['cues'])} caption(s)")
        bounds = list(s["cues"]) + [1]
        segs = []
        for k, line in enumerate(lines):
            path = clip_path(lang, line)
            if not os.path.exists(path):
                sys.exit(f"missing clip for {lang}/{s['id']}/c{k + 1}: run gen first")
            lead = LEAD_MS if k == 0 else 0
            need = lead + clip_ms(path) + GAP_MS
            orig = round((bounds[k + 1] - bounds[k]) * s["dur"])
            segs.append({"ms": max(need, orig), "lead": lead, "clip": path, "text": text_of(line)})
        out.append({"id": s["id"], "segs": segs})
    return out


def cmd_build(args):
    tl = timings()
    for ep in episodes(args):
        script = load_script(ep)
        if ep not in tl:
            sys.exit(f"{ep} is not an episode in src/site/watch/episodes")
        for lang in args.langs:
            if lang not in script:
                print(f"{ep}/{lang}: no script, skipped")
                continue
            p = plan(tl[ep], script[lang], lang)
            pcm = array.array("h")
            for s in p:
                for g in s["segs"]:
                    clip = trim(read_pcm(g["clip"]))
                    pcm.extend(array.array("h", bytes(2 * (g["lead"] * RATE // 1000))))
                    pcm.extend(clip)
                    pad = g["ms"] * RATE // 1000 - g["lead"] * RATE // 1000 - len(clip)
                    pcm.extend(array.array("h", bytes(2 * pad)))
            os.makedirs(os.path.join(OUT, ep), exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                wav, m4a = os.path.join(tmp, "a.wav"), os.path.join(tmp, "a.m4a")
                with wave.open(wav, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(RATE)
                    w.writeframes(pcm.tobytes())
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-ac", "1", "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart", m4a], check=True)
                with open(m4a, "rb") as f:
                    blob = f.read()
            name = f"{lang}.{hashlib.sha256(blob).hexdigest()[:10]}.m4a"
            for old in glob.glob(os.path.join(OUT, ep, f"{lang}.*.m4a")):
                if os.path.basename(old) != name:
                    os.remove(old)
            with open(os.path.join(OUT, ep, name), "wb") as f:
                f.write(blob)
            manifest = {
                "v": 1,
                "episode": ep,
                "lang": lang,
                "audio": name,
                "scenes": [{"id": s["id"], "seg": [g["ms"] for g in s["segs"]], "lines": [g["text"] for g in s["segs"]]} for s in p],
            }
            with open(os.path.join(OUT, ep, f"{lang}.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, separators=(",", ":"))
            total = sum(g["ms"] for s in p for g in s["segs"])
            orig = sum(s["dur"] for s in tl[ep])
            print(f"{ep}/{lang}: {total / 1000:.0f}s (video {orig / 1000:.0f}s), {len(blob) / 1e6:.2f} MB → {name}")


# ---------------------------------------------------------------- qa


def qa_scene(lang, scene, lines):
    """listen to one scene (its clips joined) and compare every line with its script; cached by the clips' content"""
    paths = [clip_path(lang, l) for l in lines]
    h = hashlib.sha1()
    for p in paths:  # by content: a clip cut again is listened to again
        with open(p, "rb") as f:
            h.update(f.read())
    cache = os.path.join(QA, f"{h.hexdigest()[:16]}.{lang}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    pcm = array.array("h")
    for p in paths:
        pcm.extend(trim(read_pcm(p)))
        pcm.extend(array.array("h", bytes(2 * (RATE * 7 // 10))))
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "s.wav")
        write_wav(wav, pcm)
        with open(wav, "rb") as f:
            audio = base64.b64encode(f.read()).decode()
    numbered = "\n".join(f"{k + 1}. {say_of(l)}" for k, l in enumerate(lines))
    prompt = (
        f"The audio reads these {len(lines)} lines in order, with a pause between lines. Expected language: {NAMES[lang]}.\n\n{numbered}\n\n"
        "For every line, transcribe what is actually said and list real problems: words skipped or added (other than filler words and "
        "sentence-final particles such as 啊, 呢, 嘅, 喎, 埋, 嗱, um), words misread or mispronounced, wrong tones that change the word, numbers "
        "or letters read wrongly, symbols read aloud, wrong language. Ignore punctuation and Chinese character variants that sound the same. "
        'Return JSON {"lines": [{"n": 1, "transcript": "...", "issues": ["..."]}, ...]}. Be strict but do not invent issues.'
    )
    body = {"contents": [{"parts": [{"inlineData": {"mimeType": "audio/wav", "data": audio}}, {"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    res = post(f"https://generativelanguage.googleapis.com/v1beta/models/{LISTENER}:generateContent", body)
    try:
        r = json.loads(res["candidates"][0]["content"]["parts"][-1]["text"])["lines"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False)
    return r


def cmd_qa(args):
    import concurrent.futures

    os.makedirs(QA, exist_ok=True)
    jobs = []
    for ep in episodes(args):
        script = load_script(ep)
        for lang in args.langs:
            for scene, lines in script[lang].items():
                if all(os.path.exists(clip_path(lang, l)) for l in lines):
                    jobs.append((ep, lang, scene, lines))
                else:
                    print(f"-- {ep}/{lang}/{scene}: not recorded yet")
    bad = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(qa_scene, lang, scene, lines): (ep, lang, scene, lines) for ep, lang, scene, lines in jobs}
        for fut in concurrent.futures.as_completed(futures):
            ep, lang, scene, lines = futures[fut]
            r = fut.result()
            if r is None:
                print(f"?? {ep}/{lang}/{scene}: unreadable answer", flush=True)
                continue
            for item in r:
                k = item.get("n", 0) - 1
                if item.get("issues") or args.verbose:
                    bad += bool(item.get("issues"))
                    say = say_of(lines[k]) if 0 <= k < len(lines) else "?"
                    print(f"{'!!' if item.get('issues') else 'OK'} {ep}/{lang}/{scene}.c{k + 1}: {item.get('issues')}\n   script: {say}\n   heard:  {item.get('transcript')}", flush=True)
    print(f"{len(jobs)} scene(s) checked, {bad} line(s) flagged")


def main():
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen", "recut", "build", "qa"])
    ap.add_argument("ids", nargs="+", help="episode ids, or all")
    ap.add_argument("--lang", default=",".join(LANGS))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--chunk", type=float, default=330, help="gen: seconds of speech per request (0: one scene per request)")
    ap.add_argument("--model", default=MODEL, help=f"TTS model (default {MODEL}); clips are cached per model")
    ap.add_argument("--jobs", type=int, default=4, help="qa: scenes checked in parallel")
    args = ap.parse_args()
    args.langs = [x for x in args.lang.split(",") if x]
    MODEL = args.model
    if any(x not in LANGS for x in args.langs):
        sys.exit(f"--lang takes {LANGS}")
    {"gen": cmd_gen, "recut": cmd_recut, "build": cmd_build, "qa": cmd_qa}[args.cmd](args)


if __name__ == "__main__":
    main()
