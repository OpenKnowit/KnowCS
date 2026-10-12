# 交接：讲解视频（/watch/）三语配音

> 2026-10-12 写。上一个 session 已经验证了 Gemini TTS 可用，选定了音色，并用 KNN 一集做了整节试配。下一个 session 的任务是**把配音接入 /watch/ 播放器**，再推广到全部 17 集。

## 1. 已定决策（用户确认过，不要再改）

| 站点语言 | 配音语言 | 音色 | style（写在 `speech_metadata.style`） |
|---|---|---|---|
| `zh` | 普通话 | **Charon** | `a warm, clear university lecturer speaking standard Mandarin, unhurried` |
| `zh-HK` | 粤语 | **Iapetus** | `a warm, clear university lecturer speaking natural Hong Kong Cantonese, unhurried` |
| `en` | 英文 | **Charon** | `a warm, clear university lecturer, unhurried` |

- 模型：`gemini-3.8-flash-tts`。
- **模型自己加的口语助词（粤语的「埋」「呢」「嗱」，普通话的「啊」）不用管**，这是用户的原话决定。念错字和念错术语仍然要修，见第 5 节。

## 2. Gemini TTS 用法要点（踩过的坑）

- **Key**：仓库根目录的 `.env` 里的 `GOOGLE_GEMINI_API_KEY`，走 Gemini 开发者 API。这个 key 调 Vertex 会被拒绝（403）。不要打印或提交 key。
- **接口**：语气只能通过 **Interactions API** 设置，`generateContent` 没有 `speech_metadata`：

  ```
  POST https://generativelanguage.googleapis.com/v1beta/interactions
  x-goog-api-key: $GOOGLE_GEMINI_API_KEY
  {
    "model": "gemini-3.8-flash-tts",
    "input": [{ "type": "user_input", "content": [{
      "type": "text",
      "text": "<只放要念的正文>",
      "annotations": [{ "type": "speech_metadata", "style": "<见上表>" }]
    }]}],
    "generation_config": { "speech_config": [{ "voice": "Charon" }] }
  }
  ```

  返回 JSON 里找 base64 的 `data` 字段，参考 `work/dubbing/knn/gen.py` 的 `find()`。
- **`text` 是逐字稿**：写在里面的指令前缀（如「用老师语气说：」）会被原样念出来。`systemInstruction` 不支持，会返回 400 "Developer instruction is not enabled"。
- **语言靠文字自动识别**，没有语言参数。粤语稿要写成口语粤语，并在 style 里写 Cantonese。
- **返回的是完整 WAV**（24 kHz、单声道、16-bit，开头就是 `RIFF`），直接写文件，**不要再加 WAV 头**。
- **`<short pause>` 这类标签**不会被念出来，但模型可能在停顿处自己加一声「嗯」。想要干净的停顿，就用句号或换行断开。
- **限流**：当前账号是 Tier 1，TTS 每分钟 10 个请求。请求之间至少间隔约 7 秒，遇到 429 退避重试。
- **网络**：本机流量走 `127.0.0.1:7897` 代理，偶尔会出现空 404、SSL EOF、`WRONG_VERSION_NUMBER` 或超时。这些都可以重试，脚本必须对 `URLError`/`OSError` 也做重试，不能只处理 HTTP 错误。
- **Python**：用的是 python.org 版 3.10，没有根证书。运行时要加 `SSL_CERT_FILE=/etc/ssl/cert.pem`，否则会报 `CERTIFICATE_VERIFY_FAILED`。
- **耗时**：生成速度约为音频时长的一半（20 秒音频大约要等 10 秒）。
- `gemini-2.5-flash` 已不对新用户开放。要做听辨或转写，用 `gemini-3.8-flash`。

## 3. 现有成果：`work/dubbing/`

这个目录在 gitignore 里，只存在于本机。

```
work/dubbing/
  samples/           7 个候选音色 × 三语的试听样本（选音色时用的）
  knn/
    script.json      KNN 三语讲稿（按场景）+ 音色 + style + 视频原场景时长
    gen.py           按 script.json 逐场景生成：python3 -I gen.py [zh yue en]（已存在的文件会跳过）
    gen_voice.py     换音色重配一种语言：python3 -I gen_voice.py en Charon → 输出到 en-Charon/
    qa.py            用 gemini-3.8-flash 听辨每段并对照讲稿 → qa-results.txt
    zh/ yue/ en/     最终选定音色的逐场景 WAV（en/ 是 Charon）
    en-Sadaltager/ en-Iapetus/   落选的英文音色
    knn-{zh,yue,en}.m4a   整节拼接版（场景之间加 0.7 秒静音，AAC 96k，每个约 3 MB / 3 分钟）
```

运行方式：

```
cd /Users/charles/repo/KnowCS && set -a && source .env && set +a
SSL_CERT_FILE=/etc/ssl/cert.pem python3 -I work/dubbing/knn/gen.py zh
```

**讲稿来源**：
- **en 和 zh**：基本照用 `src/locales/{en,zh}.json` 里 `watch.knn.scenes.*.c*` 的字幕，只做了口语化改写：单位写全（cm → 厘米 / centimetres）、`k = 1` → 「k 等于 1」/「k equals 1」。
- **ponder 问题**：拼在 vote 场景的末尾念出来。
- **recap**：把 `watch.knn.recap.1–3` 三条要点也念了。
- **yue**：现有的 `zh-HK` 字幕是书面中文，所以重写了一份**口语粤语稿**（嘅、咗、啲、俾），不是照念字幕。

## 4. 关键发现：配音比视频长一倍

视频的场景时长是按「读字幕」的速度设计的。念出来以后，KNN 一集从 95 秒变成约 180 秒：

| 场景 | 原 dur | zh | yue | en（Charon） |
|---|---|---|---|---|
| title | 5 s | 11.3 | 9.5 | 11.7 |
| data | 12 s | 21.4 | 20.3 | 21.4 |
| distance | 15 s | 15.3 | 13.3 | 15.1 |
| vote | 17 s | 29.8 | 30.4 | 31.0 |
| scale | 19 s | 42.2 | 40.6 | 43.2 |
| boundary | 15 s | 28.1 | 28.5 | 26.2 |
| recap | 12 s | 26.9 | 31.2 | 32.0 |

好在所有场景的渲染都是进度 `p ∈ [0,1]` 的纯函数（`src/lib/explainer.ts`），**把某个场景的 `dur` 拉长，动画会自然放慢，不用改任何渲染代码**。

全部 17 集共 143 个场景、414 条字幕，视频原长约 33.5 分钟，配音后预计每种语言约 65 分钟。

## 5. 已知问题（接入前要修）

由 `qa.py` 听辨发现。听辨模型本身也会出错，下面三条我已单独复查过：

1. **英文把「160, 60」念成了「160 comma 60」**（distance 场景，Charon 和 Sadaltager 都这样）。坐标不要用逗号，改写成「160 centimetres and 60 kilograms」或「(160, 60)」→「160 by 60」。
2. **英文把「the lecture's choice」念成了「the lecturer's choice」**（vote 场景）。改写成「the choice used in the lecture」。
3. **粤语把「欠擬合」的「欠」(him3) 念成了「缺」(kyut3)**（recap 场景）。这是术语，必须修。可以试换说法（如「擬合不足」）重新生成，生成后再用 qa.py 复查。其他术语（過擬合、標準化）也建议抽查。
4. 听辨还报告粤语 scale 场景把「俾埋…你」念成了「…嚟嘅」，**没有复查过**，修完上面几条后人工听一下 `knn/yue/scale.wav`。

总体而言，zh 7 段全部正确；en 除上面两处外正确；yue 除上面两处外，其余差异都是加了助词（可以忽略）。

## 6. 接入播放器：现状和建议

**现状（相关代码）**：
- `src/site/watch/Player.tsx`：基于 `requestAnimationFrame` 的时钟（`tRef`），支持倍速（`SPEEDS`）、拖动（`seek`）、章节跳转（`chapter`）、ponder 暂停（`stopBetween`），字幕取 `capKey(ep, scene, cue)`。
- `src/lib/explainer.ts`：`TimedScene { dur, cues, ponder }`，`timeline()` 和 `locate()` 把时间换算成场景和进度。
- `src/site/watch/episodes/<id>.tsx`：每集的 `scenes` 数组，`dur` 和 `cues`（字幕切换点，场景内的进度比例）是写死的。
- `src/site/watch/useEpisode.ts`：按集懒加载代码和语言包；`src/lib/lang.ts` 定义站点语言为 `en` / `zh` / `zh-HK`。
- **静态资源**：Vite 的 root 是生成出来的 `site/`，`site/public/` 由 `vite.config.js` 第 113–142 行生成和拷贝（sw.js、图标等）。音频文件要走同样的拷贝流程，或者另找放置方式。`src/sw/sw.js` 是 service worker，要决定音频缓不缓存。

**建议方案**（下一个 session 可以调整）：
1. **按字幕（cue）粒度生成音频**，不按场景：每条字幕一个音频文件。好处是字幕和语音能精确对齐：场景 `dur` = 各条音频时长之和 + 留白，`cues` 由累计时长算出来。代价是请求数从 143×3 增加到 414×3（约 1,240 次，按每分钟 10 次限流约需 2 小时），而且句与句之间的语调会略微不连贯。只按场景生成就只能按字数比例估算字幕切换点。
2. **配音时长表做成构建产物**：生成脚本输出 `<ep>.<lang>.json`，里面是每条音频的毫秒时长。播放器加载后用它覆盖 `dur` 和 `cues`；没有配音的集保持原有时长。
3. **音频同步**：播放时按当前 cue 播放对应的音频；倍速时设 `audio.playbackRate = speed`；拖动、章节跳转、ponder 暂停、全屏都要同步暂停或定位音频。需要一个静音开关。没开配音的时候，行为要和现在完全一样。
4. **体积**：AAC 96k 下一集一种语言约 3 MB，17 集 × 3 种语言约 150 MB。建议用 Opus 或 AAC 单声道 32–48 kbps，人声足够清晰，体积可以降到四分之一左右。
5. **讲稿管理**：讲稿（尤其是和字幕不同的粤语口语稿）应该进版本库。建议放在 `src/content/narration/<ep>.json` 之类的位置，脚本放 `scripts/tts/`。`.env` 必须保持在 gitignore 里。

## 7. 需要问用户的问题

- **粤语模式下的字幕**：语音是口语粤语，而 `zh-HK` 字幕是书面中文，两者对不上。是把字幕换成口语稿、保留书面字幕，还是两种都提供？
- **默认是否自动播放配音**（还是用户点了才开），以及音频文件放在哪里（随 dist 发布，还是放 CDN 或对象存储）。
- 先只做 KNN 一集上线验证，还是 17 集一起做。

## 8. 其它

- 声音复刻（voice replication）可以用，但只能复刻本人的声音（需要录一段授权声明），而且支持的语言里没有粤语。这次没有用。
- 官方文档：<https://ai.google.dev/gemini-api/docs/speech-generation>（语音生成），<https://ai.google.dev/gemini-api/docs/voice-design>（音色设计）。
