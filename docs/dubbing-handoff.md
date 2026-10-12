# 交接：讲解视频（/watch/）三语配音

> 2026-10-12 写，同日第二个 session 更新：播放器已接入，KNN 已上线，17 集讲稿已写好；其余音频受每日 100 次限额约束，按第 4 节分批生成。

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

## 3. 现状（2026-10-12 第二个 session 之后）

已接入播放器并上线 KNN 一集；其余 16 集的讲稿已写好，音频受每日限额约束，正在分批生成。

**用户已确认的决定**：配音开启时字幕显示讲稿（粤语为口语稿，en/zh 仍显示站点原字幕，只有朗读文本不同）；默认开，控制栏有开关（M 键），选择记在 `knowcs-voice`；音频**不进 Git**，放服务器共享目录，每个 release 软链接过去；先上线 KNN 验证，其余后台生成。

| 部分 | 位置 |
|---|---|
| 讲稿（17 集 × 3 语，入库） | `src/content/narration/<ep>.json`：`{lang: {scene: [每条字幕一句]}}`；一句可写成 `{"text": 字幕, "say": 朗读文本}` |
| 生成 / 构建 / 听辨 | `scripts/tts/narrate.py gen|build|qa <ep…|all> [--lang] [--per-line]`；`scripts/tts/timings.mjs` 导出各场景 dur/cues |
| 片段缓存（不入库） | `work/dubbing/clips/<lang>/<sha1(模型|音色|style|朗读文本)>.wav`，改一句只重录那一句 |
| 构建产物（不入库） | `work/dubbing/out/audio/<ep>/<lang>.json` + `<lang>.<hash>.m4a`（AAC 48k 单声道，约 1.2 MB / 3 分钟） |
| 播放器 | `src/site/watch/narration.ts`（读 manifest、重排时间、`remapTime`）、`Player.tsx`（音频即时钟、倍速、拖动、ponder、开关）、`WatchPage.tsx` |
| 本地预览 | `vite.config.js` 的 `narrationAudio()` 在 dev/preview 下从 `work/dubbing/out/audio` 提供 `/audio/`（支持 Range） |
| 测试 | `src/site/watch/narration.test.ts`：每条字幕一句、en/zh 字幕与站点一致、朗读文本不含 `= × − ² σ ₁ [ ]` 等难读符号 |

**时间对齐**：每条字幕的段长 = 300 ms（场景首句）+ 片段时长（去掉首尾静音）+ 450 ms，且不短于原视频该段；播放器用分段线性映射把每段动画对到原来的 cue 区间（`narratedTiming` / `piecewise`），所以每句话说的时候画面正演到它对应的部分。manifest 与代码里场景 id 或字幕条数对不上时，该集自动退回无声原版。

**生成方式**：`gen` 把连续几个整场景都缺的部分合成一次请求（`--chunk`，默认 330 秒语音，即一集一种语言一次请求，约 5,000–7,500 个输出 token，上限 16,384；各句用空行隔开），每段整段录音存到 `work/dubbing/takes/`，再切成每条字幕一段。只看停顿分不出行间与句间（两者都在 0.5–0.9 秒），所以先请 gemini-3.8-flash 报出每行起始时间（两次误差 ≤ 0.3 秒），在其前方最近的较长停顿处下刀；拿不到时间才按字数比例切。切不开时先退回按场景，再退回逐句。`recut` 用存好的整段重切，不花 TTS 配额。`--model` 可换模型（片段按模型分开缓存）。

## 4. 配额（最大的限制）

Tier 1 的 `gemini-3.8-flash-tts` **每天 100 次请求**（429 提示 "limit: 100 requests per day"，约在香港时间早上重置）。按场景生成，16 集约 405 次，约 4–5 天；`gen` 碰到日限额会直接退出（不空耗重试）。批量命令（每天跑一次即可，已生成的会跳过）：

```
cd /Users/charles/repo/KnowCS && set -a && source .env && set +a
SSL_CERT_FILE=/etc/ssl/cert.pem python3 -I scripts/tts/narrate.py gen perceptron kmeans bayes xor backprop cnn evaluate broadcast gaussian imagenp otsu affine alphabeta dilated autograd ethics
SSL_CERT_FILE=/etc/ssl/cert.pem python3 -I scripts/tts/narrate.py qa  <同上>
python3 -I scripts/tts/narrate.py build <完成的集>
```

升级到更高 tier 可以一次做完。听辨用的 `gemini-3.8-flash` 是另一份配额。

## 5. 已知问题

- KNN 讲稿已按上一版 QA 修正：坐标改成「160 centimetres and 60 kilograms」，「the lecture's choice」改为「the choice used in the lecture」，粤语「欠擬合」改为「擬合不足」，「俾埋…你」改为「提供」。
- KNN 英文 boundary 第 3 句和 recap 两句来自上一版整场景录音的切分（当天配额用完），其余为逐句录音。
- KNN 上线版听辨（`qa knn`）：zh、en 全部通过；yue 7 处被标记，多为近义替换（入面→入邊、拉闊→拉開），可接受；「擬合」的「擬」两处读音不稳。粤语讲稿已改用课堂常说的 overfit / underfit（`尺碼`→`碼數`，gaussian 的「擬合一條曲線」→「fit 一條曲線」），这 3 句等配额重置后由批量任务重录，再 `build knn` 并上传。
- 2026-10-12 用 `gemini-3.8-flash-lite-tts` 试录 perceptron（3 语 × 2 段，额度与 flash 分开）：英文无误；普通话一处把 z 念成 c；粤语明显较差（「階躍」→「截躍」、「輸出」→「衰出」、「指向」→「指將」）。结论：继续用 flash。试录片段在 lite 的缓存里，不影响正式音轨。英文讲稿里「one, one」这类坐标会被念出 comma，已改成「one one」。
- 同日又用 lite 试录了 xor 英文版：一次请求 270 秒、32 句，全部切对，听辨无实质问题。因此改为每集一次请求，中英文 16 集共 32 次。
- 用户决定：先录完所有课的中文和英文，粤语之后再录（KNN 粤语那 3 句的重录也一并推后）。
- 批量任务：`work/dubbing/batch.sh`（nohup 后台，日志 `work/dubbing/batch.log`，每小时重试，完成后自动 `qa`（→ `qa-all.log`）和 `build`；停止：`pkill -f work/dubbing/batch.sh`）。上传到服务器仍需手动。
- ponder 停住时，下方字幕已是下一场景的第一句（原版行为，配音版沿用）。

## 6. 发布

见 `deploy/README.md` 的 “Narration audio”：本地 `build` 后把 `work/dubbing/out/audio/<ep>/` 传到服务器 `/opt/1panel/www/sites/knowcs/audio/<ep>/`；新 release 需要 `ln -s ../../audio releases/<ts>/audio`。只更新音频不需要发新 release。

## 7. 待定

- 全部 17 集生成、QA、上线后，考虑字幕跟读到句子级（目前一条字幕一段，recap 的三条要点只在朗读文本里）。

## 8. 其它

- 声音复刻（voice replication）可以用，但只能复刻本人的声音（需要录一段授权声明），而且支持的语言里没有粤语。这次没有用。
- 官方文档：<https://ai.google.dev/gemini-api/docs/speech-generation>（语音生成），<https://ai.google.dev/gemini-api/docs/voice-design>（音色设计）。
