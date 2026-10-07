# CS Helper (KnowitCS) — 开发日志

> 记录项目的关键进展、决策与变更。最新在上。
> 体例：每条含「日期 / 类型 / 摘要」，类型沿用 Conventional Commits（feat / fix / chore / docs / ci）。

---

## 进展时间线

### 2026-10-08 — 概念自测（/drill/）
- **feat**：42 道原创判断题（期末 Problem 1 题型），按讲次分组，可按期中 / 期末筛选、只看答错的题；作答后显示解释并链接到对应讲解视频或练习页；答题记录仅存本机。导航新增「自测」，首页增加入口；首页每个讲次列出该讲的讲解视频。
- **test**：`drill.test.ts` 校验每题都有文案与有效链接，并用库函数核对可计算的答案（广播形状、参数个数、输出尺寸、空洞卷积、密度 > 1、准确率 / 召回率、交叉验证 0%、α-β 剪枝）。

### 2026-10-08 — 讲解视频 15 集；按语言懒加载文案
- **feat**：新增「直方图与 Otsu 阈值」（像素倒入直方图、拉伸、阈值、Otsu 迭代；Final 2024 Q5(b) 原题数据，测试锁定 μ1 = 21、μ2 = 128、T = 74.5）与「期末的 AI 伦理题」（Final 2024 P9 三个方面、第 11 讲欧盟七原则 → 五个共同基础的官方配对、六种不公平成因、反馈循环）。
- **perf**：三份文案改为按语言分 chunk 懒加载（i18next 自定义 backend，加载完成后再渲染），主入口 chunk 749KB → 340KB，每页只多下载当前语言约 137KB。
- **test**：`locales.test.ts`（三语键一致）、`episodes.test.ts`（每集场景字幕数与 cues 对齐、停顿题存在），共 372 个测试通过。

### 2026-10-08 — 讲解视频第二批：共 13 集；画廊按讲次分组
- **feat**：新增 6 集 —— 感知机学习规则（第 5 讲 AND 表逐行）、模型评估（混淆矩阵 / 精确率 / 召回率 / F1 与不打乱交叉验证 0%）、广播与一行算两两距离（2022 春 Q2c）、连续特征朴素贝叶斯（第 2 讲血压 66 / 发烧 97，σ 用 n − 1）、空洞卷积（Final 2024 Q6，平移相加，与试卷输出逐一核对）、不用循环的图像处理（Final 2024 Q2，圆形掩码与展平卷积技巧）。
- **feat**：画廊按讲次分组并标注期中 / 期末范围；看完 90% 的视频在本机记为「已看」；首页讲解栏改为展示考频最高的三集。
- **lib/test**：`geom2d.ts`（直线 / 半平面裁剪，XOR 与感知机共用）、`conv2d.dilatedConv`（以 Final 2024 Q6 测试脚本输出为黄金值）、`imageNp.ts`（对应官方答案的掩码与展平卷积）；共 352 个测试通过。
- **勘误**：Final 2024 Q2 题面示例 `np.convolve([1,2,3,4],[0,1,0.5],'valid')` 印为 `[2.5, 3.5]`，按 NumPy 及题面自带的 `convolve_valid` 定义应为 `[2.5, 4.0]`，已写入测试与考点角。

### 2026-10-08 — 讲解视频（/watch/）：7 集 3Blue1Brown 风格动画讲解
- **feat**：新增「讲解视频」栏目。播放器按时间轴驱动（`src/lib/explainer.ts`，每一帧都是时间的纯函数，所以可拖动、跳章、变速，画廊缩略图也是同一套渲染），支持字幕、全屏、键盘操作，以及「停下来想一想」暂停点。每集一个页面：可点击的章节文字稿、考点角（要点 + 一道往年题与答案）、对应练习页链接。
- **内容**（按讲次）：贝叶斯与基础比率（10,000 人点阵、朴素贝叶斯零频与拉普拉斯平滑）、KNN 与标准化（第 3 讲 T 恤数据）、K-Means（分配 / 更新、SSE 单调、局部最优、第 4 讲药物表、肘部法）、XOR 与隐藏层（线性变换 + sigmoid 挤压 + 隐空间一条直线）、反向传播数值演算（第 6 讲第 1 轮第 1 步）、卷积到 CNN 形状表（Final 2022 网络 2.07M 对 MLP 150.5M）、α-β 剪枝（逐步追踪、两种剪枝、走法顺序）。画面上的数字全部由 `src/lib/` 中带测试的函数计算。
- **结构**：`src/site/watch/`（Player、stage 坐标系与配色、episodes/）；`sitemap.ts` 的 `EPISODES` 生成 `/watch/<id>/` 页面；模块页标题下出现「先看讲解」按钮，首页增加讲解视频栏。
- **test**：新增 `explainer`、`xorWarp`、`gradient`、`kmeansDemo` 测试及 `alphabeta` 的叶子计数用例，共 344 个通过。
- **勘误说明**：第 6 讲 w₅ 更新课件印 0.833245（先把 δk 舍入），不舍入为 0.833244，视频与答案两者并列。

### 2026-10-08 — 合并为多页站点：首页改为课程地图，10 个考点页三语化
- **feat（结构）**：主站由单文件 SPA 改为多页静态站。页面清单集中在 `src/lib/sitemap.ts`，`vite.config.js` 据此生成每页的 HTML 入口，所有页面共用 `src/site/main.tsx`。每个模块一个 URL（`/knn/`、`/bayes-virus/` …），笔记在 `/notes/<id>/`，拓展在 `/extend/<id>/`。首页采用 Lab 草图 A「课程地图」（按讲次、搜索、期中/期末范围、已看标记），B / C 草图移除。
- **feat（兼容）**：旧链接 `#/course/x`、`#/package/x`、`#/extend/x` 在首页自动跳到新 URL；`/lab/*.html` 构建为跳转页。
- **feat（i18n）**：10 个考点页、页面骨架与首页全部三语化（`site.*`、`lab.<page>.*`）。解析错误改为 `LabError` 错误码，文案在 `lab.errors.*`；`cnnShapes` 的层错误也改为错误码。
- **chore**：移除 `vite-plugin-singlefile`、`App.tsx`、Hash 路由（`route.ts`、`useHashRoute`）与 `vite.lab.config.js`；发布改为上传整个 `dist/`。

### 2026-10-08 — NumPy 面板：多项式；合并成对距离视频
- **merge**：合并 `worktree-pairwise-video`（成对距离考点视频，此前只部署未合入）。其解释器新增的 square / expand_dims / argsort 由 main 上的扩展 API 覆盖，保留 argsort 的语义（默认 axis=-1，None 展平）。
- **feat**：NumPy 面板新增「多项式」分组：`from numpy.polynomial import Polynomial`、求值 / 求导 / 积分 / 求根（含复根）/ 运算 / 拟合 / convert，以及 `np.polyfit` / `np.polyval`，结果画成曲线图。笔记第 3 节的代码块可一键在面板中运行。
- **test**：新增 `poly.test.ts`，`npapi.test.ts` 增加 10 个多项式用例（NumPy 2.2.6 真实输出），共 329 个通过。

### 2026-10-08 — NumPy 笔记右侧的 API 可视化面板
- **feat**：Package 的 NumPy 笔记右侧新增「NumPy 实时演示」面板：55 个 API 示例分 8 组，可搜索、可改代码即时运行；每次调用都画出输入与结果网格，悬停结果格子显示它由哪些输入格子得到（规约标出被合并的轴和被选中的最值，矩阵乘标出行与列，搬运类显示原样复制自哪里）。笔记中可演示的名字可点击跳转。整页在该笔记下放宽到 1440px。
- **feat（解释器）**：新增约 45 个 NumPy 函数 / 方法与 `np.linalg`、`np.random` 子模块；为函数、方法、运算符、比较、`.T`、下标读取记录 `CallTrace`，搬运类调用在元素编号数组上重放得到来源。
- **fix**：科学计数法输出保留 8 位有效数字（此前会打印 `2.6642920868471265e-01`，NumPy 为 `2.66429209e-01`）；负零显示为 `-0.`。
- **test**：新增 `npapi.test.ts`（70 例，NumPy 2.2.6 真实输出）与 `npTrace.test.ts`（16 例），共 305 个通过。

### 2026-10-07 — KnowCS Lab：10 个新模块页 + 3 个首页草图（/lab/）
- **feat**：新增独立多页原型 `/lab/`（React + TS + Tailwind，`vite.lab.config.js`），每个模块一个 HTML 页面，全宽布局、上一页/下一页导航、「Quiz me」把考试表格变成可判分的填空。10 个新页面对应历年试卷高频而主站缺失的考点：贝叶斯基率、Gaussian 朴素贝叶斯、混淆矩阵与 F1/MCC、D 折交叉验证陷阱、K-Means 手算表、感知机更新表、XOR 与反向传播、卷积填充与翻转、直方图与 Otsu、CNN 形状与参数。
- **feat**：三个首页草图供选择：A 课程地图、B 考点雷达（16 个题型 × 9 份试卷）、C 学习路径（自查清单 + 本机进度）。
- **决策**：题目数据改写数字后标注「Pattern of 20xx Qn」，不直接搬运试卷原题（试卷标注仅限校内学术使用）；讲义例题原样使用。
- **发现**：2022 期末 Part B Q1(c) 评分标准第 4 层输出尺寸有误（13×13，应为 14×14），页面中提示。
- **test**：新增 10 个 lib + `src/lab/format.ts` 共 58 个用例，以讲义与评分标准的数值为黄金值（如 5/68、0.0194、macro-F1 0.407、MCC 0.14、1,558,656 参数、反传 δk = 0.107022），共 219 个通过。
- **chore**：ESLint 忽略 `dist-lab`、`work`；`.gitignore` 加 `dist-lab`；部署时 release 目录同时包含 `index.html` 与 `lab/`。
### 2026-10-07 — 考点视频：一行 NumPy 求成对距离
- **feat**：NumPy 模块新增第 3 节「一行 NumPy 求成对距离（考点讲解视频）」（`NumpyPairwiseVideo.tsx`）。这是一段约 2 分 10 秒、共 9 个场景的动画视频：原题与评分标准 → 要算什么 → 一行拆解 → 直接相减为何报错 → 插入新轴 → 广播成 (n, m, d) → 平方并沿 axis=-1 求和（对比错误 axis）→ 开方与 argsort 取 KNN → 考前清单与内存代价。三语字幕，配有时间轴、章节、倍速和键盘控制。
- **feat**：`lib/pairwise.ts` 提供差张量、平方距离、循环版、展开式版、kNearest、形状追踪等纯函数，另加 9 个测试，用 2022S Q2(c) 的输出作为黄金值。
- **feat**：minipy 新增 `np.square`、`np.expand_dims`、`np.argsort` / `.argsort()`，输出已与真 NumPy 核对。
- **决策**：视频做成站内时间轴动画，不嵌 MP4，原因是单文件产物、字幕可三语切换、画面数值由代码计算。另用 Playwright 录屏导出了中英两版 MP4，供站外分享，不入库。

### 2026-10-05 — Package / Extend 内容三语化
- **fix（i18n）**：此前资料包 5 篇笔记与 Attention 拓展页只有简体，COMP2211 课程笔记只有英文，与站点语言无关。现在每项内容都有 en / zh 两份源文件，zh-HK 在构建期由简体生成，切换语言即时生效。
- **feat**：新增英文版 NumPy / pandas / PyTorch / TensorFlow / Keras 笔记和 Attention 页，新增中文版 COMP2211 课程笔记。英文课程笔记中夹杂的中文注释改为英文。清理笔记中会原样显示的 Obsidian 语法（`> [!tip]`、`[[numpy]]`）。
- **chore**：简→港繁转换抽出为 `scripts/zh-hk.mjs`（`zh-HK.json` 产物逐字不变）。新增 Vite 查询 `?html-hk` / `?raw-hk`。源文件按语言重命名（`keres.md` → `keras.zh.md` 等）。
- **test**：新增 `src/content/content.test.ts`（26 个用例），共 161 个通过。17 张配图仍只内联一次。产物 gzip 增加约 78KB（两份新语言内容）。

### 2026-10-05 — 广播模块重构为交互实验台
- **feat**：NumPy 模块第 2 节由静态 `(3,1)+(1,4)` 图示改为**广播实验台**：自选形状与运算符、6 个场景预设、逐轴规则表（可逐步演示）、实心真实元素 / 虚线虚拟副本的拉伸可视化（悬停联动 + 算式）、stride 0 内存说明、不兼容时的冲突轴与一键修复（`B[:, None]`）、站内解释器实际运行的等价代码，以及 4 张知识卡片（规则 / 为何快 / COMP2211 用例 / `(n,)` vs `(n,1)` 陷阱）。
- **feat**：新增 `lib/broadcast.ts`（`planBroadcast` / `sourceIndex` / `isRealCell` / `broadcastStrides` / `suggestFix`）及 11 个用例；文案迁至 `numpy_module.broadcast.*`（三语）。共 135 个测试通过，产物 gzip +10KB。

### 2026-10-05 — NumPy 索引实验台（可运行代码）
- **feat**：NumPy 模块第 1 节由「4×4 三按钮」改为可编辑代码的**索引实验台**。代码在浏览器内运行，每次下标读写都可视化：源→结果格子映射与悬停联动、View/Copy 判定与原因、逐轴解释、掩码网格、底层一维缓冲区（offset/strides）、写穿视图时连带修改的变量告警。附 8 个预设示例。
- **决策**：不用 Pyodide（约 10MB、国内 CDN 慢、拿不到元素溯源），改为自研 TS 迷你 NumPy（`lib/ndarray.ts`）+ Python 子集解释器（`lib/minipy.ts`），产物约 +20KB gzip。与本机 NumPy 2.2 做差分：95 个片段中 90 个的输出与报错逐字一致。
- **chore**：移除 `lib/numpy.ts`、`SliceType`、`INITIAL_MATRIX` 与旧文案键；新增 `numpy_module.playground.*` 三语文案（zh-HK 由脚本生成）。
- **test**：新增 ndarray / minipy 用例（含「arange 源上结果值 == 溯源下标」性质测试），共 124 个通过。

### 2026-10-05 — 迁移至 knowcs.online
- **fix**：损坏的 Hash URI（如 `#/course/%`）安全回退，避免 `decodeURIComponent` 抛错导致全站白屏；补充回归用例，94 个测试通过。
- **ci**：移除 PinMe 自动发布，GitHub Actions 只保留构建门禁；新增 `deploy/` 下的 OpenResty 配置与服务器发布/回滚说明。
- **deploy**：`knowcs.online` 根域名 A → `129.226.210.66`（DNSPod RecordId `2424635031`，TTL 600），`ssh pastpaper` 对应 `lhins-ir66ks5j`；新增独立 OpenResty 站点 `/opt/1panel/www/sites/knowcs`，发布 `20261005-2158`，保留配置备份。Let’s Encrypt 证书签发，安装 webroot 自动续期钩子，Certbot 模拟续期成功。HTTPS 200、HTTP 301、gzip、线上/本地 SHA-256 一致；现有 Minecraft/Pastpaper 站点均仍返回 200。
- **test（线上）**：真实 Chrome 在 1440px/390px 下覆盖 36 次模块/笔记/拓展/异常路由检查，控制台错误和未捕获异常均为 0；异常 Hash 刷新也安全回退。
- **docs**：READMEs / CLAUDE / design / plan 的生产入口改为 `https://knowcs.online`。

### 2026-10-05 — 全面审查修复与性能优化（v2.1.0）
- **perf**：单文件产物 4.55MB → 1.99MB（gzip 2.46MB → 1.07MB）。KaTeX 只内联 woff2 且去掉未用字族（60 → 11 个字体）；笔记配图由原始 PNG 重新编码为 WebP；移除 Recharts（自研 SVG `LineChart`）与 react-markdown（Markdown 构建期预渲染，`scripts/vite-plugins.mjs`）；Framer Motion 改 `LazyMotion` + `m.*`。
- **fix（教学正确性）**：朴素贝叶斯 log 模式不再用 `1e-10` 掩盖 `log 0`，如实显示 −∞；类别样本数由计数表推出；α 输入 clamp；log/连乘切换现在真正显示分数（此前切换无可见效果）。KNN 平票改由最近邻裁决并提示（原先恒判 M）。火警案例 P(Smoke) 受全概率下界约束（原先可算出 2000% 的后验）。
- **fix（渲染）**：NB 寄语「条件独立」被渲染成 α（`<Trans>` 占位错位）；KNN 寄语出现字面 `**欠拟合**`；大写标题把 α/δ 变成 Α/Δ（`.katex { text-transform:none }`）；KNN 画布缩放后点击位置偏移；卷积核输入 `-` / 清空时输出 NaN。
- **feat**：Backprop 模块重写——6 个参数滑块、四步推导全部代入实时数值、反向动画揭示 Δw、「执行更新」+ 误差下降曲线（`lib/backprop.ts`，数值梯度校验）。KNN「K vs 误差」改为真实留一法曲线（`looErrorCurve`），标出最优 K。
- **feat**：Hash 路由（`#/course/knn`、`#/package/numpy`…），支持刷新 / 后退 / 分享；`App.tsx` 改为 `COURSE_TABS` 注册表驱动；侧栏副标题 i18n 化；移动端横向模块条；页脚去掉虚构的「Latency: 12ms」；版本号取自 package.json。
- **feat（i18n）**：修复语言检测被 `lng:'en'` 覆盖而失效；检测顺序 `?lang=` > localStorage > 浏览器，zh-TW/zh-Hant → 繁体；生产环境关闭 i18next debug；同步 `<html lang>`。
- **feat（a11y）**：跳到内容链接、`aria-current` / `aria-pressed` / `aria-live`、语言菜单键盘操作、画布方向键操作、图表 `<title>`、尊重「减少动态效果」。
- **chore**：移除误装依赖 `rechart@0.0.1` 与未用的 `autoprefixer`；`@tailwindcss/postcss` 移入 devDependencies；删除模板残留（`vite.svg` / `react.svg` / `App.css`）；`dist/index.html` 移出版本库。
- **ci**：Node 20 → 22；`pinme` 锁定 2.0.10。
- **test**：59 → 93 个用例（新增 backprop / chart / route / lang，补 KNN 平票与 LOO、NB −∞ 与 classTotals、minEvidence）。
- **docs**：`design.md` 按现状重写（架构 / 路由 / 构建优化 / 扩展指引）。

### 2026-06-07 — 三栏顶层导航：Course / Package / Extend
- **feat**：Header 重构——移除环境状态框，新增三栏分段切换（Course 课程 / Package 资料包 / Extend 拓展，当前项蓝色胶囊高亮）。
- **feat**：**Package 资料包**——6 篇 Markdown 笔记（NumPy/pandas/PyTorch/TensorFlow/Keras/COMP2211 课程笔记）3×2 卡片网格，点开 react-markdown + remark-gfm 渲染（表格/代码/图片），返回按钮回网格；笔记与 17 张配图经 `?raw`/glob 构建期内联，兼容单文件打包。
- **feat**：**Extend 拓展**——「Attention Is All You Need 可视化解读」自包含 HTML 经 iframe(srcDoc, sandbox) 隔离渲染。
- **feat**：卡片标题/描述走 i18n 三语；新增 `src/vite-env.d.ts`（vite/client 类型）。
- **chore**：移植 Perceptron / PyTorch / autograd 模块代码与测试入仓（lib 测试已跑通，UI 暂不接入侧栏，待后续上架）；测试 49 → 59。

### 2026-06-07 — K-Means 聚类模块上线（第 8 个模块，L4）
- **feat**：从 `plan-features` worktree 移植 `KMeansModule.tsx` + `lib/kmeans.ts`（+9 个 Vitest 用例）：三簇散点 EM 迭代动画（播放/单步/重置）、K=2-5 选择、WCSS 实时显示、Elbow Method 曲线、Raw/Z-score 切换；接入 Tab 路由、三语文案与专属 Exam Tip（WCSS 单调不增 + 肘部法则）。
- 覆盖表最大缺口（L4）与期中高频手推考点补齐，plan.md P1 完成。

### 2026-06-07 — Alpha-Beta 剪枝模块上线（第 7 个模块，L10）
- **feat**：从 `plan-features` worktree 移植 `AlphaBetaModule.tsx` + `lib/alphabeta.ts`（+6 个 Vitest 用例）：3 层博弈树 SVG 可视化、DFS + α/β 剪枝逐步追踪（播放/步进/调速）、叶子值沙盒与三种预设、i18n 化步骤解释；接入 Tab 路由、三语文案与专属 Exam Tip。
- **chore**：tsconfig target/lib 升至 ES2022（`Array.prototype.at` 需要）。
- **决策**：worktree 中其余 6 个新模块（KMeans/感知机/PyTorch/归一化/Stump/复习自测）暂不合并，后续逐个移植以控制风险。

### 2026-06-07 — 侧栏 Exam Tip 模块化
- **feat**：侧栏 Exam Tip 由全局一条（Broadcasting）改为随当前模块切换，6 条各自贴切的考试提示（NumPy 广播 / Backprop 误差反向与 Δw / Kernel 输出尺寸 N−K+1 / Bayes 后验∝先验×似然 / NB 零频率与 log / KNN 奇数 K 与标准化），i18n 三语同步（zh-HK 经脚本重新生成）。

### 2026-06-07 — 单元测试体系（Vitest）
- **feat**：引入 Vitest，新增 `src/lib/`（knn / kernel / bayes / numpy）——把组件内嵌的计算逻辑原样提取为纯函数，组件改为调用；34 个用例覆盖黄金值（NB α=0 → 20.46%/79.54%、火警 9.00%、KNN 默认点 M 4:1、Laplacian 200）与边界（零频率、α 平滑、log/连乘一致性、clamp、平票）。
- **ci**：deploy.yml 在 lint 后插入 `npm test`，部署门禁三连 → 四连；CLAUDE.md 同步更新。
- **决策**：Backprop 模块无计算逻辑（公式为静态展示），不提取；测试范围由 `vitest.config.ts` 限定在 `src/**`，避免误跑 `.claude/worktrees/` 下的文件。

### 2026-06-07 — 繁体中文（zh-HK）支持
- **feat**：i18n 新增香港繁体 `zh-HK`，Header 语言切换由双语 toggle 升级为三语下拉菜单（English / 简体中文 / 繁體中文，Framer Motion 动画 + 点击外部收起）。
- **feat**：新增 `scripts/gen-zh-hk.mjs`（`npm run gen:zh-hk`）——OpenCC `cn→hk` 字级转换后套用香港术语映射（內存→記憶體、算法→演算法、交互→互動、創建→建立、噪聲→雜訊、過濾器→濾波器、學長寄語→師兄寄語）；`zh-HK.json` 为生成产物，严禁手改。
- **决策**：繁体走「脚本生成」而非手工第三份维护，避免三份文案同步负担；保留「概率/數據/網絡/過擬合」等香港本就通用的写法，不做台化。i18next fallback 链 zh-HK → zh → en 兜底漏译。

### 2026-05-31 — 文档体系搭建
- **docs**：整理文档目录，将 `design.md` 归入 `docs/`，新增 `log.md`（本文件）与 `plan.md`。
- **docs**：根目录新增 `CLAUDE.md`，统一索引 `docs/` 下的 设计 / 日志 / 计划 三份文档，作为 AI 协作与新成员上手的入口。

### 部署与质量基线
- **fix**（`689455e`）：修复全部 lint 报错，并把 `npm run lint` 设为部署门禁——lint 不过不允许部署。
- **ci**（`3aaf1c3`）：新增 GitHub Actions 工作流，自动构建并部署到 PinMe（`knowcs.pinme.dev`）。

### 可视化模块迭代
- **feat**（`f36cda0`）：新增 **KNN 可视化模块**，含 i18n 支持（点击画布设测试点、K 值投票、Z-score 标准化开关、模型复杂度曲线）。
- **chore**（`5c963d3`）：更新本地化文案。
- **feat**（`864ae9a`）：接入 **贝叶斯模块**（贝叶斯基础 Basics + 朴素贝叶斯 Naive），含 i18n 支持。

### 架构与国际化
- **feat**（`9a09422`）：恢复 V1 布局并加上 i18n 支持，同步更新文档。
- **docs**（`bfda082`）：补充英文 README，实现中英双语支持。
- **docs**（`3f6f425`）：新增 README，说明项目信息与技术栈。
- **feat**（`835e1b7`）：加入 i18n 支持，重建单文件 HTML 产物。

### 项目初始化
- **chore**（`3ca7917`）：Initial commit，搭建 React 19 + Vite 7 + Tailwind 4 单页骨架。

---

## 已知问题 / 技术债

> 与 [design.md](./design.md) 第 12 节「已知约束」呼应，落地为可追踪条目。

- [x] ~~`App.jsx` 约 1065 行，所有模块与外壳耦合在单文件~~——已拆分为 `src/modules/` 六模块 + 薄壳结构。
- [x] ~~纯计算逻辑（距离、卷积、贝叶斯）无单元测试~~——已抽取到 `src/lib/` 并配 Vitest 用例（2026-06-07 起，现 94 个），`npm test` 为部署门禁；UI/交互层测试仍缺。
- [x] ~~`en.json` / `zh.json` 双份手工维护，缺键一致性校验~~——`src/lib/locales.test.ts` 校验三份文案键完全一致，`src/site/watch/episodes.test.ts` 校验讲解视频字幕条数与场景 cues 对齐（2026-10-08）。
- [x] ~~KNN「K vs 误差」曲线为示意性合成数据~~——已改为真实留一法误差（2026-10-05）。
- [ ] 教学数据（`BAYES_DATA`、`KNN_RAW_DATA`）硬编码，与讲义绑定。

---

## 维护约定

- 每次合并到 `main` 的有意义变更，在「进展时间线」追加一条（最新在上）。
- 引入新约束或填掉技术债时，同步更新「已知问题」清单。
- 涉及架构/技术栈的决策写入 [design.md](./design.md)；未来规划写入 [plan.md](./plan.md)。
