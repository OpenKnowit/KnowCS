# CS Helper (KnowitCS) — 设计文档

> 面向 HKUST COMP2211（机器学习）的交互式学习网站。通过可视化帮助学生理解机器学习核心概念。

---

## 1. 项目概述

CS Helper 是一个**纯前端、单页**的交互式教学应用。它把机器学习课程中较难理解的抽象概念（内存切片、广播、反向传播、卷积、贝叶斯、KNN、Alpha-Beta、K-Means）拆解成一组**可交互的可视化模块**，让学生通过拖动滑块、点击画布、修改参数等方式实时观察算法行为。

设计目标：

- **可视化优先**：每个概念都配一张动态图 / 动画，而不是静态文字。
- **即时反馈**：所有计算在浏览器端实时完成，参数一改结果立刻更新。
- **数值真实**：图上的每个数字都由 `src/lib/` 的纯函数真实算出并有单元测试，不放示意性假数据。
- **三语**：English / 简体中文 / 繁體中文（港），自动检测浏览器语言。
- **零后端、易分发**：整站打包成**单个 HTML 文件**，可直接作为静态站点托管，或离线双击打开。

在线 Demo：<https://knowcs.online>

---

## 2. 技术栈

| 维度 | 选型 | 说明 |
|------|------|------|
| 框架 | **React 19** | 函数组件 + Hooks |
| 语言 | **TypeScript（strict）** | `tsc --noEmit` 为部署门禁 |
| 构建 | **Vite 7** | 开发服务器 / 打包；自定义插件见 `scripts/vite-plugins.mjs` |
| 单文件打包 | **vite-plugin-singlefile** | 全站内联为一个 `index.html` |
| 样式 | **Tailwind CSS 4** | Utility-first，配合 `@tailwindcss/postcss` |
| 动画 | **Framer Motion 12** | `LazyMotion` + `domAnimation` + `m.*` 组件；`MotionConfig reducedMotion="user"` |
| 公式渲染 | **KaTeX 0.16** | 只内联用到的 woff2 字体（见 §10） |
| 图表 | **自研 `LineChart`（SVG）** | `src/components/LineChart.tsx`，替代 Recharts（省 ~300KB） |
| Markdown | **remark / rehype（仅构建期）** | 笔记在构建期预渲染为 HTML，运行时零解析器 |
| 国际化 | **react-i18next / i18next** | 语言检测：`?lang=` > localStorage > 浏览器 |
| 测试 | **Vitest** | `src/lib/**/*.test.ts`，部署门禁 |
| 图标 | **lucide-react** | 线性图标 |

---

## 3. 整体架构

单组件树 + **Hash 路由**，没有外部状态库，也没有路由库。

```
main.tsx                       入口，挂载 <App/>，引入 i18n
 └─ App.tsx                    外壳：Header / 模块导航 / Content / Footer
     ├─ COURSE_TABS 注册表     每个课程模块一项：{ id, icon, Component, tip }
     ├─ useHashRoute           location.hash ⇄ Route（src/hooks + src/lib/route.ts）
     ├─ 通用组件（src/components）
     │   ├─ Latex              KaTeX 渲染封装
     │   ├─ LineChart          深色面板轻量折线图（悬停 / 方向键查看数值）
     │   ├─ SectionTitle       模块标题
     │   └─ SeniorAdvice       「学长寄语」提示框
     └─ 视图
         ├─ Course：8 个课程模块（src/modules）
         ├─ Package：Markdown 笔记（构建期预渲染）
         └─ Extend：自包含 HTML 讲解（iframe sandbox 隔离）
```

### 路由

| Hash | 视图 |
|------|------|
| `#/course/<tabId>` | 课程模块（默认 `numpy`） |
| `#/package[/<noteId>]` | 笔记列表 / 笔记阅读 |
| `#/extend[/<id>]` | 拓展列表 / 拓展内容 |

`parseRoute` 对任何非法片段回退到最近的合法视图；导航写 `location.hash`，浏览器前进 / 后退 / 刷新 / 分享链接均可用。从 Package/Extend 切回 Course 时恢复上次打开的模块。

### 数据流

- **单向、局部**：每个子模块自管理状态（`useState`），派生值用 `useMemo` 调 `src/lib/` 纯函数计算。
- **模块切换**：`AnimatePresence mode="wait"` 淡入淡出。
- **语言**：i18next 全局托管；切换时同步 `<html lang>`。

---

## 4. 目录结构

```
KnowCS/
├─ index.html              Vite 入口（title / description / OG meta / 内联 favicon）
├─ vite.config.js          插件链：katexFontSlim → markdownHtml → react → singlefile；注入 __APP_VERSION__
├─ vite.lab.config.js      KnowCS Lab 多页构建（lab/*.html → dist-lab/lab/）
├─ lab/                    Lab 各页面的 HTML 入口（index、home-a/b/c、10 个模块）
├─ vitest.config.ts        测试范围限定 src/**/*.test.ts
├─ scripts/
│   ├─ vite-plugins.mjs    构建期插件（KaTeX 字体瘦身、Markdown 预渲染）
│   ├─ zh-hk.mjs           简→港繁转换（OpenCC + 港式术语），locale 与内容共用
│   └─ gen-zh-hk.mjs       zh.json → zh-HK.json
├─ docs/                   design / log / plan
└─ src/
   ├─ main.tsx / App.tsx / i18n.ts / index.css
   ├─ components/          Latex · LineChart · SectionTitle · SeniorAdvice
   ├─ hooks/               useHashRoute
   ├─ lib/                 纯计算 + *.test.ts：knn · bayes · kernel · ndarray · minipy · broadcast · alphabeta · kmeans
   │                       · backprop · perceptron · autograd · chart · route · lang
   │                       · bayesRule · gaussianNb · metrics · crossval · kmeansTable · perceptronTable · mlp · conv2d · otsu · cnnShapes（Lab 用）
   ├─ lab/                 KnowCS Lab：main.tsx · ui.tsx · registry.ts · format.ts · lab.css · pages/ · homes/
   ├─ modules/             各视图组件（Perceptron / PyTorch 已就绪但暂未上架）
   ├─ data/                constants（教学数据）· notes · extensions
   ├─ content/             notes/<id>.{en,zh}.md + images/*.webp · attention.{en,zh}.html
   ├─ locales/             en.json · zh.json · zh-HK.json（生成，勿手改）
   └─ ref/                 历史参考版本（不参与构建 / lint / tsc）
```

> `dist/` 为构建产物，已从版本库移除（`.gitignore`），由 CI 每次重新构建。

---

## 5. 通用组件设计

### 5.1 `Latex`
对 KaTeX 的薄封装。用 `useRef` 拿到 DOM 容器，在 `useEffect` 中调用 `katex.render`，依赖 `[formula, displayMode]`。设置 `throwOnError:false`，公式出错时降级而非崩溃。全局样式 `.katex { text-transform: none }` 防止外层 `uppercase` 把 α 变成 Α。

### 5.2 `LineChart`
纯 SVG 折线图：y 轴「好看」刻度（1/2/5×10ⁿ）、可选竖直参考线（当前 K）与高亮点（最优 K）、悬停 / 键盘方向键显示提示框、`role="img"` + `<title>` 供读屏。刻度与比例尺逻辑在 `src/lib/chart.ts`。

### 5.3 `SectionTitle`
模块顶部标题，统一「图标 + 大写标题 + 斜体副标题」的视觉规范。

### 5.4 `SeniorAdvice`（学长寄语）
琥珀色提示框，承载每个模块的「考试 / 易错点」总结。内容通过 `<Trans components={{…}}>` 传入，可内嵌 `<Latex>` 与 `<strong>`。

> `<Trans>` 约定：一律用 `components={{ 1: …, 3: … }}` 显式映射占位序号，不要依赖 children 下标——children 写法曾导致「条件独立」被渲染成 α。

---

## 6. 可视化模块设计

每个模块都是「左侧交互区 / 右侧结果区」或「输入 → 可视化 → 寄语」的统一布局。

### 6.1 NumpyModule — NumPy 内存机制
- **索引实验台**（`NumpyPlayground.tsx`）：学生在浏览器里写 NumPy 代码，输入停顿 300ms 自动运行（或 Ctrl/⌘+Enter）。每次下标读写都会生成一条 `IndexTrace`，界面据此绘制：源数组→结果的格子映射（悬停联动、结果顺序号、源坐标）、View/Copy/Scalar/Write 徽章与原因、逐轴解释、布尔掩码网格、一维底层缓冲区（视图显示 offset/strides，副本显示新缓冲区）、写穿视图时「被连带修改的变量」。提供 8 个预设示例（切片、fancy、掩码、视图陷阱、三维混合索引、None、ReLU 掩码赋值、`&` 优先级陷阱）。
  - **`lib/ndarray.ts`**：迷你 NumPy 内核。共享 `data` 缓冲区 + `shape/strides/offset/base`；`planIndex` 忠实实现 numpy 索引规则：基本索引返回视图，整数/布尔数组触发高级索引并复制；整数与数组一起广播；高级索引被切片隔开时，广播维度移到最前；`...`/`None`；报错文案与 numpy 一致。另含广播运算、reshape（连续时为视图）、转置、规约（axis）、dot，以及对齐 numpy 默认风格的 repr/str（75 列换行、>1000 元素省略、科学计数法）。
  - **`lib/minipy.ts`**：Python 子集解释器（词法→AST→求值），支持 `import numpy as np`、赋值/元组解包/增量赋值（数组原地写入）、完整运算符优先级与比较链、关键字参数、常用 `np.*` 与数组方法。2026-10-08 起还支持缩进语句块（if / for / while / def / class / with）、lambda、推导式、dict、f-string 与 `math` / `random`，供 6.10 的库实验台使用；不支持 try / raise / yield / *args。单个数组上限默认 4096 元素（各实验台可调高）。
  - **选型：自研 TS 解释器而非 Pyodide**。Pyodide 核心 + NumPy 约 10MB WASM，首屏需数秒，国内访问 CDN 不稳定，也与单文件产物不符；更关键的是，真 NumPy 不会告诉你「结果的每个元素来自哪个地址」，可视化所需的溯源信息仍要在 JS 里重写一遍索引规则。自研方案约 +20KB gzip、即时、离线可用。代价是只支持子集，用与真 NumPy 的差分测试保证一致（95 个片段中 90 个逐字相同，其余为有意不支持的语法）。
- **广播实验台**（`NumpyBroadcast.tsx` + `lib/broadcast.ts`）：
  - **操作**：学生用下拉框编辑 A、B 的形状（0–3 维，每维 1–5），并选择 + / − / × 运算。
  - **示例**：6 个预设，各配一句场景说明：列+行、矩阵−行（标准化）、标量、不兼容、`(3,)+(3,1)` 隐形 bug、三维+二维。
  - **第 1 步 · 规则表**：右对齐，左侧补的 1 标为 1⁺，从最后一维往前逐列判定「相等 / 拉伸 A / 拉伸 B / 冲突」。可「逐步演示」前后翻页，每一步配一句讲解。
  - **第 2 步 · 可视化**：A、B 都画成结果形状，真实元素用实心格，广播出的虚拟副本用虚线格（可切换显示）。悬停任意格子会联动三张网格，标出两边真正被读取的元素，并给出算式 `A[i,0] + B[0,j] = …`。下方展示 `broadcast_to(...).strides`（拉伸轴步长为 0），说明不复制数据。
  - **不兼容时**：标出冲突轴，`suggestFix` 尝试在末尾加一维（如 `B[:, None]`），可一键应用。
  - **代码与知识卡片**：等价 NumPy 代码交给 `lib/minipy` 实际运行，输出与报错和真 NumPy 一致。下方 4 张知识卡片：三句话规则、为何快、COMP2211 用例（标准化 / 偏置 / KNN 两两差 / 灰度化）、`(n,)` 与 `(n,1)` 陷阱。
- **成对距离讲解视频**（`NumpyPairwiseVideo.tsx` + `lib/pairwise.ts`）：考点视频，讲 `np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=-1))`。数据直接取自 2022S 期中 Q2(c) 原题（4 个训练点、2 个测试点），讲解依据该题评分标准（出现循环 0 分、未改形状减法 0 分、axis 错扣分、a²−b² 扣 3 分、行列颠倒扣 1 分），并关联 2023F Minkowski + `argsort(axis=1)[:, :k]`。
  - **实现：时间轴驱动的 React 场景，而不是嵌入视频文件**。9 个场景各有时长与字幕切换点，`requestAnimationFrame` 推进全局时间，每个场景只根据本地进度 p∈[0,1] 渲染，所以拖动和跳章节都是确定的。提供播放 / 暂停、分段时间轴、章节、0.75–1.5× 倍速、键盘（空格、←/→）。字幕走 i18n，三语共用一套动画；单文件产物不需要额外加载媒体。画面里的数值全部来自 `lib/pairwise.ts`，与原题印出的输出由单测锁定。
  - 视频下方的代码由 `lib/minipy` 实际运行。为此解释器新增了 `np.square`、`np.expand_dims`（返回视图）和 `np.argsort` / `ndarray.argsort`（稳定排序，默认 axis=-1，axis=None 时先拉平），输出已与真 NumPy 逐字核对。
  - **数据**：A = arange，B = (arange+1)×10，结果里一眼能看出两边各贡献了什么。

- **NumPy API 面板**（`NumpyApiPanel.tsx`，显示在 Package 的 NumPy 笔记右侧）：
  - **目录**：`data/numpyApis.ts` 共 55 个示例，分 8 组（创建 / 形状 / 索引 / 运算 / 规约 / 排序 / 线性代数 / 随机），说明文案在 `numpy_api.api.*`。支持搜索；笔记里能演示的行内名字（`numpy.linalg`、`ndarray` …）渲染时加 `np-link`，点击打开对应分组；沙盒外的命名空间（`np.fft`、`np.polynomial` …）给出提示。
  - **调用追踪**：`minipy` 为每次 NumPy 函数 / 数组方法 / 运算符 / 比较 / `.T` / 下标读取记录一条 `CallTrace`（源码、API 名、类别、操作数快照、axis、结果）。搬运类调用（reshape、stack、tile、flip、索引…）在「元素编号数组」上重放一次，直接得到每个结果元素的来源。
  - **溯源**（`lib/npTrace.ts`）：按类别算出结果每个元素依赖的输入元素——逐元素（含广播）、沿轴规约（最值 / argmax 标出被选中的元素）、累加、矩阵乘（第 i 行 × 第 j 列）、排序、去重、trace / inv / det。悬停结果格子高亮来源，悬停输入格子高亮受影响的结果。
  - **新增 API**（`lib/ndops.ts` + `minipy`）：concatenate / stack / vstack / hstack / tile / repeat / flip / swapaxes / expand_dims / squeeze、sort / argsort / unique、cumsum、prod / std / var / count_nonzero、round（银行家舍入）/ clip / maximum / minimum / square / floor / ceil / sign / sin / cos / add… / logical_*、outer / trace、`np.linalg.inv/det/norm`、`np.random.seed/rand/randn/randint`（自带种子生成器，数值与 NumPy 不同）、*_like / identity。`npapi.test.ts` 以 NumPy 2.2.6 的真实输出为黄金值（70 例）；已知差异：0 维结果显示为 `12` 而非 `np.int64(12)`，det 为精确值。
  - **多项式**（`lib/poly.ts`）：支持 `from numpy.polynomial import Polynomial`，Polynomial 的求值 / deriv / integ / roots（复根按 numpy 格式输出）/ + − * ** / fit / convert，以及旧版 `np.polyfit` / `np.polyval`。这类调用不画网格，而是画曲线图（数据点、求值点、实根标记）。笔记里的两个代码块带「在 NumPy 面板中运行」按钮，打开补全后可运行的版本。
  - 打开 NumPy 笔记时整页放宽到 1440px（`App.tsx` 的 `wide`），≥1280px 时面板吸顶并可独立滚动，窄屏放在正文下方。

### 6.2 BackpropModule — 反向传播（单输出 sigmoid 神经元）
- 网络 i, j → k，六个滑块：`O_i`、`O_j`、`w_ik`、`w_jk`、`T_k`、`η`。
- 「Notebook」四步全部**代入当前数值**：前向（net / O / E）→ 链式法则 → `δ_k = (T−O)·O(1−O)` → `Δw = η·δ·O`。
- 「反向传播」按钮：误差粒子从 k 沿两条边流回，结束后在边上揭示 Δw；「执行更新」做一次 `w ← w + Δw`，下方折线记录 E 随更新下降。
- 计算在 `lib/backprop.ts`，测试用数值梯度校验 δ 与 Δw。

### 6.3 KernelModule — 卷积 / 边缘检测
- 输入图 `I` × 卷积核 `K` = 输出特征图（6×6，取 abs 并 clamp 到 [0,255]）。
- 预设 Sobel-X/Y、Laplacian、Identity，可手填（非法输入按 0 处理，避免 NaN）。

### 6.4 BayesBasicsModule — 贝叶斯基础
- 公式区四色块解释 后验 / 先验 / 似然 / 边缘。
- 火警案例：`P(Fire)`、`P(Smoke)`、`P(Smoke|Fire)` 三滑块。由全概率公式 `P(S) ≥ P(S|F)·P(F)`（`lib/bayes.ts` 的 `minEvidence`），P(Smoke) 滑块下限随另两个值抬高，后验永不超过 100%，并在界面上解释原因。

### 6.5 NaiveBayesModule — 朴素贝叶斯
- 疾病 Z 数据 `BAYES_DATA`；类别样本数由计数表推出（`classTotals`），不再硬编码。
- **α 平滑**：`(count+α)/(total+m·α)`，输入框 clamp 到 [0, 2]。
- **Log 模式**：结果区显示每类的连乘积 / 对数和分数，似然链显示 `log P`；零频率时如实显示 `−∞` 并给出提示（不再用 `1e-10` 掩盖）。

### 6.6 KnnModule — K 近邻
- 400×400 SVG 散点；点击画布（按实际渲染尺寸换算回 viewBox 坐标）或聚焦后用方向键移动测试点。
- **标准化开关**：原始欧氏距离 vs Z-score 距离。
- **平票**：K 为偶数时可能出现，由最近邻裁决并显示提示（`isTie`），而非默认偏向某一类。
- **留一法误差曲线**：`looErrorCurve` 对 K=1..15 真实计算 LOO 误差率，随标准化开关重算；绿点为最优 K，虚线为当前 K。

### 6.7 AlphaBetaModule — Minimax + α-β 剪枝
- 3 层博弈树 SVG，DFS 逐步追踪（播放 / 步进 / 调速），叶子值沙盒与三种预设。

### 6.8 KMeansModule — K-Means 聚类
- EM 迭代动画（播放 / 单步 / 重置）、K=2–5、WCSS、Elbow 曲线、Raw / Z-score。

### 6.9 KnowCS Lab — `/lab/` 多页原型
独立于主站单文件产物的**多页 React 原型**，用于试验「每个模块一个 HTML 页面」的布局与新模块，选定后再迁入主站。
- **构建**：`vite.lab.config.js`，`lab/*.html` 每个文件一个入口，共用 `src/lab/main.tsx`（按 `<body data-page>` 动态 import 页面，各页只下载自己的 chunk）。产物在 `dist-lab/lab/`，部署时整个 `lab/` 目录放进 release，线上地址 `https://knowcs.online/lab/`。不使用 singlefile。
- **外壳**（`src/lab/ui.tsx`）：顶栏（返回总览 / 上一页 / 下一页 / Quiz me）、Hero（讲次 + 出现过该题型的试卷）、左控件右画布的 `Workspace`、卡片/预设/分段按钮等。`<Ans>` 单元格在「Quiz me」开启时变成输入框，失焦判分（分数、四舍五入、Unicode 负号都接受；很小的数按相对误差判）。访问记录存 `localStorage`，只做本机进度便利。
- **内容注册表**（`src/lab/registry.ts`）：讲次、9 份试卷、10 个新页面、8 个主站模块，以及 16 个「反复出现的题型 × 试卷」映射，供首页草图与页面徽章使用。
- **10 个新页面**（纯计算均在 `src/lib/` 并配测试，测试用讲义与评分标准的数字作黄金值）：
  | 页面 | lib | 要点 |
  |---|---|---|
  | Bayes & the base rate | `bayesRule.ts` | 10,000 人方格、自然频率树、P(B \| not E) 变体、「再测一次」 |
  | Gaussian Naive Bayes | `gaussianNb.ts` | 每类钟形曲线、样本标准差 (n−1)、乘积与后验 |
  | Confusion matrix & F1 | `metrics.ts` | 点类别拆出 TP/FN/FP/TN、行=实际/预测切换、macro/weighted F1、MCC、NumPy 写法 |
  | D-fold cross-validation trap | `crossval.ts` | 排序不打乱 → 0%/50%/100%，打乱与分层对比，各 D 的平均准确率 |
  | K-Means by hand | `kmeansTable.ts` | 逐轮距离表、平局归 C1、SSE、空簇、1-D/2-D、欧氏/平方/曼哈顿 |
  | Perceptron learning table | `perceptronTable.ts` | 考试表格逐行、激活约定（1/0 或 ±1，z≥0 或 z>0）、决策线、XOR 不收敛 |
  | XOR & backprop by numbers | `mlp.ts` | 讲义 2-2-1 网络，输入空间 / 隐藏空间两图，按考试公式的一步反传表，参数计数 |
  | Convolution, padding & flips | `conv2d.ts` | 四种填充（zero/replicate/reflect/mirror）、翻转与否、步长、输出尺寸、核取值范围 |
  | Histogram, contrast & Otsu | `otsu.ts` | 点运算与直方图、Otsu 迭代表、图像↔直方图配对题 |
  | CNN shapes & parameters | `cnnShapes.ts` | 可编辑层栈、逐层形状与参数及算式、与 MLP 对比、步长示意 |
- **首页草图**：A 课程地图（按讲次）、B 考点雷达（题型 × 试卷热力图）、C 学习路径（时间线 + 能力自查清单）。
- **语言**：原型暂为英文；选定版本迁入主站时按 i18n 规则补齐三语。
- **已知出入**：2022 期末 Part B Q1(c) 评分标准第 4 层写 13×13，按公式应为 14×14（遗漏 padding）；页面与测试以公式为准并注明。

### 6.10 库实验台 — matplotlib / PyTorch / Keras / TensorFlow / pandas
资料包里这五篇笔记的右侧是同一种实验台：示例目录（分组 + 搜索 + 说明）、可编辑代码（Tab 缩进、冒号后自动缩进、停顿后自动运行）、逐步可视化、输出与图。每个面板都能全屏（`useFullScreen`：固定覆盖层 + 浏览器全屏 API；Esc 或按钮退出；URL 带 `#playground` 时直接全屏打开）；NumPy 面板也用同一个钩子。笔记里能在该沙盒运行的 Python 代码块会出现「试一试」，点开即载入实验台（没有 import 的代码块自动补上该库的常用 import）。

- **选型沿用 6.1**：仍是自研 TS 解释器而非 Pyodide。PyTorch / TensorFlow 没有可用的 WASM 发行版，而可视化需要的「这一层的形状从哪来、梯度沿哪条边回传、哪些行被选中」只有自己实现才拿得到。代价是只覆盖课程用到的子集；打印格式、默认值和报错信息与真实库做差分测试。
- **解释器扩展**（`lib/minipy.ts`）：
  - 库通过 `PyLib`（模块名 → 构建函数，每次运行懒加载一次）、`PyObj`（库对象：属性、下标、运算符、迭代、上下文管理器等钩子）、`HostClass`（可被用户类继承的库基类，如 `nn.Module`）接入；`Host` 接口给库用（报错、调用、取名、记录事件、随机数、每次运行的状态）。
  - 运行限制：时间预算（默认 2 s，超时抛 `TimeoutError`，防死循环卡页面）、递归深度、单数组元素上限（`RunOptions.maxSize`，深度学习实验台为 2,000,000）。同一行代码只记录前两次执行的追踪，100 个 epoch 的循环不会产生 100 份重复步骤。
  - 结果里新增 `events`（图、DataFrame 操作、模型形状流、自动求导计算图、训练曲线、讲解提示）、`displays`（运行结束时仍打开的图）和 `outDisplay`（最后一行是 DataFrame 时以表格显示）；调用与事件共用 `seq` 排序成一条步骤时间线。
- **张量引擎**（`lib/tensorCore.ts`）：数值存于 NDArray（因而 `torch.from_numpy` 与原数组共享内存）、PyTorch 的 dtype 提升规则（Python 标量为「弱类型」）、float32 以 `Math.fround` 存储；反向模式自动求导（梯度在叶子上累加、图默认用后释放、非叶子不保留梯度）。linear / conv2d / 池化 / dropout / softmax / 各损失都是带 PyTorch grad_fn 名字的单个算子（AddmmBackward0、ConvolutionBackward0 …），计算图读起来与真实的一致。每个算子的梯度用中心差分检验。
- **各库**：
  - `lib/pyPlot.ts`（matplotlib.pyplot + `seaborn.heatmap`）：图是数据（`FigureSpec`），每次画图调用后记录一帧并标出本次新增的图元；保留关键默认值（tab10 色、二维数据默认 viridis、imshow 不给 vmin/vmax 时拉伸自身范围、图像 y 轴朝下）。`FigureView.tsx` 画成 SVG（图像用 canvas 生成 data URL），悬停像素给出「值 → 归一化 → 色图 → 颜色」。不写 cmap、不写 vmin/vmax 时给出讲解提示。
  - `lib/pyTorch.ts`：张量（打印、dtype、视图与 `view` 的连续性检查）、`nn` 层与 `nn.Module` 子类（每次前向逐层记录输入输出形状、尺寸公式、参数个数）、`F`、SGD / Adam、TensorDataset / DataLoader、内存中的 `torch.save` / `load`。`backward()` 记录计算图（参数用模型里的名字）；小张量的运算复用 NumPy 面板的元素溯源视图。提示：梯度累加、评估时 dropout 仍开、softmax 后再用 CrossEntropyLoss、MSE 形状不一致被广播。
  - `lib/pyKeras.ts`（keras + tensorflow）：Sequential / Functional、Dense / Conv2D / 池化 / Flatten / Dropout 等、L1/L2 正则、Adam / SGD / RMSprop、交叉熵与 MSE、EarlyStopping；`summary()` 输出 Keras 3 的表格并生成形状流视图，`fit` 真实训练并返回 History（训练曲线视图）。TensorFlow 部分：`tf.constant` / `Variable` / `GradientTape`、不做隐式类型提升、TF 的打印格式。提示：Dense 前漏 Flatten、整数标签配 categorical_crossentropy、验证损失回升（过拟合）。
  - `lib/pyPandas.ts`：Series / DataFrame（pandas 2 的打印规则：值前留符号位、非文本列表头也多一格、单空格分列、浮点列统一小数位后整体去尾零）、df[…] / 布尔掩码 / loc / iloc、isna / fillna / dropna、groupby、value_counts、crosstab、get_dummies、merge、describe、`read_csv(io.StringIO(...))`。每个表格操作记录源表、被选中的行列或分组着色与结果。
- **界面**（`src/site/playground/`）：`Playground.tsx`（外壳）、`CodeEditor.tsx`、`views.tsx`（按事件类型分派：调用网格 / 图 / DataFrame / 形状流 / 计算图 / 训练曲线）、`FigureView.tsx`、`useFullScreen.ts`、`noteBlocks.ts`（找出可运行的代码块）、`loaders.ts`（笔记 id → 配置 + 文案包）。配置与示例在 `src/data/playgrounds/<库>.ts`（`PlayConfig`：库、元素上限、时间预算、防抖、分组、示例、prelude、哪些调用列为步骤）。
- **文案**：`playground.*` 命名空间，按 `playground.<子键>` 拆包（ui / view / note / frame / 各库），各笔记只下载自己的。
- **示例数据**：沙盒不能下载 MNIST，训练示例的数据都在代码里现场生成（XOR、高斯团、带噪声的 8×8 横竖条图），并在说明里写明；随机数来自沙盒自己的生成器，与真实库不同。
- **测试**：每个库都有与真实库的差分测试（CPython 3.10、NumPy 2.2、pandas 2.3、matplotlib 3.10、PyTorch 2.12、Keras 3.12 的输出作黄金值）；`playgrounds.test.ts` 运行全部 77 个示例，要求无错误（或得到示例声明的错误）且有东西可看。

---

## 7. 状态管理

| 层级 | 状态 | 用途 |
|------|------|------|
| URL | `location.hash` | 当前视图 / 模块 / 笔记（单一事实来源） |
| App | `lastTab` | 切回 Course 时恢复的模块 |
| App | i18n.language | 当前语言（i18next 托管，写入 localStorage `knowcs-lang`） |
| 子模块 | 各自 `useState` | 模块内交互参数 |
| 子模块 | `useMemo` 派生 | 调用 `src/lib/` 纯函数 |

---

## 8. 国际化（i18n）设计

- **初始化**（`i18n.ts`）：`supportedLngs: en / zh / zh-HK`；`fallbackLng: zh-HK → zh → en`；检测顺序 `?lang=` → localStorage → `navigator`，经 `lib/lang.ts` 的 `normalizeLang` 归一（zh-TW / zh-Hant / zh-MO → zh-HK，其余 zh-* → zh）；`debug` 仅开发环境开启。
- **文案结构**：按模块命名空间组织（`numpy_module.*`、`bayes.naive.*`、`knn.*`、`app.*` …）。
- **富文本**：`<Trans i18nKey components={{…}}>` 注入 `<Latex>` / `<strong>`；文案中不要写 Markdown（`**x**` 不会被渲染）。
- **繁体**：`zh-HK.json` 由 `npm run gen:zh-hk` 生成，严禁手改。
- **资料包 / 拓展内容**：每篇笔记与拓展页维护 `en` / `zh` 两份源文件。`zh-HK` 在构建期由 `zh` 源转换，规则与 locale 相同（`?html-hk` / `?raw-hk`，见 `scripts/vite-plugins.mjs`）。`PackageModule` / `ExtendModule` 按 `normalizeLang(i18n.resolvedLanguage)` 选取版本，切换语言即时生效。图片以 ES import 引入，三种语言共用一份。`src/content/content.test.ts` 守卫以下几点：双语成对、图片一致、英文版无中文、无 Obsidian 专有语法。

---

## 9. 视觉、交互与无障碍

- **配色**：浅灰底（`#f1f5f9`）、白色卡片、大圆角、深色面板（`slate-900`）承载结果展示。
- **布局**：桌面端左侧竖排模块导航 + Exam Tip；移动端（< md）模块导航变为横向滚动条并自动滚到当前项，Exam Tip 移到内容底部。
- **动画**：Framer Motion；尊重系统「减少动态效果」设置。
- **无障碍**：跳到主要内容链接；导航 `aria-current`；语言菜单 `menu` / `menuitemradio`（方向键、Esc）；开关按钮 `aria-pressed`；画布 / 图表有 `aria-label` 或 `<title>`；结果区 `aria-live`；所有 range 输入有 label。

---

## 10. 构建与部署

```bash
npm run dev       # 开发服务器（http://localhost:5174）
npm run lint      # ESLint（门禁）
npm test          # Vitest（门禁）
npm run typecheck # tsc --noEmit（门禁）
npm run build     # tsc --noEmit && vite build → dist/index.html（门禁）
npm run dev:lab   # Lab 开发服务器（http://localhost:5175/lab/index.html）
npm run build:lab # tsc --noEmit && vite build -c vite.lab.config.js → dist-lab/lab/
```

**单文件体积优化**（2026-10，4.55MB → 1.99MB，gzip 2.46MB → 1.07MB）：

| 措施 | 位置 | 效果 |
|------|------|------|
| KaTeX 只保留 woff2，并去掉未用字族（Caligraphic / Fraktur / Script / SansSerif / Typewriter） | `katexFontSlim` 插件 | 60 个字体文件 → 11 个，≈ −1.2MB |
| 笔记配图由原始 PNG 重新编码为 WebP q70（≤1200px） | `src/content/notes/images` | 1.22MB → 0.60MB，且画质优于原 JPEG q50 |
| 移除 Recharts（及 d3 / redux / decimal.js 等依赖），改用自研 SVG `LineChart` | `components/LineChart.tsx` | ≈ −300KB JS |
| Markdown 构建期预渲染（`*.md?html`），移除 react-markdown 运行时 | `markdownHtml` 插件 | ≈ −130KB JS |
| Framer Motion `LazyMotion` + `domAnimation` | `App.tsx` | 只打包用到的动画特性 |

> 若以后公式用到 `\mathcal`、`\mathfrak`、`\mathscr`、`\mathsf`、`\texttt`，需把对应字族加回 `scripts/vite-plugins.mjs` 的 `KATEX_FONT_KEEP`，否则会回退为系统字体。

**CI**：Node 22，只做构建校验。生产站点由 `ssh pastpaper` 上的 OpenResty 托管，发布/HTTPS/回滚见 `deploy/README.md`。

---

## 11. 扩展指引

新增一个课程模块：

1. 纯计算写进 `src/lib/xxx.ts` 并配 `xxx.test.ts`（黄金值 + 边界）。
2. 在 `src/modules/XxxModule.tsx` 编写组件：状态自管理、派生值 `useMemo`、动画用 `m.*`（不是 `motion.*`，`LazyMotion strict` 会报错）。
3. 在 `App.tsx` 的 `COURSE_TABS` 追加 `{ id, icon, Component, tip }`，并在 `types.ts` 的 `TabId` 加上 id——路由自动生效。
4. 在 `en.json` / `zh.json` 补齐 `app.tabs.xxx`、`app.tabs_sub.xxx`、`app.section.xxx.*`、`app.sidebar.exam_tip.content_xxx` 及模块文案，然后 `npm run gen:zh-hk`。
5. 用 `SeniorAdvice` 收尾；交互元素补 `aria-label` / `aria-pressed`。

新增 Lab 页面：纯计算写 `src/lib/xxx.ts` + 测试 → 在 `lab/` 加 `xxx.html`（复制任一页，改 `data-page` 与标题）→ `src/lab/pages/Xxx.tsx` 用 `LabPage` / `Workspace` / `Ans` 搭页面 → 在 `src/lab/main.tsx` 的 `PAGES` 和 `src/lab/registry.ts` 的 `LAB` 各加一项。

新增 Package 笔记：放 `src/content/notes/x.en.md` 与 `x.zh.md`（图片放 `images/`，相对引用，两版引用一致），在 `src/data/notes.ts` 以 `?html`（en、zh）+ `?html-hk`（由 zh 生成）注册三种语言，并把 id 加进 `sitemap.ts` 的 `NOTE_IDS`。不要使用 Obsidian 的 `> [!tip]` / `[[双链]]`，它们会原样显示；笔记不渲染 LaTeX。

给笔记配实验台：库实现写 `src/lib/pyXxx.ts`（`PyLib` + `PyObj`）并配差分测试 → 示例写 `src/data/playgrounds/xxx.ts`（`PlayConfig`）并加入 `playgrounds.test.ts` → 在 `src/site/playground/loaders.ts` 的 `PLAYGROUND_LOADERS` 用笔记 id 注册（带 `playground.<id>` 文案包）→ `playground.<id>.*` 文案（title、subtitle、sandbox_note、cat.*、e.<示例>）。

---

## 12. 已知约束与改进方向

- **教学数据硬编码**（`BAYES_DATA`、`KNN_RAW_DATA` 等），与课程讲义绑定；KNN 的标准化统计量（均值 / 标准差）亦为讲义给定值。
- **单文件不利于增量缓存**：每次部署都是整包更新；如未来托管支持多文件，可去掉 singlefile 并按模块懒加载。
- **UI / 交互层无自动化测试**：`src/lib/` 有完整单测，组件层靠手动 / 截图验证。
- **i18n 双份手工维护**，可引入键一致性校验脚本防止漏翻。
- **库实验台是子集**：没有 try / *args、MultiIndex、BatchNorm / RNN / 注意力层、GPU、文件与数据集下载；随机数与真实库不同。代码在主线程运行（有时间预算），最重的示例（第 9 讲 MNIST_CNN，约 160 万参数）一次约 0.25 s。
