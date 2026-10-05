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
├─ vitest.config.ts        测试范围限定 src/**/*.test.ts
├─ scripts/
│   ├─ vite-plugins.mjs    构建期插件（KaTeX 字体瘦身、Markdown 预渲染）
│   └─ gen-zh-hk.mjs       zh.json → zh-HK.json（OpenCC + 港式术语）
├─ docs/                   design / log / plan
└─ src/
   ├─ main.tsx / App.tsx / i18n.ts / index.css
   ├─ components/          Latex · LineChart · SectionTitle · SeniorAdvice
   ├─ hooks/               useHashRoute
   ├─ lib/                 纯计算 + *.test.ts：knn · bayes · kernel · ndarray · minipy · alphabeta · kmeans
   │                       · backprop · perceptron · autograd · chart · route · lang
   ├─ modules/             各视图组件（Perceptron / PyTorch 已就绪但暂未上架）
   ├─ data/                constants（教学数据）· notes · extensions
   ├─ content/             notes/*.md + images/*.webp · attention.html
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
  - **`lib/minipy.ts`**：Python 子集解释器（词法→AST→求值），支持 `import numpy as np`、赋值/元组解包/增量赋值（数组原地写入）、完整运算符优先级与比较链、关键字参数、常用 `np.*` 与数组方法。不支持 for/if/def。单个数组上限 4096 元素。
  - **选型：自研 TS 解释器而非 Pyodide**。Pyodide 核心 + NumPy 约 10MB WASM，首屏需数秒，国内访问 CDN 不稳定，也与单文件产物不符；更关键的是，真 NumPy 不会告诉你「结果的每个元素来自哪个地址」，可视化所需的溯源信息仍要在 JS 里重写一遍索引规则。自研方案约 +20KB gzip、即时、离线可用。代价是只支持子集，用与真 NumPy 的差分测试保证一致（95 个片段中 90 个逐字相同，其余为有意不支持的语法）。
- **广播机制**：静态图示 `(3,1) + (1,4) → (3,4)`。

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

新增 Package 笔记：放 `src/content/notes/x.md`（图片放 `images/`，相对引用），在 `src/data/notes.ts` 用 `import x from '…/x.md?html'` 注册即可。

---

## 12. 已知约束与改进方向

- **教学数据硬编码**（`BAYES_DATA`、`KNN_RAW_DATA` 等），与课程讲义绑定；KNN 的标准化统计量（均值 / 标准差）亦为讲义给定值。
- **单文件不利于增量缓存**：每次部署都是整包更新；如未来托管支持多文件，可去掉 singlefile 并按模块懒加载。
- **UI / 交互层无自动化测试**：`src/lib/` 有完整单测，组件层靠手动 / 截图验证。
- **i18n 双份手工维护**，可引入键一致性校验脚本防止漏翻。
