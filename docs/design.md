# CS Helper (KnowitCS) — 设计文档

> 面向 HKUST COMP2211（机器学习）的交互式学习网站。通过可视化帮助学生理解机器学习核心概念。

---

## 1. 项目概述

CS Helper 是一个**纯前端、单页**的交互式教学应用。它把机器学习课程中较难理解的抽象概念（内存切片、广播、反向传播、卷积、贝叶斯、KNN）拆解成一组**可交互的可视化模块**，让学生通过拖动滑块、点击画布、修改参数等方式实时观察算法行为。

设计目标：

- **可视化优先**：每个概念都配一张动态图 / 动画，而不是静态文字。
- **即时反馈**：所有计算在浏览器端实时完成，参数一改结果立刻更新。
- **双语**：中英文一键切换，服务本地与国际学生。
- **零后端、易分发**：整站打包成**单个 HTML 文件**，可直接托管（含 IPFS/ENS 部署）。

在线 Demo：
- 英文版：<https://knowitcs.pinit.eth.link/>
- 中文版：<https://knowcs.pinit.eth.link/>

---

## 2. 技术栈

| 维度 | 选型 | 说明 |
|------|------|------|
| 框架 | **React 19** | 函数组件 + Hooks |
| 构建 | **Vite 7** | 开发服务器 / 打包 |
| 单文件打包 | **vite-plugin-singlefile** | 全站内联为一个 `index.html` |
| 样式 | **Tailwind CSS 4** | Utility-first，配合 `@tailwindcss/postcss` |
| 动画 | **Framer Motion 12** | 过渡、路径动画、数值过渡 |
| 公式渲染 | **KaTeX 0.16** | LaTeX 数学公式 |
| 图表 | **Recharts 3** | KNN 模型复杂度曲线 |
| 国际化 | **react-i18next / i18next** | 含浏览器语言检测 |
| 图标 | **lucide-react** | 线性图标 |

构建配置（`vite.config.js`）：开发端口 `5174`，插件链 `react()` + `viteSingleFile()`。

---

## 3. 整体架构

应用是**单组件树 + Tab 路由**的经典结构，没有外部状态库，也没有路由库——「页面切换」由顶层 `activeTab` 状态驱动。

```
main.jsx                      入口，挂载 <App/>，引入 i18n
 └─ App.jsx (default export)   主壳：Header / Sidebar / Content / Footer
     ├─ 通用组件
     │   ├─ Latex              KaTeX 渲染封装
     │   ├─ SectionTitle       模块标题
     │   └─ SeniorAdvice       「学长寄语」提示框
     └─ 可视化子模块（按 activeTab 切换其一）
         ├─ NumpyModule        内存切片 + 广播
         ├─ BackpropModule     反向传播链式法则
         ├─ KernelModule       卷积 / 边缘检测
         ├─ BayesBasicsModule  贝叶斯公式 + 火警案例
         ├─ NaiveBayesModule   朴素贝叶斯推断 + 平滑
         └─ KnnModule          KNN 投票 + 决策边界
```

### 数据流

- **单向、局部**：每个子模块自管理状态（`useState`），派生值用 `useMemo` 计算，无跨模块共享状态。
- **Tab 切换**：`App` 持有 `activeTab`，通过条件渲染挂载对应模块；切换时 `AnimatePresence mode="wait"` 做淡入淡出。
- **语言**：全局由 i18next 管理，组件用 `useTranslation()` 读取，`toggleLanguage()` 切换 `zh`/`en`。

---

## 4. 目录结构

```
KnowCS/
├─ index.html              Vite 入口 HTML
├─ vite.config.js          构建配置（单文件打包，端口 5174）
├─ package.json            依赖与脚本（dev/build/lint/preview）
├─ eslint.config.js        ESLint（react-hooks / react-refresh）
├─ postcss.config.js       Tailwind PostCSS
├─ dist/                   构建产物（单 HTML）
├─ docs/                   文档目录
├─ public/                 静态资源
└─ src/
   ├─ main.jsx             React 挂载点
   ├─ App.jsx              主应用（全部模块，约 1065 行）
   ├─ i18n.js              i18next 初始化
   ├─ index.css / App.css  全局样式
   ├─ locales/
   │   ├─ en.json          英文文案
   │   └─ zh.json          中文文案
   └─ ref/                 历史参考版本（不参与构建）
       ├─ App_v1.jsx
       ├─ App_v2_ref.jsx
       └─ App_knn.jsx
```

> `src/ref/` 保留了迭代过程中的旧版组件，作为设计参考，不被 `main.jsx` 引用。

---

## 5. 通用组件设计

### 5.1 `Latex`
对 KaTeX 的薄封装。用 `useRef` 拿到 DOM 容器，在 `useEffect` 中调用 `katex.render`，依赖 `[formula, displayMode]`。设置 `throwOnError:false`，公式出错时降级而非崩溃。支持行内 / 块级（`displayMode`）两种渲染。

### 5.2 `SectionTitle`
模块顶部标题，统一「图标 + 大写标题 + 斜体副标题」的视觉规范。

### 5.3 `SeniorAdvice`（学长寄语）
琥珀色提示框，承载每个模块的「考试 / 易错点」总结。内容通过 `<Trans>` 传入，可内嵌 `<Latex>` 与 `<strong>`，实现「富文本 + 公式 + 多语言」三合一。Framer Motion 做入场动画。

---

## 6. 可视化模块设计

每个模块都是「左侧交互区 / 右侧结果区」或「输入 → 可视化 → 寄语」的统一布局。

### 6.1 NumpyModule — NumPy 内存机制
- **切片可视化**：4×4 矩阵（`INITIAL_MATRIX`），三种模式按钮——
  - `slice`（基础切片 → View/引用）
  - `fancy`（花式索引 → Copy/新内存）
  - `mask`（布尔掩码 → Copy）
  - `getHighlight()` 决定高亮哪些格子，右侧解释 View vs Copy 的内存语义。
- **广播机制**：静态图示 `(3,1) + (1,4) → (3,4)`，配「从末位对齐、相等或其一为 1」的准则。

### 6.2 BackpropModule — 反向传播
- 左侧：神经元图（i/j → k），点击「追踪梯度」用 Framer Motion 让误差粒子从输出层反向流动（`isAnimating` 控制，2s 后复位）。
- `errorVal` 滑块表示 `(Target − Out)`。
- 右侧「Notebook」：分三步展示链式法则公式
  - `∂E/∂w_jk` 链式展开
  - `δ_k = (T_k − O_k)·f'(net_k)`
  - `Δw_jk = η·δ_k·O_j`

### 6.3 KernelModule — 卷积 / 边缘检测
- 三联视图：输入图像 `I` × 卷积核 `K` = 输出特征图。
- 预设核 `KERNEL_PRESETS`：Sobel-X/Y、Laplacian、Identity，亦可手填自定义核（触发 `Custom`）。
- `sourceImage`：8×8 中心方块；`getConvolvedVal()` 逐点做 3×3 卷积，取 `abs` 后 clamp 到 `[0,255]` 灰度渲染。

### 6.4 BayesBasicsModule — 贝叶斯基础
- 公式区：`P(B|E) = P(B)·P(E|B)/P(E)`，四个色块分别解释 后验 / 先验 / 似然 / 边缘。
- 火警案例：三个滑块 `P(Fire)`、`P(Smoke)`、`P(Smoke|Fire)`，实时算出 `P(Fire|Smoke)` 并展示代入过程，传达「证据 ≠ 结论」。

### 6.5 NaiveBayesModule — 朴素贝叶斯
- 数据集硬编码于 `BAYES_DATA`（疾病 Z 案例）：先验、各特征条件计数、各特征取值数 `m_values`。
- 四个特征下拉（BP / Fever / Diabetes / Vomit），可调：
  - **平滑系数 `α`**：`prob = (count+α)/(total+m·α)`（拉普拉斯/m-估计平滑）。
  - **Log 模式**：连乘 vs 对数求和，规避数值下溢。
- 输出归一化后的 `P(Yes)` / `P(No)` 概率条 + 似然链逐项展开。

### 6.6 KnnModule — K 近邻
- 400×400 SVG 散点（`KNN_RAW_DATA`：身高/体重 → M/L 两类），**点击画布**设置测试点 `testPoint`。
- `processedData`（`useMemo`）：计算各点距离 → 排序 → 取前 K → 标号 → 决策圆半径 → 多数投票预测。
- **标准化开关**：原始欧氏距离 vs Z-score 标准化距离（消除量纲影响）。
- 右侧 Recharts 折线展示「K vs 误差」模型复杂度曲线，`ReferenceLine` 标出当前 K，提示低 K 过拟合 / 高 K 欠拟合。
- 投票面板显示 M/L 票数与当前预测类别。

---

## 7. 状态管理

无全局状态库，遵循「就近管理」：

| 层级 | 状态 | 用途 |
|------|------|------|
| App | `activeTab` | 当前模块 |
| App | i18n.language | 当前语言（由 i18next 托管）|
| 子模块 | 各自 `useState` | 模块内交互参数 |
| 子模块 | `useMemo` 派生 | 距离/概率/卷积等实时计算 |

派生计算全部走 `useMemo`，依赖明确，避免重复运算与不必要重渲染。

---

## 8. 国际化（i18n）设计

- **初始化**（`i18n.js`）：`LanguageDetector` + `initReactI18next`，资源为 `en.json` / `zh.json`，`fallbackLng:'en'`，默认 `lng:'en'`，关闭 `escapeValue`（交给 React）。
- **文案结构**：按模块命名空间组织（如 `numpy_module.*`、`bayes.naive.*`、`knn.*`、`app.*`）。
- **富文本**：含公式 / 加粗的句子用 `<Trans i18nKey=... components={...}>`，把 `<Latex>`、`<strong>` 作为占位组件注入，实现多语言下的公式与排版一致。
- **切换**：Header 的语言按钮调用 `i18n.changeLanguage('zh'|'en')`。

> 维护要点：`en.json` 与 `zh.json` 的键必须**严格对齐**，新增文案需同步两份；带占位符的 `<Trans>` 句子需保证两语言的占位序号一致。

---

## 9. 视觉与交互规范

- **配色**：浅灰底（`#f1f5f9`）、白色卡片、大圆角（`rounded-[2rem]`）、深色面板（`slate-900`）承载结果展示。功能色——蓝（主/M 类）、红（误差/L 类）、绿（输出/View）、紫/橙/琥珀（辅助语义）。
- **布局**：顶部 Header（标题 + 状态指示 + 语言切换），左侧 Sidebar 导航（含「考试 Tip」卡片），右侧内容区，底部品牌 Footer。
- **动画**：模块切换淡入淡出；数值条、邻居连线、梯度粒子等用 Framer Motion 过渡，强化「过程感」。

---

## 10. 构建与部署

```bash
npm install      # 安装依赖
npm run dev      # 开发服务器（http://localhost:5174）
npm run build    # 生产构建 → dist/ 单 HTML
npm run preview  # 预览构建产物
npm run lint     # ESLint 检查
```

- **单文件产物**：`viteSingleFile` 将 JS/CSS 全部内联进 `dist/index.html`，无外部依赖请求，便于：直接拖入浏览器打开、CDN / IPFS / ENS（`*.pinit.eth.link`）托管。
- **代价**：单文件体积偏大（KaTeX/Recharts 等），不利于增量缓存——对教学站点可接受。

---

## 11. 扩展指引

新增一个可视化模块的步骤：

1. 在 `App.jsx` 中编写 `XxxModule` 组件，状态自管理、派生值用 `useMemo`。
2. 在 `tabs` 数组追加 `{ id, label: t('app.tabs.xxx'), icon, subtitle }`。
3. 在内容区 `AnimatePresence` 内增加 `activeTab === 'xxx'` 分支（配 `SectionTitle` + 模块）。
4. 在 `en.json` / `zh.json` 同步补齐 `app.tabs.xxx`、`app.section.xxx.*` 及模块文案键。
5. 沿用 `SeniorAdvice` 收尾，给出考试 / 易错点提示。

> 当模块增多、`App.jsx` 过长时，建议将各 `XxxModule` 拆分到 `src/modules/` 独立文件，`App.jsx` 仅保留壳与路由——可参考 `src/ref/` 的拆分思路。

---

## 12. 已知约束与改进方向

- **数据为教学硬编码**（`BAYES_DATA`、`KNN_RAW_DATA` 等），与课程讲义绑定，非通用数据集。
- **单文件 1065 行**，模块与壳耦合在一个文件，长期建议按模块拆分。
- **KNN 模型复杂度曲线**为示意性合成数据（`performanceData` 含随机项），并非真实交叉验证结果。
- **无测试**：可视化逻辑（距离、卷积、贝叶斯计算）目前缺单元测试，重构时可优先补齐纯计算函数的测试。
- **i18n 双份手工维护**，可引入键一致性校验脚本防止漏翻。
