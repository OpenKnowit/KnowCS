# CS Helper (KnowitCS) — 路线规划

> 项目的目标、计划与待办。与 [design.md](./design.md)（怎么做）、[log.md](./log.md)（做了什么）配套：本文件回答「接下来做什么」。

---

## 1. 项目愿景

面向 HKUST **COMP2211（Exploring Artificial Intelligence）** 的交互式学习网站，用**可视化 + 即时反馈**帮助学生理解课程中的抽象算法。模块内容以课程**讲义 + Lab + 历年试卷**三类官方材料为权威依据（见第 3、4 节），目标是作为 [moyunxiang/COMP2211](https://github.com/moyunxiang/COMP2211/blob/main/COMP2211.md) 学习笔记的**交互式补充**，并以 MIT 协议开源供学生学习参考。

---

## 2. 现状（Done）

- ✅ 单页骨架：React 19 + Vite 7 + Tailwind 4，单文件打包。
- ✅ **全量 TypeScript（TSX）**：核心代码已从 JSX 迁移到严格 TSX，`tsc --noEmit` 纳入 `build` 与 `typecheck`。
- ✅ **模块化结构**：`App.tsx` 瘦身为 ~209 行外壳（路由 + 布局）；六大模块独立于 `src/modules/`，通用组件在 `src/components/`，教学数据集中于 `src/data/constants.ts`，类型在 `src/types.ts`。
- ✅ 中英双语 i18n（react-i18next + 浏览器语言检测）。
- ✅ 六大可视化模块（见下方覆盖表）。
- ✅ CI 自动部署到 PinMe（`knowcs.pinme.dev`），lint 作为部署门禁；文档类改动（`*.md` / `docs/`）跳过部署。
- ✅ 文档体系：design / log / plan 三件套 + 根目录 `CLAUDE.md` 索引。

---

## 3. 课程覆盖对照表（Coverage Map）

> 依据 `../lecture/`（讲义）与 `../labs/`（Lab）——两者按 L1–L11 **一一对应**。优先补齐尚未覆盖的内容。

| 讲义 / Lab | 主题 | 对应模块 | 状态 |
|------------|------|----------|------|
| L1 / Lab1 | NumPy & 图像基础（array/切片/broadcast/灰度） | NumpyModule | ✅ 已覆盖 |
| L2 / Lab2 | 朴素贝叶斯 | BayesBasics + NaiveBayes | 🟡 缺数值特征（Gaussian）似然 |
| L3 / Lab3 | KNN | KnnModule | ✅ 已覆盖 |
| **L4 / Lab4** | **K-Means 聚类** | KMeansModule | ✅ 已上线（2026-06-07） |
| L5 / Lab5 | 人工神经元 / 感知机 | BackpropModule（仅 MLP 侧） | 🟡 缺单神经元（错误驱动更新） |
| L6 / Lab6 | 多层感知机 / 反向传播 | BackpropModule | ✅ 已覆盖 |
| L7 / Lab7 | 图像处理 / 卷积 / **数据增强** | KernelModule（仅卷积） | 🟡 缺数据增强 |
| **L8 / Lab8** | **卷积神经网络（池化/特征图）** | KernelModule（部分） | 🟡 部分 |
| **L9 / Lab9** | **PyTorch（张量/自动求导）** | — | ❌ 缺口（偏工具） |
| **L10 / Lab10** | **Minimax + Alpha-Beta 剪枝** | AlphaBetaModule | ✅ 已上线（2026-06-07） |
| L11 / Lab11 | 复习 / 收尾 | —（非可视化） | n/a |

---

## 4. 考点与资源对齐（Exam Alignment）

> 来源：`../Pastpaper/`（历年期中/期末/Sample + 解答）、`../labs/pa/`（PA1/PA2）、`Pastpaper/COMP2211midterm/HW/review.md`（期中复习大纲）。模块的「学长寄语 / 易错点」应直接取材于此，而非凭空编写。

**高频考点信号**（据 review.md）：期中重点考 **Lab2–6 的公式与算法步骤**，尤其要求**手推 Naïve Bayes 公式**与 **K-Means 迭代过程**——这正是交互可视化最能帮忙的地方，据此抬高 K-Means 与贝叶斯模块的内容优先级。

**现有模块需补强的考点**：
- **朴素贝叶斯**：补**数值特征的 Gaussian 似然**（目前仅 categorical + Laplace）；考点强调「分类变量 vs 数值变量处理方式不同」。
- **KNN**：讲清**向量化距离** `‖x−y‖² = ‖x‖² + ‖y‖² − 2x·y` 与**标准化必要性**。
- **卷积（Kernel）**：Lab7 还考**数据增强**（crop / flip / rotate / translate / mixup），当前模块仅卷积/边缘检测。
- **反向传播（Backprop）**：Lab5 的**单神经元/感知机**（`sign(w·x+b)`、错误驱动更新 `δ=y−ŷ`、线性可分、`[w,b]` 合并）是独立考点，与 MLP 区分。

**反复出现的跨模块概念**（来自 PA1/PA2，暂无模块）：**归一化对比**（Min-Max vs Z-score）、**Data Leakage**（统计量只能在训练集上算）、**Decision Stump**（单特征单阈值二分类）。

---

## 5. 进行中（In Progress）

| 事项 | 状态 | 说明 |
|------|------|------|
| **Alpha-Beta 剪枝模块（L10）** | ✅ 已集成上线（2026-06-07） | `module/alphabeta.html` / `alphabeta_aligned.html` 为独立 HTML 沙盒（DFS + α/β 剪枝逐步追踪，含 Tic-Tac-Toe 式博弈树）。下一步：移植为 `src/modules/AlphaBetaModule.tsx`，接入 Tab 路由与 i18n，按第 8 节落地步骤执行 |

---

## 6. 近期计划（Next）

> 优先级：P0 最高。

| 优先级 | 事项 | 说明 |
|--------|------|------|
| ~~P0~~ ✅ | Alpha-Beta 模块集成 | 已完成：`src/modules/AlphaBetaModule.tsx` + `lib/alphabeta.ts` + 6 测试，三语 i18n |
| P0 | 引入 Vitest + 纯函数单测 | 模块/数据已拆分到位，为距离、卷积、贝叶斯等派生计算抽出纯函数并补单测，作为后续重构与改数据的安全网 |
| ~~P1~~ ✅ | **新增 K-Means 模块（L4）** | 已完成：交互选 K、EM 迭代质心动画、Elbow Method、Z-score 标准化（`KMeansModule` + `lib/kmeans.ts` + 9 测试） |
| P1 | **贝叶斯模块补 Gaussian 似然** | 让 NaiveBayes 支持数值特征，覆盖 Lab2「分类 vs 数值」考点 |
| P1 | i18n 键一致性校验 | 加脚本校验 `en.json` / `zh.json` 键对齐，纳入 lint / CI，防漏翻 |
| P1 | 文档同步 | `design.md` 第 3/4/12 节仍按旧的「单文件 App.jsx 1065 行」描述，需更新为 TSX + 模块化现状；`log.md` 补记 TSX 迁移、模块拆分两条进展 |
| P2 | 单神经元 / 感知机模块（L5） | 错误驱动更新动画 + 线性可分判定，补 Lab5 考点（与 MLP 区分）|
| P2 | 卷积模块扩展（L7 数据增强 / L8 CNN） | 在 KernelModule 上补数据增强演示与池化/多层特征图 |
| P2 | KNN 曲线真实化 | 用真实交叉验证替换示意性合成的「K vs 误差」数据 |

---

## 7. 候选模块池（Backlog）

> 按 COMP2211 讲义/Lab 缺口排序，立项后从中挑选。（K-Means、Gaussian 似然、感知机、卷积扩展、Alpha-Beta 已分别进入第 5/6 节。）

- [ ] **L9 PyTorch**：张量 / 自动求导（autograd）计算图与梯度回传可视化（偏工具，价值待评估）
- [ ] **归一化对比小工具**：Min-Max vs Z-score 的直观对比 + **Data Leakage** 提示（贯穿 KNN / K-Means / PA1）
- [ ] **Decision Stump**：单特征单阈值分类的决策直觉（PA1 天体分类）
- [ ] **L11 / 复习**：知识点（有监督 vs 无监督）总览与自测式回顾

> 注：旧版 backlog 中的「决策树 / 逻辑回归 / PCA / 混淆矩阵」不在 COMP2211 讲义/Lab 范围内，已移除以保持与课程对齐；如未来大纲调整再行评估。

---

## 8. 新增模块清单（落地步骤）

> 摘自 [design.md](./design.md) 第 11 节，立项后照此执行：

1. 在 `src/modules/` 新建 `XxxModule.tsx`，状态自管理、派生值用 `useMemo`；教学数据放入 `src/data/constants.ts`，相关类型补到 `src/types.ts`。
2. 在 `App.tsx` 的 `tabs` 数组追加 `{ id, label, icon, subtitle }`。
3. 在内容区 `AnimatePresence` 内增加 `activeTab === 'xxx'` 分支（配 `SectionTitle` + 模块）。
4. 在 `en.json` / `zh.json` 同步补齐文案键。
5. 用 `SeniorAdvice` 收尾，**易错点取材于 `../Pastpaper` 与 review.md**（见第 4 节），给出真实考点提示。
6. 提交前确保 `npm run lint` 与 `npm run typecheck` 均通过。

---

## 9. 中长期方向（Later）

- **内容对齐讲义/Lab**：按覆盖表持续补齐缺口（L4 → L5 → L7/L8 → L9），目标是全部可视化型讲义都有交互模块。
- **考点驱动**：以 `../Pastpaper` 历年题型反向打磨模块——优先把「手推型」考点（NB 公式、K-Means 迭代、Minimax 树）做成可单步演练的交互。
- **回流开源**：合并进 `moyunxiang/COMP2211` 作为交互补充，MIT 开源。
- **工程化**：Vitest 落地后，评估补充组件级测试与 CI 测试门禁。
- **可维护性**：教学数据已集中于 `src/data/constants.ts`，可进一步评估 JSON 化与数据/组件解耦，便于非开发者贡献题目/数据集。

---

### 10. Charles补充内容

加入numpy编程模块。现在太少了，你看看pastpaper里面是怎么考python的，你总结一下。

现在那个链式法则太普通了，给我搞个像考试题型的

Naive Bayese优化你直接找一道相关的考题可视化吧

现在这个反向传播程序也不对，太简单，把代码也给展示出来

加入pytorch语法cheatsheet一页，然后对于一些重要的部分加一些可视化

卷积模块 cnn你要展示一下。

---



> 完成或调整计划时，请同步更新本文件，并在 [log.md](./log.md) 记录对应进展。







