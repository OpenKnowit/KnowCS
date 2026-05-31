# CS Helper (KnowitCS) — 路线规划

> 项目的目标、计划与待办。与 [design.md](./design.md)（怎么做）、[log.md](./log.md)（做了什么）配套：本文件回答「接下来做什么」。

---

## 1. 项目愿景

面向 HKUST COMP2211（机器学习）的交互式学习网站，用**可视化 + 即时反馈**帮助学生理解抽象概念。最终目标是作为 [moyunxiang/COMP2211](https://github.com/moyunxiang/COMP2211/blob/main/COMP2211.md) 学习笔记的**交互式补充**，并以 MIT 协议开源供学生学习参考。

---

## 2. 现状（Done）

- ✅ 单页骨架：React 19 + Vite 7 + Tailwind 4，单文件打包。
- ✅ 中英双语 i18n（react-i18next + 浏览器语言检测）。
- ✅ 六大可视化模块：NumPy 内存 / 反向传播 / 卷积 / 贝叶斯基础 / 朴素贝叶斯 / KNN。
- ✅ CI 自动部署到 PinMe（`knowcs.pinme.dev`），lint 作为部署门禁。
- ✅ 文档体系：design / log / plan 三件套 + 根目录 `CLAUDE.md` 索引。

---

## 3. 近期计划（Next）

> 优先级：P0 最高。

| 优先级 | 事项 | 说明 |
|--------|------|------|
| P0 | 纯计算函数补单测 | 为距离、卷积、贝叶斯等抽出纯函数并加单元测试，作为后续重构的安全网 |
| P0 | `App.jsx` 模块拆分 | 1065 行单文件按模块拆到 `src/modules/`，外壳仅保留路由与布局 |
| P1 | i18n 键一致性校验 | 加脚本校验 `en.json` / `zh.json` 键对齐，纳入 lint / CI |
| P1 | 新增可视化模块 | 按 Desmond 教授讲义持续补充（梯度下降、决策树、聚类等候选） |
| P2 | KNN 曲线真实化 | 用真实交叉验证替换示意性合成的「K vs 误差」数据 |

---

## 4. 中长期方向（Later）

- **内容对齐讲义**：按课程进度持续扩充模块，覆盖 COMP2211 主要知识点。
- **回流开源**：COMP2211 部分合并进 `moyunxiang/COMP2211` 作为交互补充，MIT 开源。
- **工程化**：模块拆分完成后，评估引入轻量测试框架（Vitest）与组件级测试。
- **可维护性**：考虑教学数据与组件解耦（数据 JSON 化），便于非开发者贡献题目/数据集。

---

## 5. 候选模块池（Backlog）

> 尚未排期，按需从中挑选立项。

- [ ] 梯度下降 / 学习率可视化
- [ ] 决策树 / 信息增益
- [ ] K-Means 聚类
- [ ] 线性 / 逻辑回归决策边界
- [ ] 混淆矩阵与评估指标（Precision/Recall/F1）
- [ ] PCA 降维直观演示

---

## 6. 新增模块清单（落地步骤）

> 摘自 [design.md](./design.md) 第 11 节，立项后照此执行：

1. 在 `App.jsx`（或拆分后的 `src/modules/`）编写 `XxxModule`，状态自管理，派生值用 `useMemo`。
2. `tabs` 数组追加 `{ id, label, icon, subtitle }`。
3. 内容区 `AnimatePresence` 内增加 `activeTab === 'xxx'` 分支。
4. `en.json` / `zh.json` 同步补齐文案键。
5. 用 `SeniorAdvice` 收尾，给出考试 / 易错点提示。

---

> 完成或调整计划时，请同步更新本文件，并在 [log.md](./log.md) 记录对应进展。
