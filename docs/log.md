# CS Helper (KnowitCS) — 开发日志

> 记录项目的关键进展、决策与变更。最新在上。
> 体例：每条含「日期 / 类型 / 摘要」，类型沿用 Conventional Commits（feat / fix / chore / docs / ci）。

---

## 进展时间线

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

- [ ] `App.jsx` 约 1065 行，所有模块与外壳耦合在单文件——长期应按模块拆到 `src/modules/`。
- [ ] 纯计算逻辑（距离、卷积、贝叶斯）**无单元测试**，重构前应优先补齐。
- [ ] `en.json` / `zh.json` 双份手工维护，缺键一致性校验，易漏翻。
- [ ] KNN「K vs 误差」曲线为示意性合成数据，非真实交叉验证结果。
- [ ] 教学数据（`BAYES_DATA`、`KNN_RAW_DATA`）硬编码，与讲义绑定。

---

## 维护约定

- 每次合并到 `main` 的有意义变更，在「进展时间线」追加一条（最新在上）。
- 引入新约束或填掉技术债时，同步更新「已知问题」清单。
- 涉及架构/技术栈的决策写入 [design.md](./design.md)；未来规划写入 [plan.md](./plan.md)。
