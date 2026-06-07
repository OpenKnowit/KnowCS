# CS Helper (KnowitCS) — 开发日志

> 记录项目的关键进展、决策与变更。最新在上。
> 体例：每条含「日期 / 类型 / 摘要」，类型沿用 Conventional Commits（feat / fix / chore / docs / ci）。

---

## 进展时间线

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
