# CLAUDE.md

本文件为 Claude Code（及其他 AI 助手 / 新成员）提供项目上手指引。**动手前请先读完本页并按需查阅 `docs/` 下的三份文档。**

## 项目简介

CS Helper (KnowitCS) 是面向 HKUST COMP2211（机器学习）的**交互式学习网站**：纯前端、单页、单文件打包，通过可视化模块帮助学生理解机器学习核心概念。在线 Demo：中文 <https://knowcs.pinme.dev>。

## 文档导航（docs/）

> 三份文档分工明确，按需查阅：

- **[设计文档 docs/design.md](./docs/design.md)** —— **怎么做**：技术栈、整体架构、各可视化模块设计、i18n、构建部署、扩展指引与已知约束。改代码前的权威参考。
- **[开发日志 docs/log.md](./docs/log.md)** —— **做了什么**：关键进展、决策与变更时间线，以及技术债清单。
- **[路线规划 docs/plan.md](./docs/plan.md)** —— **接下来做什么**：愿景、近期/中长期计划、候选模块池与立项步骤。

## 技术栈速览

React 19 · Vite 7 · Tailwind CSS 4 · Framer Motion 12 · KaTeX · Recharts 3 · react-i18next · `vite-plugin-singlefile`（全站内联为单个 HTML）。

## 常用命令

```bash
npm install      # 安装依赖
npm run dev      # 开发服务器（http://localhost:5174）
npm run build    # 生产构建 → dist/ 单 HTML
npm run preview  # 预览构建产物
npm run lint     # ESLint 检查（部署门禁，必须通过）
```

## 工作约定

- **lint 是部署门禁**：提交/部署前确保 `npm run lint` 通过，否则 CI 不会部署到 PinMe。
- **i18n 双份同步**：新增/修改文案必须同时更新 `src/locales/en.json` 与 `src/locales/zh.json`，键严格对齐。
- **新增模块**：照 [docs/design.md](./docs/design.md) 第 11 节 / [docs/plan.md](./docs/plan.md) 第 6 节的步骤执行。
- **文档维护**：架构决策写入 `design.md`；完成的变更追加到 `log.md`；规划调整更新 `plan.md`。

## 目录要点

- `src/App.jsx` —— 主应用，目前所有可视化模块集中于此（约 1065 行）。
- `src/i18n.js` + `src/locales/` —— 国际化初始化与中英文案。
- `src/ref/` —— 历史参考版本，**不参与构建**。
- `docs/` —— 设计 / 日志 / 计划三份文档（见上）。
