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

React 19 · TypeScript（strict）· Vite 7 · Tailwind CSS 4 · Framer Motion 12 · KaTeX · Recharts 3 · react-i18next · `vite-plugin-singlefile`（全站内联为单个 HTML）。

## 常用命令

```bash
npm install       # 安装依赖
npm run dev       # 开发服务器（http://localhost:5174）
npm run lint      # ESLint 检查（部署门禁）
npm run typecheck # tsc --noEmit 严格类型检查（部署门禁）
npm run build     # 生产构建：tsc --noEmit && vite build → dist/ 单 HTML（部署门禁）
npm run preview   # 预览构建产物
```

## CI/CD 自动部署

GitHub Actions 自动构建并部署到 PinMe（IPFS），线上地址 <https://knowcs.pinme.dev>。配置见 `.github/workflows/deploy.yml`。

**触发规则**
- push 到 `main` 自动触发；也可在仓库 **Actions → Build & Deploy to PinMe → Run workflow** 手动触发（`workflow_dispatch`）。
- **纯文档/配置改动不触发部署**：当一次 push 的改动**全部**命中 `**.md` / `.gitignore` / `LICENSE` / `docs/**` 时跳过；只要含任意源码或构建文件即照常部署。手动触发不受此限制。

**流程（任一步失败即中断，不部署）**
1. `npm ci` 安装依赖
2. `npm run lint` —— ESLint
3. `npm run typecheck` —— `tsc --noEmit` 严格类型检查
4. `npm run build` —— `tsc --noEmit && vite build`，产出单文件 `dist/index.html`
5. `pinme set-appkey` + `pinme upload dist --domain knowcs` 发布到 `knowcs` 子域名

**密钥**：PinMe appkey 存于 GitHub 仓库 Secret `PINME_APPKEY`，**严禁**写入仓库任何文件。token 会过期（约 2026-06-30），过期后在 PinMe 后台重新生成并 `gh secret set PINME_APPKEY` 更新。

**注意**：每次部署绑定域名会消耗 PinMe 钱包余额，留意余额。

## 工作约定

- **部署门禁三连**：提交/部署前确保 `npm run lint`、`npm run typecheck`、`npm run build` 均通过，否则 CI 中断、不会部署到 PinMe。
- **i18n 同步**：新增/修改文案必须同时更新 `src/locales/en.json` 与 `src/locales/zh.json`，键严格对齐；繁体 `zh-HK.json` **由脚本生成、严禁手改**——改完 `zh.json` 后运行 `npm run gen:zh-hk`（OpenCC 简→港繁 + 香港术语映射，映射表见 `scripts/gen-zh-hk.mjs` 的 `HK_TERMS`）。
- **新增模块**：照 [docs/design.md](./docs/design.md) 第 11 节 / [docs/plan.md](./docs/plan.md) 第 6 节的步骤执行。
- **文档维护**：架构决策写入 `design.md`；完成的变更追加到 `log.md`；规划调整更新 `plan.md`。

## 目录要点

- `src/App.tsx` —— 根组件：侧栏导航 + 各模块组合。
- `src/modules/*.tsx` —— 6 个可视化模块（Knn / BayesBasics / NaiveBayes / Numpy / Backprop / Kernel）。
- `src/components/*.tsx` —— 共享组件（Latex / SectionTitle / SeniorAdvice）。
- `src/types.ts` + `src/data/constants.ts` —— 共享类型定义与类型化数据常量。
- `src/i18n.ts` + `src/locales/` —— 国际化初始化与中英文案。
- `src/ref/` —— 历史参考版本（.jsx），**不参与构建、已在 lint/tsc 忽略**。
- `docs/` —— 设计 / 日志 / 计划三份文档（见上）。
