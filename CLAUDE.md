# CLAUDE.md

本文件为 Claude Code（及其他 AI 助手 / 新成员）提供项目上手指引。**动手前请先读完本页并按需查阅 `docs/` 下的三份文档。**

## 项目简介

CS Helper (KnowitCS) 是面向 HKUST COMP2211（机器学习）的**交互式学习网站**：纯前端、单页、单文件打包，通过可视化模块帮助学生理解机器学习核心概念。在线 Demo：中文 <https://knowcs.online>。

## 文档导航（docs/）

> 三份文档分工明确，按需查阅：

- **[设计文档 docs/design.md](./docs/design.md)** —— **怎么做**：技术栈、整体架构、各可视化模块设计、i18n、构建部署、扩展指引与已知约束。改代码前的权威参考。
- **[开发日志 docs/log.md](./docs/log.md)** —— **做了什么**：关键进展、决策与变更时间线，以及技术债清单。
- **[路线规划 docs/plan.md](./docs/plan.md)** —— **接下来做什么**：愿景、近期/中长期计划、候选模块池与立项步骤。

## 技术栈速览

React 19 · TypeScript（strict）· Vite 7 · Tailwind CSS 4 · Framer Motion 12（`LazyMotion` + `m.*`）· KaTeX · react-i18next · `vite-plugin-singlefile`（全站内联为单个 HTML）。图表用自研 SVG `LineChart`；Markdown 笔记在构建期预渲染（`scripts/vite-plugins.mjs`）。

## 常用命令

```bash
npm install       # 安装依赖
npm run dev       # 开发服务器（http://localhost:5174）
npm run lint      # ESLint 检查（部署门禁）
npm test          # Vitest 单元测试（部署门禁）
npm run typecheck # tsc --noEmit 严格类型检查（部署门禁）
npm run build     # 生产构建：tsc --noEmit && vite build → dist/ 单 HTML（部署门禁）
npm run preview   # 预览构建产物
```

## 构建与服务器部署

生产域名：<https://knowcs.online>。托管在 `ssh pastpaper` 服务器，由现有 1Panel OpenResty 提供静态文件和 HTTPS。

GitHub Actions（`.github/workflows/deploy.yml`）只执行 Node 22 下的 `npm ci`、lint、test、typecheck、build，不再自动发布到 PinMe，也不消耗 PinMe 余额。

发布步骤见 `deploy/README.md`。站点目录为 `/opt/1panel/www/sites/knowcs`；`releases/<timestamp>/index.html` 保存每次构建，`current` 软链接指向线上版本。通过原子替换软链接发布，保留旧版本供回滚。Nginx 配置模板为 `deploy/knowcs.online.conf`，证书由 Certbot webroot 签发及自动续期。

## 工作约定

- **部署门禁四连**：提交/部署前确保 `npm run lint`、`npm test`、`npm run typecheck`、`npm run build` 均通过，否则 CI 中断、不会部署到 PinMe。
- **计算逻辑放 `src/lib/`**：可视化模块的纯计算（距离/卷积/概率等）一律提取为 `src/lib/` 纯函数并配套 `*.test.ts`，组件只负责渲染与交互。
- **i18n 同步**：新增/修改文案必须同时更新 `src/locales/en.json` 与 `src/locales/zh.json`，键严格对齐；繁体 `zh-HK.json` **由脚本生成、严禁手改**——改完 `zh.json` 后运行 `npm run gen:zh-hk`（OpenCC 简→港繁 + 香港术语映射，映射表见 `scripts/gen-zh-hk.mjs` 的 `HK_TERMS`）。
- **新增模块**：照 [docs/design.md](./docs/design.md) 第 11 节 / [docs/plan.md](./docs/plan.md) 第 6 节的步骤执行。
- **文档维护**：架构决策写入 `design.md`；完成的变更追加到 `log.md`；规划调整更新 `plan.md`。

## 目录要点

- `src/App.tsx` —— 根组件：`COURSE_TABS` 注册表 + Hash 路由（`#/course/<id>`、`#/package/<id>`、`#/extend/<id>`）。
- `src/modules/*.tsx` —— 8 个课程模块 + Package / Extend 视图（Perceptron / PyTorch 已写好、暂未上架）。动画一律用 `m.*`，不要用 `motion.*`。
- `src/components/*.tsx` —— 共享组件（Latex / LineChart / SectionTitle / SeniorAdvice）。
- `src/lib/*.ts` —— 纯计算逻辑（knn / bayes / backprop / kmeans / alphabeta / chart / route / lang …）+ 同目录 Vitest 单元测试。
- `src/types.ts` + `src/data/constants.ts` —— 共享类型定义与类型化数据常量。
- `src/i18n.ts` + `src/locales/` —— 国际化初始化与三语文案。`<Trans>` 一律用 `components={{1: …}}` 显式映射，文案里不要写 Markdown。
- `scripts/vite-plugins.mjs` —— 构建期插件（KaTeX 字体瘦身、`*.md?html` 预渲染）。
- `dist/` —— 构建产物，不入库（CI 每次重新构建）。
- `src/ref/` —— 历史参考版本（.jsx），**不参与构建、已在 lint/tsc 忽略**。
- `docs/` —— 设计 / 日志 / 计划三份文档（见上）。
