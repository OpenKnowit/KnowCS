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

### 服务器与域名（2026-10-05 已核实）

| 项目 | 值 |
|---|---|
| GitHub | `OpenKnowit/KnowCS`，分支 `main` |
| SSH | `ssh pastpaper`，用户 `ubuntu`，IP `129.226.210.66` |
| 腾讯云实例 | 新加坡 `ap-singapore`，`lhins-ir66ks5j`，名称 `Ubuntu22.04-1Panel-Cfdo` |
| DNSPod | `knowcs.online` 根域名 A 记录 `2424635031` → `129.226.210.66`，默认线路，TTL 600 |
| OpenResty 容器 | `1Panel-openresty-QDQf` |
| 站点配置 | 宿主机 `/opt/1panel/www/conf.d/knowcs.online.conf` |
| 站点文件 | 宿主机 `/opt/1panel/www/sites/knowcs`；容器内 `/www/sites/knowcs` |
| HTTPS | Certbot webroot `/opt/1panel/www/sites/knowcs/acme`；证书 `/etc/letsencrypt/live/knowcs.online/` |
| 证书续期 | `certbot.timer`；钩子 `/etc/letsencrypt/renewal-hooks/deploy/knowcs.sh` 将证书复制到站点 `ssl/` 并校验、reload OpenResty |
| 初次发布 | `releases/20261005-2158`；对应代码提交 `bd0fbbe` |

**不要混淆实例**：`ssh pastpaper` 对应上表的 `lhins-ir66ks5j`，不是名字为 `PastpaperMaster` 的 `lhins-ik4lnte5`。后续操作前重新核实 SSH、云实例和 DNS，不能把这里的快照当作永久不变的状态。腾讯云/DNS 操作使用用户级 `tencent-cloud-ops` 技能及本机 TCCLI；认证过期由用户执行 `tccli auth login`。

### 推送与发布是两步

- **GitHub 推送**只触发 `Validate KnowCS`，不会更新服务器。完成推送后仍需上传已验证的 `dist/index.html`、核对 SHA-256、原子切换 `current`。
- GitHub SSH 曾连接失败；可使用已有 `gh` 登录凭据通过 HTTPS 推送，不在命令中放 token：

  ```bash
  git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push https://github.com/OpenKnowit/KnowCS.git main
  gh run list --repo OpenKnowit/KnowCS --limit 3
  ```

- 修改 `.github/workflows/` 需要 `gh` 的 `workflow` 权限；若 GitHub 明确拒绝，可执行 `gh auth refresh -h github.com -s workflow` 并由用户完成浏览器授权。
- 静态内容发布不需要重启或 reload OpenResty。只有修改站点配置/更新证书时，先备份，再执行 `sudo docker exec 1Panel-openresty-QDQf nginx -t`，通过后 `nginx -s reload`；不要重启容器。
- 验证 HTTPS 200、HTTP 301、gzip、线上/本地 HTML SHA-256、真实浏览器桌面/手机行为，并确认 `mc.iloveust.com`、`pastpaper.knowit.top` 仍正常。
- 回滚内容时原子地将 `current` 指回旧 release；初次上线前配置备份在站点 `backups/openresty-conf-before.tgz`。保留旧版本和备份，不要覆盖其他站点。
- `designs/knowcs-all.pen` 是用户独立修改，提交时排除；`work/` 的 DNS 快照也不入库。任何密钥、token、私钥均不得写入文档、memory 或 Git。

## 工作约定

- **部署门禁四连**：提交/部署前确保 `npm run lint`、`npm test`、`npm run typecheck`、`npm run build` 均通过。CI 只校验，服务器发布需单独执行。
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
