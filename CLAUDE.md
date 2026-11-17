# CLAUDE.md


## 项目简介

KnowCS 是面向 HKUST COMP2211（Introduction to Artificial Intelligence，人工智能导论）的**交互式学习网站**：纯前端、多页静态站（每个模块一个 URL），通过可视化模块帮助学生理解机器学习核心概念与考点。在线 Demo：中文 <https://knowcs.online>。

## 文档导航（docs/）


- **[开发日志 docs/log.md](./docs/log.md)** —— **做了什么**：关键进展、决策与变更时间线，以及技术债清单。
- **[路线规划 docs/plan.md](./docs/plan.md)** —— **接下来做什么**：愿景、近期/中长期计划、候选模块池与立项步骤。


## 构建与服务器部署

生产域名：<https://knowcs.online>。托管在 `ssh pastpaper` 服务器，由现有 1Panel OpenResty 提供静态文件和 HTTPS。


发布步骤见 `deploy/README.md`。站点目录为 `/opt/1panel/www/sites/knowcs`；`releases/<timestamp>/` 保存每次构建的整个 `dist/`，`current` 软链接指向线上版本。通过原子替换软链接发布，保留旧版本供回滚。Nginx 配置模板为 `deploy/knowcs.online.conf`，证书由 Certbot webroot 签发及自动续期。

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