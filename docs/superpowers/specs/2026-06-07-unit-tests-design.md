# 单元测试体系设计（Vitest + 纯函数提取）

日期：2026-06-07 ｜ 状态：已批准

## 目标

为 6 个可视化模块的计算逻辑建立单元测试，清偿 log.md 技术债「纯计算逻辑无单元测试」，落实 plan.md P0 的 Vitest 规划。

## 方案

提取纯函数 + 逻辑测试（不做组件渲染测试）：

- devDependency 仅新增 `vitest`，不引入 jsdom / testing-library。
- 新增 `src/lib/`，将各模块组件内嵌于 `useMemo`/handler 的计算体**原样搬出**为纯函数，组件改为 import 调用。不改算法、不改 UI、不动 i18n。
- 测试文件与源文件同目录共置（`src/lib/*.test.ts`）。
- `src/data/constants.ts` 数据常量保持原位，纯函数通过参数接收数据。

## 模块拆分

| 文件 | 提取内容 |
|---|---|
| `lib/knn.ts` | 均值/标准差统计、标准化、欧氏距离、K 近邻排序+投票、合成复杂度曲线 |
| `lib/kernel.ts` | 单像素卷积（\|sum\| + 0-255 clamp、边缘 padding）、6×6 源图生成 |
| `lib/bayes.ts` | 贝叶斯后验 P(F\|S)；朴素贝叶斯：先验、似然、α 平滑、连乘/log 双模式、归一化 |
| `lib/numpy.ts` | 三种切片模式（slice/fancy/mask）选中单元格判定 |

> 实施修正：Backprop 模块经核查**无计算逻辑**（公式为静态 KaTeX 展示，errorVal 仅用于显示），不提取、不建 `lib/backprop.ts`。

## 测试要求（每模块 5-10 用例）

- 黄金值：朴素贝叶斯 α=0 → YES 20.46% / NO 79.54%（讲义 P.29）；火警案例 9.00%；Sobel-X 已知图卷积输出。
- 边界：α 平滑消除零频率；log 与连乘模式一致（容差）；σ=0 标准化；卷积边缘；clamp 上下限；K 投票平票。

## 门禁集成

- `package.json`：`"test": "vitest run"`。
- `.github/workflows/deploy.yml`：lint 与 typecheck 之间插入 `npm run test`。
- CLAUDE.md「部署门禁三连」→ 四连（lint / test / typecheck / build）。

## 验收

lint / test / typecheck / build 全绿；dev 目检 6 模块 UI 行为不变。
