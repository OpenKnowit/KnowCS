# KnowCS —— COMP2211 交互式学习实验室

**中文** | [English](./README.md)

KnowCS 是面向香港科技大学 **COMP2211（Exploring Artificial Intelligence）** 的交互式学习网站。每个页面都围绕期中、期末反复考到的概念设计，画面上的每个数字都由经过测试的代码实时算出。

**在线访问：** <https://knowcs.online> —— English / 简体中文 / 繁體中文（`?lang=en`、`?lang=zh`、`?lang=zh-HK`）。

## 网站内容

| 栏目 | 地址 | 内容 |
|---|---|---|
| 课程地图 | `/` | 第 1–11 讲，每讲的练习页、讲解视频与考试标记。 |
| 练习页 | `/<主题>/` | 20 个动手页面（NumPy、贝叶斯、KNN、模型评估、K-Means、感知机、反向传播、卷积、Otsu、仿射变换、CNN 形状（含 Keras/PyTorch 代码）、PyTorch、α-β 剪枝……），大多可直接填写考试要求的表格，并有「自测」模式。 |
| 讲解视频 | `/watch/` | 17 集在浏览器中运行的 3Blue1Brown 风格动画讲解：章节、字幕、「停下来想一想」、文字稿，以及附往年题的考点角。 |
| 概念自测 | `/drill/` | 58 道仿期末 Problem 1 的原创判断题，附解释。 |
| 公式表 | `/formulas/` | 考试用到的全部公式及陷阱提示，可打印为三页 A4。 |
| 考点地图 | `/papers/` | 九份往年试卷分别考了哪些主题。 |
| 资料包 / 拓展 | `/notes/`、`/extend/` | 资料包笔记（NumPy 附实时 API 面板、pandas、PyTorch……）与拓展页面。 |

## 开发

```bash
npm install
npm run dev        # http://localhost:5174
npm run lint       # ESLint
npm test           # Vitest（400+ 个测试）
npm run typecheck  # tsc --noEmit
npm run build      # 多页构建，输出到 dist/
```

技术栈：React 19、TypeScript（strict）、Vite 7（多页，页面清单在 `src/lib/sitemap.ts`）、Tailwind CSS 4、Framer Motion、KaTeX、react-i18next。所有计算都在 `src/lib/` 并配有单元测试；讲解视频是 `src/site/watch/` 中按时间轴驱动的 React/SVG 场景。贡献约定见 [CLAUDE.md](./CLAUDE.md)，开发日志与规划见 [docs/](./docs/)。

## 致谢

早期内容参考了 [moyunxiang/COMP2211](https://github.com/moyunxiang/COMP2211/blob/main/COMP2211.md) 的学习笔记。考试例题均依据课程讲义与往年试卷改写或重新计算，仅供学习使用。
