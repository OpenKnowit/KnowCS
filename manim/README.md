# KnowCS Manim 讲解视频

按考试权重挑选的 5 个 COMP2211 主题，每个一段 Manim 动画（约 1.5–2 分钟，1080p60，屏幕文字为英文）。

| # | 文件 / 场景 | 主题 | 考频 | 数据来源 |
|---|---|---|---|---|
| 1 | `s1_xor.py` · `XorHiddenLayer` | 为什么 XOR 需要隐藏层：直线全失败 → 线性映射拉伸 → sigmoid 挤压 → 隐空间一条直线；结尾 x₁² 特征把边界变成抛物线 | 6 套试卷 | 2023 Fall Q9（抛物线）、2025 Spring Q6（x₁²+x₂² 仍不可分）、2025 Spring Q7c（OR/NAND） |
| 2 | `s2_broadcast.py` · `BroadcastDistances` | 广播求两两距离：3D 方块演示 `X[:, None]`、拉伸到 (n, m, d)、相减平方、沿 axis 2 求和；后半段 ‖x−y‖² 展开对应 NumPy 代码 | 6/6 期中 | 自拟 3×2 / 4×2 小例子 |
| 3 | `s3_bayes.py` · `BayesBaseRate` | 10,000 个点的病毒检测：只有约 2% 的阳性真感染 → Bayes 公式 → 朴素贝叶斯连乘，一个 0 抹掉全部证据 → α-Laplace 平滑 | 8/9 试卷 | Lecture 2 病毒例（0.1% / 99% / 95%）、疾病 Z 例 |
| 4 | `s4_backprop.py` · `BackpropXor` | 反向传播（考试记号）：前向数值流动、链式法则逐因子推出 δₖ、δ 回传得 δⱼ、`w ← w − ηδO`；结尾学习率过大/过小的梯度下降 | 3/3 期末 | Lecture 6 XOR 网络（w₁=−0.65 … η=0.5）Round 1 Step 1/2、10000 轮结果 |
| 5 | `s5_cnn.py` · `ConvToCnn` | 卷积到 CNN 形状：滑动卷积核看输出变大变小，从动画里读出 ⌊(N−K+2P)/S⌋+1；权重共享 vs 全连接参数量；2022 期末 CNN 2.07M vs MLP 150.5M | 3/3 期末 | Lecture 8（7×7、32×32×3 例）、2022 Spring Final Part B Q1 |

**勘误提示**：Lecture 6 Round 1 Step 2 的 δⱼ₁ 印成 −0.027463（两位数字颠倒），按讲义公式重算为 −0.027643，新 w₂ 相应为 0.653822（讲义 0.653732）。视频 4 使用重算值并在画面底部注明。

## 渲染

需要 Python 3.10+、LaTeX（含 `standalone`、`dvisvgm`）、ffmpeg、cairo/pango（macOS 另需 `brew install pkgconf` 才能编译 pycairo）。

```bash
python3 -m venv .venv && .venv/bin/pip install manim   # Manim Community v0.21
MANIM=.venv/bin/manim ./render.sh          # 全部 5 段，1080p60
MANIM=.venv/bin/manim ./render.sh l 1 3    # 480p 快速预览第 1、3 段
./sheet.sh media/videos/s1_xor/480p15/XorHiddenLayer.mp4   # 生成缩略图联系表检查排版
```

输出在 `media/videos/<文件>/<分辨率>/<场景>.mp4`（`media/` 不入库）。共享配色（与站点 slate/blue 一致）与标题、字幕辅助函数在 `common.py`。
