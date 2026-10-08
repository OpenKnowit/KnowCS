# COMP2211 中的 Matplotlib

> 这门课用 Matplotlib 来「看」东西：第 7 讲的图像、第 4 讲的 K-Means 簇、第 6 讲的 MNIST 网格和混淆矩阵、第 6、8、9 讲的训练曲线。考试很少要你写画图代码，但常问一张图说明了什么，而这些图都出自少数几个调用。下面每段代码都有 **试一试** 按钮，可以在实验台里打开。

## 1. 心智模型：Figure → Axes → 图元

| 对象 | 是什么 | 怎么得到 |
|---|---|---|
| **Figure** | 整张画布（尺寸以英寸计，`figsize=(宽, 高)`） | `plt.figure()`、`plt.subplots()` |
| **Axes** | 画布里的一个子图：有自己的 x/y 范围、刻度、标题、图例 | `plt.subplot(r, c, i)`、`fig, ax = plt.subplots()` |
| **图元（Artist）** | 画上去的东西：线、点、柱、图像、文字 | `ax.plot`、`ax.scatter`、`ax.bar`、`ax.imshow`、`ax.text` |

同一件事有两种写法：

| pyplot 写法（课件用这种） | 面向对象写法 |
|---|---|
| `plt.plot(x, y)` | `ax.plot(x, y)` |
| `plt.title('t')` | `ax.set_title('t')` |
| `plt.xlabel('x')` | `ax.set_xlabel('x')` |
| `plt.xlim(0, 1)` | `ax.set_xlim(0, 1)` |

`plt.*` 函数作用在**当前** Axes 上，也就是最后创建、或用 `plt.subplot` 选中的那个。所以第 7 讲每次 `plt.imshow` 之前都写一行 `plt.figure()`。如果不新建 figure，下一张图会画在旧图上面。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 41)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, 1 / (1 + np.exp(-x)), label='sigmoid')
ax.plot(x, np.maximum(0, x) / 5, '--', label='ReLU / 5')
ax.set_xlabel('z')
ax.set_title('Activation functions')
ax.legend()
```

## 2. 课程用到的调用

| 调用 | 画出 | 课程中的位置 |
|---|---|---|
| `plt.plot(x, y, 'ro--', label=…)` | 折线 / 标记点 | 激活函数、损失曲线 |
| `plt.scatter(x, y, c=labels, cmap=…)` | 带颜色的点 | K-Means 簇（L4）、KNN 数据（L3） |
| `plt.bar(names, heights)` | 柱状图 | 各个 K 的准确率、各类样本数 |
| `plt.hist(values, bins=…)` | 直方图，返回 `(n, bins, patches)` | 像素亮度、Otsu（L7） |
| `plt.imshow(X, cmap=…, vmin=…, vmax=…)` | 图像 | 第 7 讲的所有图片、第 6/8 讲的 MNIST |
| `plt.subplot(rows, cols, i)` | 选中第 *i* 格（从 1 开始） | 5 × 5 的 MNIST 网格（L6） |
| `plt.xticks([])`、`plt.axis('off')` | 隐藏刻度 | 图片网格 |
| `plt.title / xlabel / ylabel / legend` | 标注 | 处处可见 |
| `plt.show()` | 显示（并结束）这张图 | 每个代码格的最后 |

格式字符串：颜色字母（`b g r c m y k w`）+ 标记（`o s ^ x + .`）+ 线型（`- -- : -.`）。`'ro--'` 是红色圆点、虚线相连；`'bs'` 是蓝色方块、不连线。

## 3. 图像（第 7 讲）：`imshow`

**形状。** 灰度图是二维数组 `(H, W)`；彩色图是 `(H, W, 3)`，每个通道（R、G、B）各是一张二维数组。`img[:, :, 0]` 是红色通道。

**坐标。** 第 `r` 行是从顶部往下数的 *y* 位置，第 `c` 列是 *x*。`img[r, c]` 是 x = c、y = r 处的像素。`imshow` 把原点放在左上角，所以 y 轴向下增大。

**数值 → 颜色。** 对二维数组，`imshow` 把每个值经过色图（colormap）映射成颜色：

```text
颜色 = cmap( (v − vmin) / (vmax − vmin) )      # 0 → 色图起点，1 → 色图终点
```

- 不写 `cmap` 时，色图是 `'viridis'`（紫 → 黄），**不是**灰度。
- 不写 `vmin` / `vmax` 时，它们取图片自身的最小值和最大值：最暗的像素变黑、最亮的变白，*不管它们实际的数值是多少*。所以一张暗图看起来也很正常。这就是第 7 讲每次都写 `cmap='gray', vmin=0, vmax=1`（`uint8` 图则 `vmax=255`）的原因。
- `'gray'` 把 0 映射成黑；`'binary'` / `'Greys'` 把 0 映射成白，所以第 6、8 讲里的 MNIST 数字是白底黑字。

```python
import numpy as np
import matplotlib.pyplot as plt

dark = np.array([[0.0, 0.1, 0.2],
                 [0.1, 0.2, 0.3],
                 [0.2, 0.3, 0.3]])
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
axes[0].imshow(dark)
axes[0].set_title('default')
axes[1].imshow(dark, cmap='gray')
axes[1].set_title("cmap='gray'")
axes[2].imshow(dark, cmap='gray', vmin=0, vmax=1)
axes[2].set_title('vmin=0, vmax=1')
```

**彩色图的取值范围。** 浮点数必须在 `[0, 1]`，整数必须在 `[0, 255]`，超出部分会被*截断*并给出警告。卷积或对比度拉伸之后，要么自己先截断（`np.clip(out, 0, 255)`），要么先缩放再显示。

## 4. 图片网格

第 6 讲这样显示 25 张 MNIST 数字：每张图一个 `subplot`，隐藏刻度，下方写标签。

```python
import numpy as np
import matplotlib.pyplot as plt

imgs = np.zeros((4, 5, 5))
imgs[0, :, 2] = 1
imgs[1, 2, :] = 1
imgs[2] = np.eye(5)
imgs[3] = np.flip(np.eye(5), axis=1)
labels = ['|', '-', 'diag', 'anti']
plt.figure(figsize=(6, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imgs[i], cmap=plt.cm.binary)
    plt.xlabel(labels[i])
```

`fig, axes = plt.subplots(2, 3)` 得到一个 2 × 3 的 Axes **数组**：`axes[0, 1]` 是第 0 行第 1 列，`for ax in axes.flat:` 按行逐个访问。只有一行或一列时，它是一维数组（`axes[i]`）。

## 5. 机器学习中的图

**簇（第 4 讲）。** `c=labels` 通过色图给每个点上色，再用一次 `scatter` 标出质心。

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]])
labels = np.array([0, 0, 1, 1, 1, 1])
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=80)
plt.scatter([1.25, 4], [1.5, 5.25], c='black', marker='x', s=120, label='centroids')
plt.legend()
```

**训练曲线（第 6、8、9 讲）。** 以 epoch 为横轴画出 `history.history['loss']` 和 `['val_loss']`。如果训练损失一直下降、验证损失却开始回升，模型就在**过拟合**。提前停止（early stopping）会保留验证损失最低的那个 epoch。

**混淆矩阵（第 3、6 讲）。** 第 6 讲用的是 `seaborn.heatmap(cm, annot=True, fmt='d')`，相当于 `imshow` 再在每格写上数字。行是真实类别，列是预测类别。

```python
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

cm = np.array([[50, 10],
               [5, 35]])
f, ax = plt.subplots(figsize=(4, 4))
sn.heatmap(cm, annot=True, fmt='d', square=True, ax=ax)
```

**决策边界（第 5 讲）。** 感知机的决策边界是直线 w₁x₁ + w₂x₂ + b = 0，用 `plt.plot(xs, -(w1 * xs + b) / w2)` 画出。

## 6. 易错清单

- **同一格里连续画图忘了 `plt.figure()`**：第二张图会叠在第一张上。
- **`imshow` 不写 `vmin` / `vmax`**：显示的是拉伸过对比度的图；不写 `cmap` 时还是 viridis 配色。
- **把 `img[y, x]` 写成 `img[x, y]`**：数组按 `[行, 列]` 也就是 `[y, x]` 索引。
- **彩色图超出 [0, 1] / [0, 255]**：会被截断。
- **OpenCV 的 `cv2.imread` 返回 BGR**：用 `imshow` 显示时红蓝互换。第 7 讲用 `mpimg.imread` 读 PNG，得到 [0, 1] 的 RGB 浮点数。
- **直接画需要梯度的 PyTorch 张量** 会报错，先 `t.detach().numpy()`。
- **`plt.subplot` 的序号从 1 开始**，而 Python 下标从 0 开始，所以写 `plt.subplot(5, 5, i + 1)`。

## 7. 自测

1. 一张 28 × 28 的浮点图，取值在 [0, 0.3]。`plt.imshow(img, cmap='gray')` 显示什么？`plt.imshow(img, cmap='gray', vmin=0, vmax=1)` 又显示什么？
2. `fig, axes = plt.subplots(2, 3)`：`axes.shape` 是多少？`axes[1, 0]` 是哪一格？
3. `n, bins, _ = plt.hist(pixels, bins=4, range=(0, 256))`：`bins` 是什么？
4. 为什么用 `cmap=plt.cm.binary` 时 MNIST 数字是白底黑字？

*答案：(1) 前者被拉伸，最亮的像素是白色；后者是一张暗图。(2) `(2, 3)`，左下角。(3) `[0, 64, 128, 192, 256]`。(4) `binary` 把 0 映射为白、1 映射为黑。*
