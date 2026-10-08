# Matplotlib for COMP2211

> Matplotlib is how the course *looks at* things: the images in lecture 7, the K-Means clusters in lecture 4, the MNIST grids and confusion matrix in lecture 6, the training curves in lectures 6, 8 and 9. You are rarely asked to write plotting code in an exam, but you are asked what a picture shows, and the pictures come from a handful of calls. Every code block below has a **Try it** button that opens it in the playground.

## 1. The mental model: Figure → Axes → artists

| Object | What it is | How you get it |
|---|---|---|
| **Figure** | the whole canvas (size in inches, `figsize=(w, h)`) | `plt.figure()`, `plt.subplots()` |
| **Axes** | one plot inside the figure: its own x/y range, ticks, title, legend | `plt.subplot(r, c, i)`, `fig, ax = plt.subplots()` |
| **Artist** | what you draw: a line, dots, bars, an image, text | `ax.plot`, `ax.scatter`, `ax.bar`, `ax.imshow`, `ax.text` |

There are two ways to write the same thing:

| pyplot style (the lectures) | object style |
|---|---|
| `plt.plot(x, y)` | `ax.plot(x, y)` |
| `plt.title('t')` | `ax.set_title('t')` |
| `plt.xlabel('x')` | `ax.set_xlabel('x')` |
| `plt.xlim(0, 1)` | `ax.set_xlim(0, 1)` |

The `plt.*` functions act on the **current** Axes: the last one created or picked with `plt.subplot`. That is why lecture 7 writes `plt.figure()` before every `plt.imshow`. Without a new figure, the next image would be drawn on top of the old one.

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

## 2. The calls the course uses

| Call | Draws | Where in the course |
|---|---|---|
| `plt.plot(x, y, 'ro--', label=…)` | lines / markers | activation functions, loss curves |
| `plt.scatter(x, y, c=labels, cmap=…)` | coloured points | K-Means clusters (L4), KNN data (L3) |
| `plt.bar(names, heights)` | bars | accuracy for each K, class counts |
| `plt.hist(values, bins=…)` | histogram, returns `(n, bins, patches)` | pixel intensities, Otsu (L7) |
| `plt.imshow(X, cmap=…, vmin=…, vmax=…)` | an image | every image in L7, MNIST in L6/L8 |
| `plt.subplot(rows, cols, i)` | picks cell *i* (from 1) | the 5 × 5 MNIST grid (L6) |
| `plt.xticks([])`, `plt.axis('off')` | hides ticks | image grids |
| `plt.title / xlabel / ylabel / legend` | labels | everywhere |
| `plt.show()` | displays (and finishes) the figure | end of every cell |

Format strings: colour letter (`b g r c m y k w`) + marker (`o s ^ x + .`) + line style (`- -- : -.`). `'ro--'` is red circles joined by a dashed line; `'bs'` is blue squares with no line.

## 3. Images (lecture 7): `imshow`

**Shapes.** A grayscale image is a 2-D array `(H, W)`. A colour image is `(H, W, 3)`, one 2-D array per channel (R, G, B). `img[:, :, 0]` is the red channel.

**Coordinates.** Row `r` is the *y* position counted downwards from the top, column `c` is *x*. `img[r, c]` is the pixel at x = c, y = r. `imshow` puts the origin at the top-left, so the y-axis increases downwards.

**Value → colour.** For a 2-D array, `imshow` maps each value through a colormap:

```text
colour = cmap( (v − vmin) / (vmax − vmin) )      # 0 → first colour, 1 → last colour
```

- If you leave out `cmap`, the colormap is `'viridis'` (purple → yellow), **not** grey.
- If you leave out `vmin` / `vmax`, they become the image's own min and max. The darkest pixel turns black and the brightest white, *whatever their real values*. A dark image therefore looks normal. This is why lecture 7 always writes `cmap='gray', vmin=0, vmax=1` (or `vmax=255` for `uint8`).
- `'gray'` maps 0 to black; `'binary'` / `'Greys'` map 0 to white. That is why MNIST digits in lectures 6 and 8 appear black on white.

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

**Value ranges for colour images.** Floats must lie in `[0, 1]` and integers in `[0, 255]`. Anything outside is *clipped*, with a warning. After a convolution or contrast stretch, either clip the result yourself (`np.clip(out, 0, 255)`) or rescale it before showing it.

## 4. Grids of images

Lecture 6 shows 25 MNIST digits like this: one `subplot` per image, with ticks hidden and the label underneath.

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

`fig, axes = plt.subplots(2, 3)` gives a 2 × 3 **array** of Axes: `axes[0, 1]` is row 0, column 1, and `for ax in axes.flat:` visits them row by row. With one row or one column, the array is 1-D (`axes[i]`).

## 5. Plots for machine learning

**Clusters (lecture 4).** `c=labels` colours each point through a colormap. A second `scatter` marks the centroids.

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]])
labels = np.array([0, 0, 1, 1, 1, 1])
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=80)
plt.scatter([1.25, 4], [1.5, 5.25], c='black', marker='x', s=120, label='centroids')
plt.legend()
```

**Training curves (lectures 6, 8, 9).** Plot `history.history['loss']` and `['val_loss']` against the epoch. If training loss keeps falling while validation loss turns up, the model is **overfitting**. Early stopping keeps the epoch where validation loss is lowest.

**Confusion matrix (lectures 3, 6).** Lecture 6 uses `seaborn.heatmap(cm, annot=True, fmt='d')`. That is `imshow` plus a number in each cell. Rows are the true classes and columns the predictions.

```python
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

cm = np.array([[50, 10],
               [5, 35]])
f, ax = plt.subplots(figsize=(4, 4))
sn.heatmap(cm, annot=True, fmt='d', square=True, ax=ax)
```

**Decision boundaries (lecture 5).** A perceptron's boundary is the line w₁x₁ + w₂x₂ + b = 0, drawn as `plt.plot(xs, -(w1 * xs + b) / w2)`.

## 6. Pitfalls checklist

- **Forgetting `plt.figure()`** between images in one cell draws the second image on top of the first.
- **`imshow` without `vmin` / `vmax`** shows contrast-stretched images, and without `cmap` it shows viridis colours.
- **`img[x, y]` instead of `img[y, x]`**: arrays are indexed `[row, column]`, i.e. `[y, x]`.
- **Colour images outside [0, 1] / [0, 255]** are clipped.
- **OpenCV's `cv2.imread` returns BGR**, so red and blue swap when shown with `imshow`. Lecture 7 reads PNGs with `mpimg.imread`, which returns RGB floats in [0, 1].
- **Plotting a PyTorch tensor that requires grad** fails. Use `t.detach().numpy()` first.
- **`plt.subplot` indices start at 1** while Python indices start at 0, hence `plt.subplot(5, 5, i + 1)`.

## 7. Self-check

1. A 28 × 28 float image has values in [0, 0.3]. What does `plt.imshow(img, cmap='gray')` show, and what does `plt.imshow(img, cmap='gray', vmin=0, vmax=1)` show?
2. `fig, axes = plt.subplots(2, 3)`: what is `axes.shape`, and which cell is `axes[1, 0]`?
3. `n, bins, _ = plt.hist(pixels, bins=4, range=(0, 256))`: what are `bins`?
4. Why does an MNIST digit look black on white with `cmap=plt.cm.binary`?

*Answers: (1) the first is stretched, so the brightest pixel is white; the second is a dark image. (2) `(2, 3)`, bottom-left. (3) `[0, 64, 128, 192, 256]`. (4) `binary` maps 0 → white and 1 → black.*
