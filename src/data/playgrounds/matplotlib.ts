// matplotlib 实验台：课程里出现过的画法（激活函数、imshow 灰度与 vmin/vmax、子图网格、K-Means 散点、损失曲线、直方图与 Otsu 阈值 …）
import { MATPLOTLIB } from '../../lib/pyPlot'
import type { PlayConfig, PlayEntry } from './types'

const H = 'import numpy as np\nimport matplotlib.pyplot as plt\n\n'

const ENTRIES: PlayEntry[] = [
  // ---- basics
  {
    id: 'anatomy', cat: 'basics', label: 'fig, ax = plt.subplots()', focus: ['figure'],
    code: `${H}x = np.linspace(-5, 5, 41)
fig, ax = plt.subplots(figsize=(6, 4))      # a Figure holding one Axes
ax.plot(x, 1 / (1 + np.exp(-x)), label='sigmoid')
ax.plot(x, np.maximum(0, x) / 5, '--', label='ReLU / 5')
ax.set_title('Activation functions')
ax.set_xlabel('z')
ax.set_ylabel('f(z)')
ax.legend()
ax.grid(True)`,
  },
  {
    id: 'subplot', cat: 'basics', label: 'plt.subplot(2, 2, i)',
    code: `${H}x = np.arange(10)
plt.figure(figsize=(8, 5))
plt.subplot(2, 2, 1); plt.plot(x, x);       plt.title('subplot(2, 2, 1)')
plt.subplot(2, 2, 2); plt.plot(x, x ** 2);  plt.title('subplot(2, 2, 2)')
plt.subplot(2, 2, 3); plt.bar(x, x % 3);    plt.title('subplot(2, 2, 3)')
plt.subplot(2, 2, 4); plt.scatter(x, 10 - x); plt.title('subplot(2, 2, 4)')
plt.suptitle('rows, columns, index (counted from 1, row by row)')`,
  },
  {
    id: 'fmt', cat: 'basics', label: "plt.plot(x, y, 'ro--')",
    code: `${H}x = np.arange(6)
plt.plot(x, x, 'ro--', label="'ro--'  red, circles, dashed")
plt.plot(x, x + 2, 'bs', label="'bs'  blue squares, no line")
plt.plot(x, x + 4, 'g^-', label="'g^-'  green triangles, solid")
plt.plot(x, x + 6, color='C1', linestyle=':', linewidth=3, label='keywords')
plt.legend(loc='upper left')`,
  },

  // ---- images (lecture 7)
  {
    id: 'imshow_gray', cat: 'images', label: "imshow(img, cmap='gray')", focus: ['figure'],
    code: `${H}img = np.array([[0.0, 0.2, 0.4, 0.6],
                [0.2, 0.4, 0.6, 0.8],
                [0.4, 0.6, 0.8, 1.0],
                [0.6, 0.8, 1.0, 1.0]])
plt.imshow(img, cmap='gray', vmin=0, vmax=1)   # how lecture 7 shows every image
plt.title('cmap="gray", vmin=0, vmax=1')
plt.colorbar()`,
  },
  {
    id: 'vmin_vmax', cat: 'images', label: 'vmin / vmax pitfall',
    code: `${H}dark = np.array([[0.0, 0.1, 0.2],
                 [0.1, 0.2, 0.3],
                 [0.2, 0.3, 0.3]])          # a dark image: nothing above 0.3
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
axes[0].imshow(dark)                        # default colormap: viridis
axes[0].set_title('default')
axes[1].imshow(dark, cmap='gray')           # still stretched to its own range
axes[1].set_title("cmap='gray'")
axes[2].imshow(dark, cmap='gray', vmin=0, vmax=1)
axes[2].set_title('vmin=0, vmax=1')`,
  },
  {
    id: 'rgb', cat: 'images', label: 'RGB (H, W, 3)',
    code: `${H}img = np.zeros((4, 4, 3))
img[:2, :2, 0] = 1        # top-left: red channel on
img[:2, 2:, 1] = 1        # top-right: green
img[2:, :2, 2] = 1        # bottom-left: blue
img[2:, 2:] = 1           # bottom-right: all three -> white
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
axes[0].imshow(img)
axes[0].set_title('RGB  (4, 4, 3)')
for c, name in enumerate(['R', 'G', 'B']):
    axes[c + 1].imshow(img[:, :, c], cmap='gray', vmin=0, vmax=1)
    axes[c + 1].set_title(f'img[:, :, {c}]  ({name})')`,
  },
  {
    id: 'grid', cat: 'images', label: 'plt.subplot(n, n, i + 1)',
    code: `${H}import math

imgs = np.zeros((6, 7, 7))             # six tiny 7 x 7 "digits"
imgs[0, :, 3] = 1                      # a vertical stroke
imgs[1, 3, :] = 1                      # a horizontal stroke
imgs[2] = np.eye(7)                    # a diagonal
imgs[3] = np.flip(np.eye(7), axis=1)   # the other diagonal
imgs[4, 1:6, 1:6] = 1
imgs[4, 2:5, 2:5] = 0                  # a hollow square
imgs[5] = np.maximum(np.eye(7), np.flip(np.eye(7), axis=1))
labels = ['|', '-', 'diag', 'anti', 'box', 'X']

# lecture 6's MNIST grid, line for line
num_cells = math.ceil(math.sqrt(len(imgs)))
plt.figure(figsize=(6, 6))
for i in range(len(imgs)):
    plt.subplot(num_cells, num_cells, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imgs[i], cmap=plt.cm.binary)
    plt.xlabel(labels[i])`,
  },
  {
    id: 'axes_flat', cat: 'images', label: 'for ax in axes.flat',
    code: `${H}imgs = np.random.rand(6, 5, 5)     # six random 5 x 5 images

fig, axes = plt.subplots(2, 3, figsize=(6, 4.4))
print(axes.shape)                  # a 2 x 3 array of Axes
for i, ax in enumerate(axes.flat): # row by row
    ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'axes.flat[{i}]')
    ax.axis('off')`,
  },
  {
    id: 'sobel', cat: 'images', label: 'convolution → imshow',
    code: `${H}img = np.zeros((8, 8))
img[2:6, 2:6] = 1                                  # a white square
k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel x: vertical edges

out = np.zeros((6, 6))                             # valid: 8 - 3 + 1 = 6
for i in range(6):
    for j in range(6):
        out[i, j] = (img[i:i+3, j:j+3] * k).sum()

fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('input 8 x 8')
axes[1].imshow(out, cmap='gray')
axes[1].set_title('Sobel-x output 6 x 6')`,
  },
  {
    id: 'otsu', cat: 'images', label: 'plt.hist + Otsu',
    code: `${H}pixels = np.array([20, 25, 30, 28, 35, 40, 30, 180, 190, 200, 210, 185, 195, 205])

# Otsu: try every threshold, keep the one with the largest between-class variance
best_t, best_var = 0, -1.0
for t in range(1, 256):
    bg, fg = pixels[pixels < t], pixels[pixels >= t]
    if len(bg) == 0 or len(fg) == 0:
        continue
    w0, w1 = len(bg) / len(pixels), len(fg) / len(pixels)
    var = w0 * w1 * (bg.mean() - fg.mean()) ** 2
    if var > best_var:
        best_t, best_var = t, var

plt.hist(pixels, bins=16, range=(0, 256), color='gray')
plt.axvline(best_t, color='r', linestyle='--', label=f'Otsu threshold = {best_t}')
plt.xlabel('pixel intensity')
plt.ylabel('count')
plt.legend()`,
  },

  // ---- machine learning plots
  {
    id: 'kmeans', cat: 'ml', label: 'scatter(c=labels)', focus: ['figure'],
    code: `${H}X = np.array([[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]])
centroids = np.array([[1, 1], [5, 7]])

# one K-Means assignment step: every point joins its nearest centroid
d = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
labels = np.argmin(d, axis=1)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=80)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=120, label='centroids')
plt.title('K-Means: assignment step')
plt.legend()`,
  },
  {
    id: 'boundary', cat: 'ml', label: 'decision boundary',
    code: `${H}X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])              # logical AND
w, b = np.array([1.0, 1.0]), -1.5       # a perceptron that solves AND

xs = np.linspace(-0.5, 1.5, 9)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=120, edgecolors='k')
plt.plot(xs, -(w[0] * xs + b) / w[1], 'k--', label='w·x + b = 0')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.legend()
plt.title('Perceptron decision boundary (AND)')`,
  },
  {
    id: 'history', cat: 'ml', label: 'loss curves',
    code: `${H}# made-up numbers in the shape of a Keras history.history
history = {'loss':     [0.90, 0.60, 0.45, 0.36, 0.30, 0.26, 0.23, 0.21],
           'val_loss': [0.95, 0.68, 0.55, 0.50, 0.49, 0.51, 0.55, 0.60]}
epochs = range(1, len(history['loss']) + 1)
plt.plot(epochs, history['loss'], 'o-', label='training loss')
plt.plot(epochs, history['val_loss'], 's--', label='validation loss')
best = int(np.argmin(history['val_loss'])) + 1
plt.axvline(best, color='gray', linestyle=':', label=f'best epoch = {best}')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Overfitting: validation loss turns up')`,
  },
  {
    id: 'confusion', cat: 'ml', label: 'confusion matrix',
    code: `${H}cm = np.array([[50, 10],
               [5, 35]])          # rows = true class, columns = predicted
plt.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xticks([0, 1], ['pred 0', 'pred 1'])
plt.yticks([0, 1], ['true 0', 'true 1'])
plt.colorbar()
precision = cm[1, 1] / cm[:, 1].sum()
recall = cm[1, 1] / cm[1, :].sum()
plt.title(f'precision {precision:.2f}, recall {recall:.2f}')`,
  },
  {
    id: 'heatmap', cat: 'ml', label: 'sn.heatmap(cm, annot=True)',
    code: `${H}import seaborn as sn

cm = np.array([[48, 2, 0],
               [3, 41, 6],
               [0, 5, 45]])      # 3 classes: rows = true, columns = predicted
f, ax = plt.subplots(figsize=(5, 5))
sn.heatmap(cm, annot=True, fmt='d', square=True, ax=ax)   # as in lecture 6
ax.set_xlabel('predicted')
ax.set_ylabel('true')
accuracy = np.trace(cm) / cm.sum()
ax.set_title(f'accuracy = trace / total = {accuracy:.2f}')`,
  },
  {
    id: 'choose_k', cat: 'ml', label: 'plt.bar',
    code: `${H}ks = [1, 3, 5, 7, 9]
acc = [0.82, 0.88, 0.91, 0.90, 0.87]     # made-up cross-validation accuracies
plt.bar([str(k) for k in ks], acc, color='C2')
plt.ylim(0.7, 1)
plt.xlabel('K')
plt.ylabel('validation accuracy')
plt.title(f'Choosing K: best K = {ks[int(np.argmax(acc))]}')`,
  },
  {
    id: 'knn_regions', cat: 'ml', label: "contourf: KNN regions", focus: ['figure'], 
    code: `${H}X = np.array([[1, 2], [2, 3], [3, 1], [2, 1], [6, 5], [7, 7], [8, 6], [6, 7], [4, 4], [5, 3]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
xx, yy = np.meshgrid(np.linspace(0, 9, 46), np.linspace(0, 8, 41))
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)            # (1886, 2): every point of the plane
d = np.sqrt(((grid[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))   # (1886, 10) distances
nearest = np.argsort(d, axis=1)[:, :3]                       # k = 3
Z = (y[nearest].mean(axis=1) > 0.5).astype(int).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.35, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title('KNN (k = 3) decision regions')`,
  },
  {
    id: 'errorbar', cat: 'ml', label: "plt.errorbar (CV)", 
    code: `${H}ks = np.array([1, 3, 5, 7, 9])
acc = np.array([[0.80, 0.78, 0.83, 0.79, 0.81],      # 5-fold accuracy for each k
                [0.86, 0.84, 0.88, 0.85, 0.87],
                [0.88, 0.87, 0.90, 0.86, 0.89],
                [0.87, 0.85, 0.88, 0.84, 0.88],
                [0.84, 0.83, 0.86, 0.82, 0.85]])
plt.errorbar(ks, acc.mean(axis=1), yerr=acc.std(axis=1), fmt='o-', capsize=4)
plt.xlabel('k')
plt.ylabel('cross-validation accuracy')
plt.title('Pick k: mean ± std over the folds')`,
  },
  {
    id: 'boxplot', cat: 'ml', label: "plt.boxplot", 
    code: `${H}height = np.array([152, 158, 160, 163, 165, 168, 170, 172, 175, 181, 199])
weight = np.array([45, 50, 52, 55, 58, 60, 62, 65, 70, 74, 80])
plt.boxplot([height, weight], tick_labels=['height (cm)', 'weight (kg)'])
plt.title('Different scales (and an outlier): standardise before KNN')`,
  },
  {
    id: 'pie', cat: 'ml', label: "plt.pie", 
    code: `${H}labels = ['cat', 'dog', 'bird']
counts = np.array([620, 300, 80])
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Class balance: is accuracy a fair score here?')`,
  },
  {
    id: 'annotate', cat: 'basics', label: "plt.annotate", 
    code: `${H}val = np.array([0.92, 0.70, 0.55, 0.47, 0.44, 0.43, 0.45, 0.49, 0.54, 0.60])
epochs = np.arange(1, 11)
best = np.argmin(val)
plt.plot(epochs, val, 'o-', label='val_loss')
plt.annotate('stop here', xy=(epochs[best], val[best]), xytext=(epochs[best] + 1.5, val[best] + 0.25),
             arrowprops=dict(arrowstyle='->'))
plt.xlabel('epoch')
plt.legend()`,
  },
  {
    id: 'step', cat: 'basics', label: "plt.step(where='post')", 
    code: `${H}z = np.linspace(-3, 3, 13)
plt.step(z, (z >= 0).astype(int), where='post', label='step (perceptron)')
plt.plot(z, 1 / (1 + np.exp(-z)), label='sigmoid')
plt.legend()
plt.title('Hard threshold vs smooth activation')`,
  },
]

export const MATPLOTLIB_PLAYGROUND: PlayConfig = {
  id: 'matplotlib',
  libs: [MATPLOTLIB],
  maxSize: 65536,
  timeBudget: 2000,
  debounce: 350,
  cats: ['basics', 'images', 'ml'],
  entries: ENTRIES,
  prelude: H,
  callApis: [],
}
