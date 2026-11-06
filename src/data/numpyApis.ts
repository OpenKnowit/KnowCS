// NumPy API 目录：右侧可视化面板的分类与示例代码。说明文案在 locales 的 numpy_api.api.<id>。
// match：示例运行后，第一个 api 名在列表里的调用会被选中展示（找不到就展示最后一次调用）。

export type ApiCat = 'create' | 'shape' | 'index' | 'math' | 'reduce' | 'stats' | 'sort' | 'linalg' | 'ml' | 'poly' | 'random' | 'files'

export interface ApiEntry {
  id: string
  cat: ApiCat
  label: string
  match: string[]
  code: string
}

export const API_CATS: ApiCat[] = ['create', 'shape', 'index', 'math', 'reduce', 'stats', 'sort', 'linalg', 'ml', 'poly', 'random', 'files']

const H = 'import numpy as np\n'

export const NUMPY_APIS: ApiEntry[] = [
  // ---- create
  { id: 'array', cat: 'create', label: 'np.array', match: ['np.array'], code: `${H}a = np.array([[1, 2, 3], [4, 5, 6]])\na.shape` },
  { id: 'arange', cat: 'create', label: 'np.arange', match: ['np.arange'], code: `${H}np.arange(2, 20, 3)   # start, stop (excluded), step` },
  { id: 'linspace', cat: 'create', label: 'np.linspace', match: ['np.linspace'], code: `${H}np.linspace(0, 1, 5)   # 5 points, both ends included` },
  { id: 'zeros', cat: 'create', label: 'np.zeros / ones', match: ['np.zeros', 'np.ones'], code: `${H}np.zeros((2, 3))\nnp.ones((2, 3), dtype=int)` },
  { id: 'full', cat: 'create', label: 'np.full', match: ['np.full'], code: `${H}np.full((2, 4), 7)` },
  { id: 'eye', cat: 'create', label: 'np.eye', match: ['np.eye'], code: `${H}np.eye(3)   # identity matrix` },
  { id: 'zeros_like', cat: 'create', label: 'np.zeros_like', match: ['np.zeros_like', 'np.ones_like'], code: `${H}a = np.arange(6).reshape(2, 3)\nnp.zeros_like(a)   # same shape and dtype as a` },
  { id: 'meshgrid', cat: 'create', label: 'np.meshgrid', match: ['np.meshgrid'], code: `${H}xs = np.array([0, 1, 2, 3])\nys = np.array([10, 20, 30])\nX, Y = np.meshgrid(xs, ys)   # every (x, y) pair on a grid\nY` },

  // ---- shape & data movement
  { id: 'reshape', cat: 'shape', label: 'reshape', match: ['ndarray.reshape', 'np.reshape'], code: `${H}a = np.arange(12)\na.reshape(3, 4)   # same 12 values, read row by row` },
  { id: 'reshape_neg', cat: 'shape', label: 'reshape(-1, …)', match: ['ndarray.reshape'], code: `${H}a = np.arange(12)\na.reshape(-1, 6)   # -1 = "work it out": 12 / 6 = 2` },
  { id: 'ravel', cat: 'shape', label: 'ravel / flatten', match: ['ndarray.ravel', 'ndarray.flatten', 'np.ravel'], code: `${H}a = np.arange(6).reshape(2, 3)\na.ravel()` },
  { id: 'transpose', cat: 'shape', label: '.T / transpose', match: ['ndarray.T', 'ndarray.transpose', 'np.transpose'], code: `${H}a = np.arange(6).reshape(2, 3)\na.T` },
  { id: 'swapaxes', cat: 'shape', label: 'np.swapaxes', match: ['np.swapaxes', 'ndarray.swapaxes'], code: `${H}a = np.arange(12).reshape(2, 3, 2)\nnp.swapaxes(a, 0, 2)` },
  { id: 'expand_dims', cat: 'shape', label: 'expand_dims / None', match: ['np.expand_dims', 'index'], code: `${H}x = np.array([1, 2, 3])\nnp.expand_dims(x, axis=1)   # same as x[:, None]` },
  { id: 'squeeze', cat: 'shape', label: 'np.squeeze', match: ['np.squeeze', 'ndarray.squeeze'], code: `${H}a = np.arange(3).reshape(1, 3, 1)\nnp.squeeze(a)   # drop every length-1 axis` },
  { id: 'concatenate', cat: 'shape', label: 'np.concatenate', match: ['np.concatenate'], code: `${H}a = np.array([[1, 2], [3, 4]])\nb = np.array([[5, 6]])\nnp.concatenate([a, b], axis=0)` },
  { id: 'stack', cat: 'shape', label: 'np.stack', match: ['np.stack'], code: `${H}a = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\nnp.stack([a, b], axis=1)   # new axis at position 1` },
  { id: 'vstack', cat: 'shape', label: 'vstack / hstack', match: ['np.vstack', 'np.hstack'], code: `${H}a = np.array([1, 2])\nb = np.array([3, 4])\nnp.vstack([a, b])\nnp.hstack([a, b])` },
  { id: 'tile', cat: 'shape', label: 'np.tile', match: ['np.tile'], code: `${H}a = np.array([[1, 2], [3, 4]])\nnp.tile(a, (2, 2))` },
  { id: 'repeat', cat: 'shape', label: 'np.repeat', match: ['np.repeat', 'ndarray.repeat'], code: `${H}a = np.array([[1, 2], [3, 4]])\nnp.repeat(a, 2, axis=1)` },
  { id: 'flip', cat: 'shape', label: 'np.flip', match: ['np.flip'], code: `${H}a = np.arange(6).reshape(2, 3)\nnp.flip(a, axis=1)` },
  { id: 'pad', cat: 'shape', label: 'np.pad', match: ['np.pad'], code: `${H}img = np.array([[1, 2], [3, 4]])\nnp.pad(img, 1)   # zero padding P = 1 before a 3×3 convolution` },

  // ---- indexing & selection
  { id: 'slice', cat: 'index', label: 'a[r0:r1, c0:c1]', match: ['index'], code: `${H}a = np.arange(16).reshape(4, 4)\na[1:3, ::2]` },
  { id: 'fancy', cat: 'index', label: 'a[[i, j]]', match: ['index'], code: `${H}a = np.arange(16).reshape(4, 4)\na[[0, 2, 3], [1, 3, 0]]   # points (0,1) (2,3) (3,0)` },
  { id: 'mask', cat: 'index', label: 'a[a > k]', match: ['index'], code: `${H}a = np.arange(12).reshape(3, 4)\nmask = a % 3 == 0\na[mask]` },
  { id: 'where', cat: 'index', label: 'np.where', match: ['np.where'], code: `${H}z = np.array([[-2, 3], [4, -1]])\nnp.where(z > 0, z, 0)   # ReLU without changing z` },
  { id: 'isin', cat: 'index', label: 'np.isin', match: ['np.isin'], code: `${H}labels = np.array([3, 7, 1, 7, 2])\nnp.isin(labels, [1, 7])   # is each label one of these?` },

  // ---- element-wise math
  { id: 'broadcast', cat: 'math', label: 'a + b (broadcast)', match: ['op:+'], code: `${H}a = np.arange(3)\nb = np.array([[0], [10]])\na + b   # (3,) + (2, 1) → (2, 3)` },
  { id: 'scalar_op', cat: 'math', label: 'a * 2, a ** 2', match: ['op:**', 'op:*'], code: `${H}a = np.array([1, 2, 3, 4])\na ** 2` },
  { id: 'compare', cat: 'math', label: 'a > b', match: ['op:>'], code: `${H}a = np.array([[1, 5], [7, 2]])\na > 3` },
  { id: 'sqrt', cat: 'math', label: 'sqrt / exp / log', match: ['np.sqrt', 'np.exp', 'np.log'], code: `${H}a = np.array([1, 4, 9, 16])\nnp.sqrt(a)` },
  { id: 'maximum', cat: 'math', label: 'np.maximum', match: ['np.maximum', 'np.minimum'], code: `${H}z = np.array([-2, -1, 0, 1, 2])\nnp.maximum(z, 0)   # ReLU` },
  { id: 'clip', cat: 'math', label: 'np.clip', match: ['np.clip', 'ndarray.clip'], code: `${H}a = np.array([-5, 0, 3, 8, 12])\nnp.clip(a, 0, 10)` },
  { id: 'round', cat: 'math', label: 'np.round', match: ['np.round', 'ndarray.round'], code: `${H}a = np.array([0.5, 1.5, 2.5, 2.567])\nnp.round(a)   # halves go to the even neighbour` },
  { id: 'abs', cat: 'math', label: 'np.abs / np.square', match: ['np.abs', 'np.square'], code: `${H}d = np.array([[1, -3], [-2, 4]])\nnp.abs(d)` },
  { id: 'diff', cat: 'math', label: 'np.diff', match: ['np.diff'], code: `${H}loss = np.array([0.9, 0.6, 0.45, 0.4, 0.41])\nnp.diff(loss)   # change per epoch: > 0 means the loss went up` },
  { id: 'isnan', cat: 'math', label: 'np.isnan', match: ['np.isnan'], code: `${H}x = np.array([1.5, np.nan, 3.0, np.nan])\nnp.isnan(x)   # x == np.nan is always False` },
  { id: 'tanh', cat: 'math', label: 'tanh / log2', match: ['np.tanh', 'np.log2'], code: `${H}z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])\nnp.tanh(z)   # squashes to (-1, 1)` },

  // ---- reductions
  { id: 'sum', cat: 'reduce', label: 'sum(axis=…)', match: ['np.sum', 'ndarray.sum'], code: `${H}a = np.arange(6).reshape(2, 3)\na.sum(axis=0)   # collapse the rows` },
  { id: 'sum_all', cat: 'reduce', label: 'sum()', match: ['np.sum', 'ndarray.sum'], code: `${H}a = np.arange(6).reshape(2, 3)\nnp.sum(a)` },
  { id: 'mean', cat: 'reduce', label: 'mean(axis=1)', match: ['np.mean', 'ndarray.mean'], code: `${H}scores = np.array([[70, 80, 90], [60, 65, 95]])\nscores.mean(axis=1)   # one mean per student` },
  { id: 'max', cat: 'reduce', label: 'max / min', match: ['np.max', 'ndarray.max', 'np.min', 'ndarray.min'], code: `${H}a = np.array([[3, 9, 2], [8, 1, 7]])\na.max(axis=1)` },
  { id: 'argmax', cat: 'reduce', label: 'argmax / argmin', match: ['np.argmax', 'ndarray.argmax', 'np.argmin', 'ndarray.argmin'], code: `${H}probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])\nnp.argmax(probs, axis=1)   # predicted class` },
  { id: 'std', cat: 'reduce', label: 'std / var', match: ['np.std', 'ndarray.std', 'np.var', 'ndarray.var'], code: `${H}x = np.array([[2, 4, 4], [4, 5, 7]])\nx.std(axis=0)   # population std (ddof=0)` },
  { id: 'prod', cat: 'reduce', label: 'np.prod', match: ['np.prod', 'ndarray.prod'], code: `${H}a = np.array([[1, 2, 3], [4, 5, 6]])\nnp.prod(a, axis=1)` },
  { id: 'cumsum', cat: 'reduce', label: 'np.cumsum', match: ['np.cumsum', 'ndarray.cumsum'], code: `${H}a = np.arange(1, 7).reshape(2, 3)\nnp.cumsum(a, axis=1)   # running total along each row` },
  { id: 'any', cat: 'reduce', label: 'any / all', match: ['np.any', 'ndarray.any', 'np.all', 'ndarray.all'], code: `${H}a = np.array([[0, 0, 1], [0, 0, 0]])\na.any(axis=1)` },
  { id: 'count_nonzero', cat: 'reduce', label: 'count_nonzero', match: ['np.count_nonzero'], code: `${H}pred = np.array([1, 0, 1, 1])\ntrue = np.array([1, 1, 1, 0])\nnp.count_nonzero(pred == true)   # correct predictions` },


  // ---- counting and statistics
  { id: 'bincount', cat: 'stats', label: 'np.bincount', match: ['np.bincount'], code: `${H}labels = np.array([0, 2, 1, 2, 2, 0])\nnp.bincount(labels)   # how many 0s, 1s, 2s` },
  { id: 'bincount_w', cat: 'stats', label: 'bincount(weights=…)', match: ['np.bincount'], code: `${H}votes = np.array([0, 1, 1, 0])          # labels of the 4 nearest neighbours\ndist = np.array([0.5, 2.0, 2.5, 1.0])\nnp.bincount(votes, weights=1 / dist)   # distance-weighted KNN vote` },
  { id: 'histogram', cat: 'stats', label: 'np.histogram', match: ['np.histogram'], code: `${H}pixels = np.array([12, 15, 40, 44, 47, 200, 210, 205, 198, 30])\ncounts, edges = np.histogram(pixels, bins=4, range=(0, 256))\ncounts   # Otsu's method starts from this` },
  { id: 'median', cat: 'stats', label: 'median / percentile', match: ['np.median', 'np.percentile'], code: `${H}x = np.array([[1, 9, 3], [4, 2, 100]])\nnp.median(x, axis=1)   # robust to the outlier 100` },
  { id: 'average', cat: 'stats', label: 'np.average(weights=…)', match: ['np.average'], code: `${H}marks = np.array([70, 85, 60])\nweights = np.array([0.5, 0.3, 0.2])\nnp.average(marks, weights=weights)` },
  // ---- sorting
  { id: 'sort', cat: 'sort', label: 'np.sort', match: ['np.sort'], code: `${H}a = np.array([[3, 1, 2], [9, 7, 8]])\nnp.sort(a, axis=1)` },
  { id: 'argsort', cat: 'sort', label: 'np.argsort', match: ['np.argsort', 'ndarray.argsort'], code: `${H}d = np.array([4.2, 0.5, 3.1, 1.0])\nnp.argsort(d)[:2]   # indices of the 2 nearest (KNN)` },
  { id: 'unique', cat: 'sort', label: 'np.unique', match: ['np.unique'], code: `${H}labels = np.array([2, 0, 2, 1, 0, 2])\nnp.unique(labels)` },

  // ---- linear algebra
  { id: 'matmul', cat: 'linalg', label: 'A @ B / np.dot', match: ['op:@', 'np.dot', 'np.matmul', 'ndarray.dot'], code: `${H}A = np.arange(6).reshape(2, 3)\nB = np.arange(6).reshape(3, 2)\nA @ B` },
  { id: 'matvec', cat: 'linalg', label: 'X @ w', match: ['op:@'], code: `${H}X = np.array([[1, 2], [3, 4], [5, 6]])\nw = np.array([0.5, -1])\nX @ w   # one prediction per row` },
  { id: 'outer', cat: 'linalg', label: 'np.outer', match: ['np.outer'], code: `${H}np.outer(np.array([1, 2, 3]), np.array([10, 20]))` },
  { id: 'trace', cat: 'linalg', label: 'np.trace', match: ['np.trace'], code: `${H}a = np.arange(9).reshape(3, 3)\nnp.trace(a)` },
  { id: 'norm', cat: 'linalg', label: 'np.linalg.norm', match: ['np.linalg.norm'], code: `${H}v = np.array([[3, 4], [6, 8]])\nnp.linalg.norm(v, axis=1)   # length of each row` },
  { id: 'inv', cat: 'linalg', label: 'np.linalg.inv', match: ['np.linalg.inv'], code: `${H}A = np.array([[4.0, 7.0], [2.0, 6.0]])\nAinv = np.linalg.inv(A)\nA @ Ainv   # ≈ identity` },
  { id: 'det', cat: 'linalg', label: 'np.linalg.det', match: ['np.linalg.det'], code: `${H}A = np.array([[3, 1], [2, 4]])\nnp.linalg.det(A)` },


  // ---- course recipes
  { id: 'kmeans_assign', cat: 'ml', label: 'K-Means: nearest centroid', match: ['np.argmin'], code: `${H}X = np.array([[1, 1], [2, 1], [8, 9], [9, 8]])\nC = np.array([[1, 2], [9, 9]])\nd = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))   # (4, 2)\nnp.argmin(d, axis=1)   # cluster of each point` },
  { id: 'knn_vote', cat: 'ml', label: 'KNN: majority vote', match: ['np.bincount'], code: `${H}X = np.array([[1, 1], [2, 2], [8, 8], [9, 9], [2, 1]])\ny = np.array([0, 0, 1, 1, 0])\nq = np.array([3, 3])\nd = np.linalg.norm(X - q, axis=1)\nnearest = np.argsort(d)[:3]\nnp.bincount(y[nearest]).argmax()` },
  { id: 'softmax', cat: 'ml', label: 'softmax (stable)', match: ['op:/'], code: `${H}z = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.0]])\ne = np.exp(z - z.max(axis=1, keepdims=True))   # subtract the max: no overflow\ne / e.sum(axis=1, keepdims=True)   # each row sums to 1` },
  { id: 'zscore', cat: 'ml', label: 'standardise (z-score)', match: ['op:/'], code: `${H}X = np.array([[150, 50], [160, 60], [170, 80], [180, 90]])\n(X - X.mean(axis=0)) / X.std(axis=0)   # each column: mean 0, std 1` },
  { id: 'onehot', cat: 'ml', label: 'one-hot: np.eye(k)[y]', match: ['index'], code: `${H}y = np.array([2, 0, 1, 2])\nnp.eye(3)[y]   # row y of the identity = one-hot label` },
  { id: 'confusion', cat: 'ml', label: 'confusion matrix', match: ['np.bincount'], code: `${H}true = np.array([0, 1, 2, 2, 1, 0, 2])\npred = np.array([0, 2, 2, 2, 1, 0, 1])\nk = 3\nnp.bincount(true * k + pred, minlength=k * k).reshape(k, k)   # rows = true, columns = predicted` },
  { id: 'perceptron', cat: 'ml', label: 'perceptron predict', match: ['np.where'], code: `${H}X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])\nw = np.array([1.0, 1.0])\nb = -1.5\nnp.where(X @ w + b >= 0, 1, 0)   # AND gate` },
  // ---- numpy.polynomial (the note's section 3)
  { id: 'poly_note', cat: 'poly', label: 'the note’s example', match: ['op:*'], code: `from numpy.polynomial import Polynomial\n\np = Polynomial([1, 2, 3])   # coefficients low → high: 1 + 2x + 3x²\nq = Polynomial([0, 1])      # q(x) = x (the note leaves q undefined)\np(2)          # evaluate → 17.0\np.roots()     # roots (may be complex)\np.deriv()     # derivative → 2 + 6x\np.integ()     # integral → x + x² + x³\np * q         # multiply with plain operators` },
  { id: 'poly_eval', cat: 'poly', label: 'Polynomial(…)(x)', match: ['Polynomial.__call__'], code: `${H}from numpy.polynomial import Polynomial\np = Polynomial([1, -3, 2])   # 1 - 3x + 2x²\np(np.array([-1, 0, 0.5, 1, 2]))` },
  { id: 'poly_deriv', cat: 'poly', label: '.deriv() / .integ()', match: ['Polynomial.deriv'], code: `from numpy.polynomial import Polynomial\np = Polynomial([0, -3, 0, 1])   # x³ - 3x\np.deriv()   # 3x² - 3: zero where p turns around` },
  { id: 'poly_roots', cat: 'poly', label: '.roots()', match: ['Polynomial.roots'], code: `from numpy.polynomial import Polynomial\np = Polynomial([-6, 11, -6, 1])   # (x - 1)(x - 2)(x - 3)\np.roots()` },
  { id: 'poly_fit', cat: 'poly', label: 'Polynomial.fit', match: ['Polynomial.fit'], code: `${H}from numpy.polynomial import Polynomial\nx = np.linspace(0, 4, 9)\nnoise = np.array([0.3, -0.2, 0.1, 0.25, -0.3, 0.05, -0.1, 0.2, -0.15])\ny = 0.5 * x ** 2 - 2 * x + 3 + noise\nfit = Polynomial.fit(x, y, deg=2)\nfit.convert()   # back to plain coefficients ≈ 3 - 2x + 0.5x²` },
  { id: 'polyfit', cat: 'poly', label: 'np.polyfit (legacy)', match: ['np.polyfit'], code: `${H}x = np.array([0, 1, 2, 3, 4])\ny = np.array([1.1, 2.9, 5.2, 7.1, 8.8])\nnp.polyfit(x, y, 1)   # legacy: HIGH → low, so [slope, intercept]` },

  // ---- random
  { id: 'rand', cat: 'random', label: 'np.random.rand', match: ['np.random.rand'], code: `${H}np.random.seed(0)\nnp.random.rand(2, 3)   # uniform in [0, 1)` },
  { id: 'randint', cat: 'random', label: 'np.random.randint', match: ['np.random.randint'], code: `${H}np.random.seed(1)\nnp.random.randint(0, 10, size=(2, 4))` },
  { id: 'randn', cat: 'random', label: 'np.random.randn', match: ['np.random.randn'], code: `${H}np.random.seed(2)\nnp.random.randn(5)   # standard normal` },
  {
    id: 'split', cat: 'shape', label: 'np.split', match: ['np.split'],
    code: `import numpy as np

a = np.arange(6)
print(np.split(a, 3))
print(np.split(a, [1, 4]))    # cut before index 1 and 4
print(np.array_split(np.arange(7), 3))   # sizes may differ`,
  },
  {
    id: 'moveaxis', cat: 'shape', label: 'np.moveaxis', match: ['np.moveaxis'],
    code: `import numpy as np

images = np.zeros((10, 32, 32, 3))
print(np.moveaxis(images, -1, 1).shape)     # channels last to first
print(np.transpose(images, (0, 3, 1, 2)).shape)   # the same, spelled out`,
  },
  {
    id: 'nonzero', cat: 'index', label: 'np.nonzero / argwhere', match: ['np.nonzero'],
    code: `import numpy as np

a = np.array([[0, 3], [5, 0]])
print(np.nonzero(a))       # (rows, columns)
print(np.argwhere(a > 0))  # one [row, column] per hit
print(a[np.nonzero(a)])`,
  },
  {
    id: 'take_along', cat: 'index', label: 'np.take_along_axis', match: ['np.take_along_axis'],
    code: `import numpy as np

scores = np.array([[3, 1, 2], [1, 3, 2]])
print(np.take(scores, [0, 2], axis=1))
order = np.argsort(scores, axis=1)
print(np.take_along_axis(scores, order, axis=1))`,
  },
  {
    id: 'percentile', cat: 'stats', label: 'np.percentile', match: ['np.percentile'],
    code: `import numpy as np

x = np.array([1, 2, 3, 4, 10])
print(np.percentile(x, 50), np.quantile(x, 0.25))
q1, q3 = np.percentile(x, [25, 75])
print(q3 - q1)    # interquartile range`,
  },
  {
    id: 'corrcoef', cat: 'stats', label: 'np.corrcoef / np.cov', match: ['np.corrcoef'],
    code: `import numpy as np

x = np.array([1., 2., 3., 4.])
y = np.array([2., 4., 5., 9.])
print(np.corrcoef(x, y))
print(np.cov(x, y))     # n − 1 in the denominator`,
  },
  {
    id: 'solve', cat: 'linalg', label: 'np.linalg.solve', match: ['np.linalg.solve'],
    code: `import numpy as np

A = np.array([[2., 1.], [1., 3.]])
b = np.array([3., 5.])
x = np.linalg.solve(A, b)
print(x, A @ x)
print(np.linalg.matrix_rank(A), np.linalg.cond(A))`,
  },
  {
    id: 'eigh', cat: 'linalg', label: 'np.linalg.eigh', match: ['np.linalg.eigh'],
    code: `import numpy as np

S = np.array([[2., 1.], [1., 2.]])
values, vectors = np.linalg.eigh(S)
print(values)
print(np.round(np.abs(vectors), 4))
print(np.allclose(S @ vectors, vectors * values))`,
  },
  {
    id: 'svd', cat: 'linalg', label: 'np.linalg.svd', match: ['np.linalg.svd'],
    code: `import numpy as np

A = np.array([[3., 0.], [4., 5.], [0., 1.]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(U.shape, s.shape, Vt.shape)
print(np.round(s, 4))
print(np.allclose(U @ np.diag(s) @ Vt, A))`,
  },
  {
    id: 'lstsq', cat: 'linalg', label: 'np.linalg.lstsq', match: ['np.linalg.lstsq'],
    code: `import numpy as np

X = np.array([[1., 1.], [1., 2.], [1., 3.]])
y = np.array([2., 3., 5.])
w, *_ = np.linalg.lstsq(X, y, rcond=None)
print(w)
print(X @ w)`,
  },
  {
    id: 'rng', cat: 'random', label: 'default_rng', match: [],
    code: `import numpy as np

rng = np.random.default_rng(0)
print(rng.random((2, 2)).shape, rng.integers(0, 10, 3).shape)
print(sorted(rng.choice(5, 3, replace=False).tolist()) != [])
print(sorted(rng.permutation(4).tolist()))`,
  },
  {
    id: 'save_load', cat: 'files', label: 'np.save / np.load', match: ['np.save'],
    code: `import numpy as np

a = np.arange(6).reshape(2, 3)
np.save('a.npy', a)
b = np.load('a.npy')
print(b, b.dtype, b.shape)`,
  },
  {
    id: 'savez', cat: 'files', label: 'np.savez', match: ['np.savez'],
    code: `import numpy as np

np.savez('data.npz', x=np.ones((2, 2)), y=np.array([0, 1]))
d = np.load('data.npz')
print(d.files)
print(d['x'], d['y'])`,
  },
  {
    id: 'savetxt', cat: 'files', label: 'np.savetxt / loadtxt', match: ['np.savetxt'],
    code: `import numpy as np

np.savetxt('t.csv', np.array([[1, 2], [3, 4]]), delimiter=',', fmt='%d')
print(open('t.csv').read())
print(np.loadtxt('t.csv', delimiter=','))`,
  },
  {
    id: 'append_delete', cat: 'shape', label: 'np.append / delete / insert', match: ['np.append'],
    code: `import numpy as np

a = np.array([1, 2, 3])
print(np.append(a, 4), np.append(a, [4, 5]), np.insert(a, 1, 9), np.delete(a, 0), np.delete(a, [0, 2]))
m = np.array([[1, 2], [3, 4]])
print(np.append(m, [[5, 6]], axis=0))
print(np.delete(m, 0, axis=1), np.insert(m, 1, 0, axis=0))
print(a)`,
  },
  {
    id: 'copy_view', cat: 'shape', label: 'copy / view / astype', match: [],
    code: `import numpy as np

a = np.arange(4)
v = a[1:3]
c = a[1:3].copy()
v[0] = 99
print(a, c, v.base is a, c.base is None, np.shares_memory(a, v))
b = a.astype(float)
print(b.dtype, a.dtype, np.array([1.9, -1.9]).astype(int), np.array([1, 0]).astype(bool))
print(a.ndim, a.size, a.shape, a.dtype, a.itemsize, a.nbytes)
print(np.array_equal(a, a.copy()), a.tolist(), a[0].item(), a.fill(0), a)`,
  },
  {
    id: 'stack_more', cat: 'shape', label: 'column_stack / atleast_2d / flipud', match: [],
    code: `import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])
print(np.column_stack([a, b]), np.dstack([a, b]).shape, np.hstack([a, b]), np.vstack([a, b]).shape, np.block([[a], [b]]).shape)
print(np.atleast_1d(1), np.atleast_2d(a).shape, np.atleast_3d(a).shape)
m = np.arange(4).reshape(2, 2)
print(np.fliplr(m), np.flipud(m), np.rot90(m), np.roll(m, 1), np.diag(m), np.diag([1, 2]), np.tril(m), np.triu(m))
print(np.broadcast_to(a, (2, 2)), np.broadcast_shapes((2, 1), (3,)), [x.shape for x in np.array_split(np.arange(5), 2)], [x.shape for x in np.hsplit(m, 2)], [x.shape for x in np.vsplit(m, 2)])`,
  },
  {
    id: 'trig', cat: 'math', label: 'floor / sin / arctan2 / degrees', match: [],
    code: `import numpy as np

x = np.array([-1.5, 0.5, 2.5])
print(np.floor(x), np.ceil(x), np.trunc(x), np.rint(x), np.round(x), np.fix(x), np.sign(x))
print(np.sin(0), np.cos(0), np.tan(0), np.arcsin(1), np.arccos(1), np.arctan(1), np.arctan2(1, 1), np.sinh(0), np.cosh(0), np.pi, np.e)
print(np.degrees(np.pi), np.radians(180), np.deg2rad(90), np.rad2deg(np.pi / 2), np.hypot(3, 4))
print(np.exp(1), np.expm1(0), np.log1p(0), np.log10(100), np.power(2, 3), np.cbrt(8), np.floor_divide(7, 2), np.mod(7, 2), np.remainder(-7, 2), np.fmod(-7, 2), np.divmod(7, 2))
print(np.add(1, 2), np.subtract(1, 2), np.multiply(2, 3), np.divide(1, 2), np.negative(1), np.reciprocal(2.), np.minimum(1, 2), np.fmax(1, np.nan), np.nan_to_num(np.nan), np.isfinite(1.), np.isinf(1.), np.inf, np.nan)`,
  },
  {
    id: 'linalg_more', cat: 'linalg', label: 'np.linalg.eig / cholesky / qr / pinv', match: ['np.linalg.eig'],
    code: `import numpy as np

A = np.array([[2., 1.], [1., 3.]])
v, w = np.linalg.eig(A)
print(np.round(np.sort(v), 4), w.shape, np.linalg.eigvals(A).shape)
print(np.round(np.linalg.cholesky(A), 4))
Q, R = np.linalg.qr(A)
print(np.allclose(Q @ R, A), np.round(np.linalg.pinv(A), 4), np.linalg.matrix_rank(A), np.linalg.matrix_power(A, 2))
print(np.round(np.linalg.cond(A), 4), np.linalg.slogdet(A), np.linalg.multi_dot([A, A, A]).shape)
print(np.dot([1, 2], [3, 4]), np.inner([1, 2], [3, 4]), np.vdot([1, 2], [3, 4]), np.cross([1, 0, 0], [0, 1, 0]), np.kron([1, 2], [1, 1]))
print(np.einsum('ij,ij->', A, A), np.einsum('ii', A), np.einsum('ij,j->i', A, [1, 1]), np.tensordot(A, A, axes=1).shape)
try:
    np.linalg.inv(np.zeros((2, 2)))
except np.linalg.LinAlgError as e:
    print('LinAlgError', e)`,
  },
  {
    id: 'stats_more', cat: 'stats', label: 'digitize / convolve / nanmean', match: [],
    code: `import numpy as np

x = np.array([1., 2., np.nan, 4.])
print(np.nanmean(x), np.nansum(x), np.nanmax(x), np.nanmin(x), np.nanstd(x), np.nanmedian(x), np.nanargmax(x), np.nanpercentile(x, 50))
print(np.mean([1, 2]), np.sum([1, 2]), np.min([1, 2]), np.max([1, 2]), np.amax([1, 2]), np.amin([1, 2]), np.ptp([1, 5]), np.std([1, 3]), np.var([1, 3]), np.cumprod([1, 2, 3]))
print(np.digitize([0.5, 1.5, 3.], [1, 2]), np.searchsorted([1, 3, 5], 4), np.convolve([1, 2, 3], [1, 1]), np.convolve([1, 2, 3], [1, 1], 'same'), np.correlate([1, 2], [1, 1]), np.interp(1.5, [1, 2], [10, 20]), np.trapezoid([1, 2, 3]), np.gradient([1, 2, 4]))
print(np.unique([3, 1, 3], return_counts=True), np.unique([3, 1, 3], return_index=True), np.intersect1d([1, 2], [2, 3]), np.union1d([1, 2], [2, 3]), np.setdiff1d([1, 2], [2]), np.flatnonzero([0, 1, 1]), np.partition([3, 1, 2], 1)[1], np.argpartition([3, 1, 2], 0)[0], np.lexsort(([1, 2], [2, 1])))`,
  },
  {
    id: 'random_more', cat: 'random', label: 'np.random.choice / shuffle / normal', match: ['np.random.choice'],
    code: `import numpy as np

np.random.seed(0)
print(np.random.random(2).shape, np.random.random_sample(2).shape, np.random.uniform(0, 1, 2).shape, np.random.normal(0, 1, (2, 2)).shape, np.random.standard_normal(2).shape)
print(np.random.choice(5, 2).shape, np.random.choice([7, 8], 3, p=[1., 0.]), np.random.choice(4, 4, replace=False).shape, sorted(np.random.permutation(3).tolist()))
a = np.arange(4)
np.random.shuffle(a)
print(sorted(a.tolist()))
print(np.random.binomial(1, 1., 2), np.random.poisson(1., 2).shape, np.random.exponential(1., 2).shape, np.random.beta(1, 1, 2).shape, np.random.gamma(1., 1., 2).shape, np.random.multinomial(4, [1., 0.]), np.random.laplace(0, 1, 2).shape, np.random.multivariate_normal([0, 0], np.eye(2), 3).shape, np.random.dirichlet([1, 1]).shape)
r = np.random.RandomState(0)
print(r.randint(0, 5, 2).shape, r.rand(2).shape)
g = np.random.default_rng(0)
print(g.normal(size=2).shape, g.choice(3, 2, replace=False).shape, g.permutation(3).shape, g.uniform(size=1).shape)`,
  },
  {
    id: 'index_more', cat: 'index', label: 'np.ix_ / c_ / r_ / argwhere', match: ['np.ix_'],
    code: `import numpy as np

a = np.arange(9).reshape(3, 3)
print(a[np.ix_([0, 2], [0, 1])], a[:, [0, 2]], a[[0, 2]], a[1, ...], a[..., 1], a[::2, ::-1], a[-1], a[a[:, 0] > 2])
print(np.c_[[1, 2], [3, 4]], np.r_[[1, 2], [3]], np.s_[1:2], np.mgrid[0:2, 0:2].shape, np.ogrid[0:2, 0:2][1].shape, np.indices((2, 2)).shape)
print(np.take(a, [0, 8]), np.put(b := a.copy(), 0, 9), b[0, 0], np.diag_indices(2), np.tril_indices(2), np.triu_indices(2), np.unravel_index(5, (3, 3)), np.ravel_multi_index((1, 2), (3, 3)))
print(np.select([a[0] > 1], [1], default=0), np.choose([0, 1], [[1, 2], [3, 4]]), np.extract(a[0] > 0, a[0]), np.compress([0, 1], [5, 6]), np.argwhere(a > 7), np.nonzero(a[0]))
print(np.apply_along_axis(np.sum, 1, a), np.vectorize(lambda v: v + 1)([1]), np.fromfunction(lambda i, j: i + j, (2, 2)), np.copyto(c := np.zeros(1), 5), c, np.fill_diagonal(d := np.zeros((2, 2)), 1), d)`,
  },
  {
    id: 'errstate', cat: 'math', label: 'np.errstate / set_printoptions / finfo', match: ['np.errstate'],
    code: `import numpy as np

with np.errstate(divide='ignore', invalid='ignore'):
    print(np.array([1., 0.]) / 0)
print(np.finfo(np.float64).eps, np.finfo(np.float32).max, np.iinfo(np.int32).max, np.iinfo(np.uint8).max)
with np.printoptions(precision=2, suppress=True):
    print(np.array([1.23456, 1e-9]))
np.set_printoptions(precision=3)
print(np.array([1.23456]), np.array2string(np.array([1, 2]), separator=','), np.array_str(np.array([1])), np.array_repr(np.array([1])))
np.set_printoptions(precision=8)
print(np.result_type(np.int64, np.float32), np.can_cast(np.int64, np.float64), np.issubdtype(np.int64, np.integer), np.dtype('float32').name)
print(np.isclose(0.1 + 0.2, 0.3), np.allclose([1.], [1.0000001]), np.array_equiv([1, 1], 1), np.binary_repr(5), np.base_repr(10, 16), np.isscalar(1), np.ndim([1]), np.shape([1]), np.size([1]))`,
  },
  {
    id: 'windows', cat: 'ml', label: 'one-liners: sigmoid, MSE, batches', match: [],
    code: `import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
y = np.array([1, 0, 1])
p = np.array([.9, .2, .6])
print(sigmoid(0.), np.mean((y - p) ** 2), -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), np.mean((p > .5) == y))
X = np.arange(12.).reshape(6, 2)
idx = np.random.permutation(len(X))
for i in range(0, len(X), 4):
    print(X[idx[i:i + 4]].shape)
print(np.linalg.norm(X - X[0], axis=1).argmin(), np.argsort(np.linalg.norm(X - X[0], axis=1))[:2], np.bincount([0, 1, 1]).argmax(), (X - X.mean(0)).mean(0), np.c_[np.ones(6), X].shape, np.hanning(3), np.hamming(3), np.blackman(3).shape, np.bartlett(3))`,
  },
]

/** Code blocks in the note → a runnable version in the panel (matched by their content). */
export const blockEntry = (code: string): string | null =>
  code.includes('Polynomial.fit') ? 'poly_fit' : code.includes('Polynomial(') ? 'poly_note' : null

/** Inline `code` in the NumPy note → which group of the panel to open. Namespaces outside the sandbox map to null. */
export const NOTE_LINKS: Record<string, ApiCat | null> = {
  numpy: 'create',
  ndarray: 'shape',
  dtype: 'create',
  pi: 'create',
  e: 'create',
  inf: 'create',
  'numpy.linalg': 'linalg',
  'np.linalg': 'linalg',
  'numpy.random': 'random',
  'np.random': 'random',
  'random.Generator': 'random',
  'numpy.fft': null,
  'np.fft': null,
  'numpy.polynomial': 'poly',
  'np.polynomial': 'poly',
  polynomial: 'poly',
  Polynomial: 'poly',
  'np.polyfit()': 'poly',
  'Chebyshev.fit()': null,
  Chebyshev: null,
  'np.poly1d': null,
  poly1d: null,
  'np.ma': null,
  'numpy.ma': null,
  'np.strings': null,
  'numpy.strings': null,
  'np.char': null,
  'numpy.char': null,
}
