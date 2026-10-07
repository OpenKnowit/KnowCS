// NumPy API 目录：右侧可视化面板的分类与示例代码。说明文案在 locales 的 numpy_api.api.<id>。
// match：示例运行后，第一个 api 名在列表里的调用会被选中展示（找不到就展示最后一次调用）。

export type ApiCat = 'create' | 'shape' | 'index' | 'math' | 'reduce' | 'sort' | 'linalg' | 'random'

export interface ApiEntry {
  id: string
  cat: ApiCat
  label: string
  match: string[]
  code: string
}

export const API_CATS: ApiCat[] = ['create', 'shape', 'index', 'math', 'reduce', 'sort', 'linalg', 'random']

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

  // ---- indexing & selection
  { id: 'slice', cat: 'index', label: 'a[r0:r1, c0:c1]', match: ['index'], code: `${H}a = np.arange(16).reshape(4, 4)\na[1:3, ::2]` },
  { id: 'fancy', cat: 'index', label: 'a[[i, j]]', match: ['index'], code: `${H}a = np.arange(16).reshape(4, 4)\na[[0, 2, 3], [1, 3, 0]]   # points (0,1) (2,3) (3,0)` },
  { id: 'mask', cat: 'index', label: 'a[a > k]', match: ['index'], code: `${H}a = np.arange(12).reshape(3, 4)\nmask = a % 3 == 0\na[mask]` },
  { id: 'where', cat: 'index', label: 'np.where', match: ['np.where'], code: `${H}z = np.array([[-2, 3], [4, -1]])\nnp.where(z > 0, z, 0)   # ReLU without changing z` },

  // ---- element-wise math
  { id: 'broadcast', cat: 'math', label: 'a + b (broadcast)', match: ['op:+'], code: `${H}a = np.arange(3)\nb = np.array([[0], [10]])\na + b   # (3,) + (2, 1) → (2, 3)` },
  { id: 'scalar_op', cat: 'math', label: 'a * 2, a ** 2', match: ['op:**', 'op:*'], code: `${H}a = np.array([1, 2, 3, 4])\na ** 2` },
  { id: 'compare', cat: 'math', label: 'a > b', match: ['op:>'], code: `${H}a = np.array([[1, 5], [7, 2]])\na > 3` },
  { id: 'sqrt', cat: 'math', label: 'sqrt / exp / log', match: ['np.sqrt', 'np.exp', 'np.log'], code: `${H}a = np.array([1, 4, 9, 16])\nnp.sqrt(a)` },
  { id: 'maximum', cat: 'math', label: 'np.maximum', match: ['np.maximum', 'np.minimum'], code: `${H}z = np.array([-2, -1, 0, 1, 2])\nnp.maximum(z, 0)   # ReLU` },
  { id: 'clip', cat: 'math', label: 'np.clip', match: ['np.clip', 'ndarray.clip'], code: `${H}a = np.array([-5, 0, 3, 8, 12])\nnp.clip(a, 0, 10)` },
  { id: 'round', cat: 'math', label: 'np.round', match: ['np.round', 'ndarray.round'], code: `${H}a = np.array([0.5, 1.5, 2.5, 2.567])\nnp.round(a)   # halves go to the even neighbour` },
  { id: 'abs', cat: 'math', label: 'np.abs / np.square', match: ['np.abs', 'np.square'], code: `${H}d = np.array([[1, -3], [-2, 4]])\nnp.abs(d)` },

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

  // ---- random
  { id: 'rand', cat: 'random', label: 'np.random.rand', match: ['np.random.rand'], code: `${H}np.random.seed(0)\nnp.random.rand(2, 3)   # uniform in [0, 1)` },
  { id: 'randint', cat: 'random', label: 'np.random.randint', match: ['np.random.randint'], code: `${H}np.random.seed(1)\nnp.random.randint(0, 10, size=(2, 4))` },
  { id: 'randn', cat: 'random', label: 'np.random.randn', match: ['np.random.randn'], code: `${H}np.random.seed(2)\nnp.random.randn(5)   # standard normal` },
]

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
  'numpy.polynomial': null,
  'np.polynomial': null,
  'np.ma': null,
  'numpy.ma': null,
  'np.strings': null,
  'numpy.strings': null,
  'np.char': null,
  'numpy.char': null,
}
