// TensorFlow 实验台：张量、严格的 dtype、Variable 与 GradientTape、tf.keras 与 tf.math.confusion_matrix
import { KERAS } from '../../lib/pyKeras'
import { MATPLOTLIB } from '../../lib/pyPlot'
import type { PlayConfig, PlayEntry } from './types'

const H = 'import tensorflow as tf\n\n'

const ENTRIES: PlayEntry[] = [
  // ---- tensors
  {
    id: 'constant', cat: 'tensors', label: 'tf.constant',
    code: `${H}x = tf.constant([[1., 2.],
                 [3., 4.]])
print(x)                    # value, shape and dtype
print(x.shape, x.dtype)
print(x.numpy())            # back to a NumPy array
print(tf.constant([1, 2, 3]).dtype)   # Python ints -> int32 (NumPy gives int64)`,
  },
  {
    id: 'dtypes', cat: 'tensors', label: 'no automatic casting', expectError: 'InvalidArgumentError',
    code: `${H}a = tf.constant([1.5, 2.5])      # float32
b = tf.constant([1, 2])          # int32
print(a + tf.cast(b, tf.float32))   # cast explicitly ...
print(a + b)                        # ... TensorFlow never casts for you`,
  },
  {
    id: 'ops', cat: 'tensors', label: 'matmul, reduce_*',
    code: `${H}x = tf.constant([[1., 2.], [3., 4.]])
print(tf.matmul(x, x))          # matrix product (also x @ x)
print(x * x)                    # element-wise
print(tf.reduce_sum(x, axis=0)) # sum down each column
print(tf.reduce_mean(x))
print(tf.reduce_mean(tf.constant([1, 2])))   # int32: the mean is truncated to 1!`,
  },

  // ---- autograd
  {
    id: 'variable', cat: 'autograd', label: 'tf.Variable',
    code: `${H}w = tf.Variable([1.0, 2.0])     # a trainable, mutable tensor
print(w)
w.assign([5.0, 6.0])
w.assign_sub([0.5, 0.5])         # w = w - 0.5, in place
print(w.numpy())`,
  },
  {
    id: 'tape', cat: 'autograd', label: 'tf.GradientTape',
    code: `${H}x = tf.Variable(3.0)
with tf.GradientTape() as tape:   # record the operations ...
    y = x ** 2 + 2 * x
dy_dx = tape.gradient(y, x)       # ... then differentiate: 2x + 2 = 8
print(dy_dx)`,
  },
  {
    id: 'linreg', cat: 'autograd', label: 'gradient descent with a tape',
    code: `${H}import matplotlib.pyplot as plt

X = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y = tf.constant([[3.0], [5.0], [7.0], [9.0]])     # y = 2x + 1
w = tf.Variable([[0.0]])
b = tf.Variable([0.0])
losses = []
for step in range(150):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((tf.matmul(X, w) + b - y) ** 2)
    dw, db = tape.gradient(loss, [w, b])
    w.assign_sub(0.05 * dw)
    b.assign_sub(0.05 * db)
    losses.append(float(loss))
print(w.numpy(), b.numpy())
plt.plot(losses)
plt.xlabel('step')
plt.ylabel('MSE')`,
  },

  // ---- keras inside tensorflow
  {
    id: 'tf_keras', cat: 'keras', label: 'tf.keras.Sequential', focus: ['flow'],
    code: `${H}model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.summary()`,
  },
  {
    id: 'confusion', cat: 'keras', label: 'tf.math.confusion_matrix',
    code: `${H}import matplotlib.pyplot as plt

labels      = [0, 1, 2, 2, 1, 0, 2, 1]
predictions = [0, 2, 2, 2, 1, 0, 1, 1]
cm = tf.math.confusion_matrix(labels, predictions)   # rows = true, columns = predicted
print(cm)
correct = sum(cm.numpy()[i, i] for i in range(3))
print('accuracy', correct / len(labels))
plt.imshow(cm.numpy(), cmap='Blues')
plt.xlabel('predicted')
plt.ylabel('true')
plt.colorbar()`,
  },
  // ---- what the sandbox gained later; each one was run on the real library
  {
    id: 'select', cat: 'tensors', label: 'tf.where / gather',
    code: `import tensorflow as tf

x = tf.constant([[1., 2.], [3., 4.]])
print(tf.where(x > 2))
print(tf.where(x > 2, x, 0.))
print(tf.boolean_mask(x, x > 2))
print(tf.gather(x, [1, 0]))
print(tf.gather_nd(x, [[0, 1]]))`,
  },
  {
    id: 'reduce', cat: 'tensors', label: 'reduce_* / argmax',
    code: `import tensorflow as tf

x = tf.constant([[1., 5.], [3., 2.]])
print(tf.reduce_sum(x), tf.reduce_mean(x, axis=0), tf.reduce_max(x, axis=1, keepdims=True))
print(tf.argmax(x, axis=1), tf.one_hot([0, 1], 2))
print(tf.nn.softmax(x), tf.math.top_k(x, 1).indices)`,
  },
  {
    id: 'dtype_strict', cat: 'tensors', label: 'dtype errors', expectError: 'InvalidArgumentError',
    code: `import tensorflow as tf

a = tf.constant([1.5, 2.5])
b = tf.constant([1, 2])
print(a.dtype, b.dtype, repr(a.dtype))
print(a + tf.cast(b, tf.float32))
print(tf.add(tf.constant(2.0), 1))    # a Python number follows the tensor
tf.add(1, tf.constant(2.0))           # but as the first argument of tf.add it does not`,
  },
  {
    id: 'second_order', cat: 'autograd', label: 'nested tapes',
    code: `import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = x ** 3
    g = inner.gradient(y, x)     # 3x² = 27
print(g, outer.gradient(g, x))   # 6x = 18`,
  },
  {
    id: 'persistent', cat: 'autograd', label: 'persistent / watch',
    code: `import tensorflow as tf

c = tf.constant(2.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(c)              # constants are not watched by default
    a = c * c
    b = a * c
print(t.gradient(a, c), t.gradient(b, c))
with tf.GradientTape() as t2:
    d = c * 2
print(t2.gradient(d, c))    # no watch: None`,
  },
  {
    id: 'tf_function', cat: 'autograd', label: '@tf.function',
    code: `import tensorflow as tf

@tf.function
def f(x):
    return x * 2 + 1

print(f(tf.constant(3.0)))
print(tf.function(lambda x: x + 1)(1))    # the result is a tensor`,
  },
  {
    id: 'dataset', cat: 'data', label: 'tf.data.Dataset',
    code: `import tensorflow as tf

ds = tf.data.Dataset.from_tensor_slices(([1., 2., 3., 4., 5.], [0, 1, 0, 1, 0]))
print(ds, ds.cardinality())
for x, y in ds.map(lambda a, b: (a * 2, b)).batch(2).take(2):
    print(x.numpy().tolist(), y.numpy().tolist())
print(ds.batch(2), ds.batch(2).element_spec)`,
  },
  {
    id: 'from_gen', cat: 'data', label: 'from_generator',
    code: `import tensorflow as tf

def gen():
    for i in range(4):
        yield float(i), i % 2

ds = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32)))
print(ds)
print([(float(a), int(b)) for a, b in ds])`,
  },
  // ---- the everyday APIs; each one was run on the real library
  {
    id: 'create', cat: 'tensors', label: 'tf.zeros / ones / range / linspace',
    code: `import tensorflow as tf

print(tf.zeros((2, 2)))
print(tf.ones_like(tf.constant([1, 2])))
print(tf.fill((1, 2), 7))
print(tf.eye(2))
print(tf.range(0, 6, 2))
print(tf.linspace(0.0, 1.0, 3))
print(tf.convert_to_tensor([1.5, 2.5]))
print(tf.random.set_seed(0), tf.random.normal((2, 2)).shape, tf.random.uniform((1, 3), 0, 1).shape, tf.random.truncated_normal((1, 2)).shape)`,
  },
  {
    id: 'shapes', cat: 'tensors', label: 'tf.reshape / expand_dims / concat',
    code: `import tensorflow as tf

x = tf.reshape(tf.range(6), (2, 3))
print(x.shape, x.ndim, tf.rank(x), tf.size(x), tf.shape(x))
print(tf.reshape(x, (-1,)), tf.transpose(x).shape)
print(tf.expand_dims(x, 0).shape, tf.expand_dims(x, -1).shape, tf.squeeze(tf.zeros((1, 2, 1))).shape)
print(tf.concat([x, x], axis=0).shape, tf.stack([x, x]).shape)
print(tf.unstack(x)[0], [p.shape for p in tf.split(x, 3, axis=1)])
print(tf.tile(tf.constant([1, 2]), [2]), tf.reverse(tf.constant([1, 2]), [0]))
print(tf.pad(tf.constant([[1]]), [[1, 1], [0, 1]]))`,
  },
  {
    id: 'maths', cat: 'maths', label: 'tf.add / sqrt / clip_by_value',
    code: `import tensorflow as tf

a = tf.constant([1., 4., 9.])
b = tf.constant([2., 2., 2.])
print(tf.add(a, b), tf.subtract(a, b), tf.multiply(a, b), tf.divide(a, b), a + b, a * b)
print(tf.sqrt(a), tf.square(a), tf.exp(tf.zeros(1)), tf.math.log(a), tf.pow(a, 2.), tf.abs(-a), tf.negative(a))
print(tf.round(tf.constant(2.5)), tf.floor(a / 2), tf.math.ceil(a / 2), tf.sign(-a))
print(tf.clip_by_value(a, 2., 5.), tf.maximum(a, b), tf.minimum(a, b))
print(tf.math.divide_no_nan(a, tf.zeros(3)), tf.math.is_nan(a), tf.math.count_nonzero(a), tf.add_n([a, b]))
print(tf.sin(tf.zeros(1)), tf.cos(tf.zeros(1)), tf.tanh(tf.zeros(1)), tf.sigmoid(tf.zeros(1)))`,
  },
  {
    id: 'compare', cat: 'maths', label: 'tf.equal / greater / logical_and',
    code: `import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([3, 2, 1])
print(tf.equal(a, b), tf.not_equal(a, b), tf.greater(a, b), tf.less(a, b), tf.greater_equal(a, b), tf.less_equal(a, b))
m = a > 1
print(tf.logical_and(m, a < 3), tf.logical_or(m, a < 1), tf.logical_not(m))
print(tf.reduce_any(m), tf.reduce_all(m), tf.boolean_mask(a, m))
print(tf.argmin(a), tf.sort(b), tf.argsort(b), tf.unique(tf.constant([1, 1, 2]))[0], tf.cumsum(a))
print(tf.reduce_prod(a), tf.reduce_min(a), tf.math.reduce_std(tf.cast(a, tf.float32)), tf.math.reduce_variance(tf.cast(a, tf.float32)))`,
  },
  {
    id: 'accuracy', cat: 'maths', label: 'argmax + cast: accuracy',
    code: `import tensorflow as tf

probs = tf.constant([[.7, .3], [.2, .8], [.4, .6]])
labels = tf.constant([0, 1, 0])
pred = tf.argmax(probs, axis=1)
print(pred, pred.dtype)
hits = tf.equal(tf.cast(pred, tf.int32), labels)
print(hits, tf.reduce_mean(tf.cast(hits, tf.float32)))
print(tf.one_hot(labels, 2))`,
  },
  {
    id: 'activations', cat: 'maths', label: 'tf.nn.softmax / relu / losses',
    code: `import tensorflow as tf

x = tf.constant([[1., 2., 3.]])
print(tf.nn.relu(x - 2), tf.nn.leaky_relu(x - 2), tf.nn.softmax(x), tf.nn.log_softmax(x), tf.nn.sigmoid(x), tf.nn.tanh(x))
y = tf.constant([2])
print(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x))
print(tf.nn.softmax_cross_entropy_with_logits(labels=[[0., 0., 1.]], logits=x))
print(tf.nn.sigmoid_cross_entropy_with_logits(labels=[1.], logits=[0.]))
print(tf.nn.l2_loss(x), tf.nn.l2_normalize(x, axis=1), tf.nn.moments(x, axes=[1]))`,
  },
  {
    id: 'conv_pool', cat: 'maths', label: 'tf.nn.conv2d / max_pool2d',
    code: `import tensorflow as tf

x = tf.reshape(tf.range(16, dtype=tf.float32), (1, 4, 4, 1))
k = tf.reshape(tf.constant([[1., 0.], [0., -1.]]), (2, 2, 1, 1))
print(tf.nn.conv2d(x, k, strides=1, padding='VALID').shape)
print(tf.squeeze(tf.nn.conv2d(x, k, strides=1, padding='VALID')))
print(tf.squeeze(tf.nn.max_pool2d(x, 2, 2, 'VALID')))
print(tf.squeeze(tf.nn.avg_pool2d(x, 2, 2, 'VALID')))
print(tf.nn.dropout(tf.ones((1, 2)), rate=0.0))`,
  },
  {
    id: 'linalg', cat: 'maths', label: 'tf.linalg',
    code: `import tensorflow as tf

A = tf.constant([[2., 1.], [1., 3.]])
print(tf.linalg.det(A), tf.linalg.trace(A))
print(tf.linalg.inv(A))
print(tf.linalg.solve(A, tf.constant([[3.], [5.]])))
print(tf.linalg.diag(tf.constant([1., 2.])), tf.linalg.diag_part(A), tf.linalg.eye(2), tf.linalg.matrix_transpose(A))
print(tf.norm(A), tf.linalg.norm(tf.constant([3., 4.])), tf.tensordot(A, A, 1))`,
  },
  {
    id: 'image', cat: 'image', label: 'tf.image',
    code: `import tensorflow as tf

img = tf.reshape(tf.range(12, dtype=tf.float32), (2, 2, 3))
print(tf.image.flip_left_right(img).shape, tf.image.flip_up_down(img)[0, 0])
print(tf.image.rot90(img).shape)
print(tf.image.resize(img, (4, 4)).shape, tf.image.resize(img, (4, 4), method='nearest')[0, 0])
print(tf.image.rgb_to_grayscale(img).shape, tf.image.grayscale_to_rgb(tf.ones((1, 1, 1))).shape)
u8 = tf.constant([[[255]]], dtype=tf.uint8)
print(tf.image.convert_image_dtype(u8, tf.float32))`,
  },
  {
    id: 'ds_more', cat: 'data', label: 'shuffle / repeat / filter / cache',
    code: `import tensorflow as tf

ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
print([int(x) for x in ds.filter(lambda x: x % 2 == 1)])
print([int(x) for x in ds.map(lambda x: x * 2).skip(1).take(2)])
print([b.numpy().tolist() for b in ds.batch(2)])
print([int(x) for x in ds.batch(2).unbatch().take(2)])
print([int(x) for x in ds.take(2).repeat(2)])
print(sorted(int(x) for x in ds.shuffle(5, seed=0)))
print(ds.cache().prefetch(tf.data.AUTOTUNE).cardinality())`,
  },
  {
    id: 'strings', cat: 'data', label: 'tf.strings',
    code: `import tensorflow as tf

s = tf.constant(['Hello ', 'World'])
print(tf.strings.lower(s), tf.strings.upper(s), tf.strings.strip(s), tf.strings.length(s))
print(tf.strings.join(['a', 'b'], separator='-'))
print(s.dtype, s.shape)`,
  },
  {
    id: 'config', cat: 'data', label: 'tf.config / executing_eagerly',
    code: `import tensorflow as tf

print(tf.executing_eagerly())
print(type(tf.config.list_physical_devices('CPU')).__name__, tf.config.list_physical_devices('GPU'))
print(tf.is_tensor(tf.constant(1)), tf.is_tensor(1))
print(tf.TensorShape([2, None]), tf.TensorShape([2, 3]).as_list())
print(tf.identity(tf.constant(1)), tf.stop_gradient(tf.constant(1.)))`,
  },
]

export const TENSORFLOW_PLAYGROUND: PlayConfig = {
  id: 'tensorflow',
  libs: [KERAS, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 4000,
  debounce: 600,
  cats: ['tensors', 'autograd', 'keras', 'data', 'maths', 'image'],
  entries: ENTRIES,
  prelude: H,
  callApis: ['tf.', 'op:'],
}
