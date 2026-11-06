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
]

export const TENSORFLOW_PLAYGROUND: PlayConfig = {
  id: 'tensorflow',
  libs: [KERAS, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 4000,
  debounce: 600,
  cats: ['tensors', 'autograd', 'keras', 'data'],
  entries: ENTRIES,
  prelude: H,
  callApis: ['tf.', 'op:'],
}
