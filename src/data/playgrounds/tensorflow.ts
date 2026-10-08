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
]

export const TENSORFLOW_PLAYGROUND: PlayConfig = {
  id: 'tensorflow',
  libs: [KERAS, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 4000,
  debounce: 600,
  cats: ['tensors', 'autograd', 'keras'],
  entries: ENTRIES,
  prelude: H,
  callApis: ['tf.', 'op:'],
}
