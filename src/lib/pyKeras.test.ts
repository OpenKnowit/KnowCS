import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { KERAS } from './pyKeras'

const run = (src: string) => runPython(src, { libs: [KERAS], maxSize: 2_000_000, timeBudget: 20_000 })
const H = 'import numpy as np\nimport keras\nfrom keras import layers\nfrom keras.models import Sequential\nfrom keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input\n'

describe('model.summary() matches Keras 3.12', () => {
  it('the lecture-8 style CNN', () => {
    const r = run(`${H}m = Sequential([Input(shape=(28, 28, 1)), Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Flatten(), Dropout(0.5), Dense(10, activation='softmax')])\nm.summary()\nprint([l.name for l in m.layers])`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe(`Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 5408)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 5408)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 10)             │        54,090 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 54,410 (212.54 KB)
 Trainable params: 54,410 (212.54 KB)
 Non-trainable params: 0 (0.00 B)
['conv2d', 'max_pooling2d', 'flatten', 'dropout', 'dense']
`)
    const flow = r.events.find((e) => e.type === 'flow')
    expect(flow && flow.type === 'flow' && flow.rows.map((x) => x.note)).toEqual(['⌊(28 − 3) / 1⌋ + 1 = 26 · params (3·3·1 + 1)·32', '⌊(26 − 2) / 2⌋ + 1 = 13', '13 × 13 × 32 = 5408', null, '(5408 + 1) × 10 = 54,090'])
  })

  it('model.add with input_shape, and the Functional API', () => {
    const r = run(`${H}m2 = Sequential()\nm2.add(Dense(64, activation='relu', input_shape=(784,)))\nm2.add(Dense(10, activation='softmax'))\nprint(m2.count_params())
inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10, activation='softmax')(x)
f = keras.Model(inputs=inputs, outputs=outputs, name='mnist_mlp')
f.summary()
print(x.shape)`)
    expect(r.error).toBeNull()
    expect(r.stdout).toContain('50890\nModel: "mnist_mlp"')
    expect(r.stdout).toContain('│ input_layer (InputLayer)        │ (None, 784)            │             0 │')
    expect(r.stdout).toContain('│ dense_2 (Dense)                 │ (None, 64)             │        50,240 │')
    expect(r.stdout).toContain(' Total params: 50,890 (198.79 KB)')
    expect(r.stdout.endsWith('(None, 64)\n')).toBe(true)
  })

  it('Dense on an image without Flatten keeps the spatial axes (and warns)', () => {
    const r = run(`${H}m = Sequential([Input(shape=(28, 28, 1)), Dense(10)])\nm.summary()`)
    expect(r.stdout).toContain('│ dense (Dense)                   │ (None, 28, 28, 10)     │            20 │')
    expect(r.events.some((e) => e.type === 'note' && e.key === 'keras_dense_nd')).toBe(true)
  })
})

describe('training', () => {
  it('fit prints Keras logs, returns History, learns XOR', () => {
    const r = run(`${H}keras.utils.set_random_seed(0)
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 1, 1, 0])
m = Sequential([Input(shape=(2,)), Dense(8, activation='tanh'), Dense(1, activation='sigmoid')])
m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])
h = m.fit(X, y, epochs=200, batch_size=4, verbose=0)
print(sorted(h.history.keys()), len(h.history['loss']), h.epoch[:3])
loss, acc = m.evaluate(X, y, verbose=0)
print(acc, (m.predict(X, verbose=0) > 0.5).astype(int).ravel())
h2 = m.fit(X, y, epochs=2, batch_size=2)`)
    expect(r.error).toBeNull()
    expect(r.stdout).toMatch(/^\['accuracy', 'loss'\] 200 \[0, 1, 2\]\n1\.0 \[0 1 1 0\]\nEpoch 1\/2\n2\/2 ━━━━━━━━━━━━━━━━━━━━ - accuracy: 1\.0000 - loss: 0\.\d{4}\nEpoch 2\/2\n/)
    expect(r.events.filter((e) => e.type === 'history').length).toBe(2)
  })

  it('categorical_crossentropy with integer labels: the shape error; sparse works', () => {
    const src = `${H}X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
mc = Sequential([Input(shape=(2,)), Dense(3, activation='softmax')])
mc.compile(optimizer='adam', loss='LOSS', metrics=['accuracy'])
mc.fit(X, np.array([0, 1, 2, 1]), epochs=1, verbose=0)`
    expect(run(src.replace('LOSS', 'categorical_crossentropy')).error?.message).toBe('Arguments `target` and `output` must have the same shape. Received: target.shape=(None,), output.shape=(None, 3)')
    expect(run(src.replace('LOSS', 'sparse_categorical_crossentropy')).error).toBeNull()
    expect(run(`${H}from keras.utils import to_categorical\nprint(to_categorical([0, 2, 1], 3))`).stdout).toBe('[[1. 0. 0.]\n [0. 0. 1.]\n [0. 1. 0.]]\n')
  })

  it('validation data and EarlyStopping', () => {
    const r = run(`${H}keras.utils.set_random_seed(1)
X = np.random.rand(40, 3)
y = (X.sum(axis=1) > 1.5).astype(int)
m = Sequential([Input(shape=(3,)), Dense(16, activation='relu'), Dense(2, activation='softmax')])
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)
h = m.fit(X, y, epochs=50, validation_split=0.25, callbacks=[stop], verbose=0)
print(sorted(h.history.keys()), len(h.history['loss']) <= 50)`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe("['accuracy', 'loss', 'val_accuracy', 'val_loss'] True\n")
  })
})

describe('tensorflow', () => {
  const T = 'import tensorflow as tf\n'
  it('prints tensors as TensorFlow does', () => {
    const r = run(`${T}print(tf.constant([[1., 2.], [3., 4.]]))\nprint(tf.constant([1, 2, 3]))\nprint(repr(tf.constant([1, 2, 3])))\nprint(repr(tf.constant(0.1)))\nx = tf.constant([[1., 2.], [3., 4.]])\nprint(x.shape, x.dtype)`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe(`tf.Tensor(
[[1. 2.]
 [3. 4.]], shape=(2, 2), dtype=float32)
tf.Tensor([1 2 3], shape=(3,), dtype=int32)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
<tf.Tensor: shape=(), dtype=float32, numpy=0.1>
(2, 2) tf.float32
`)
  })
  it('no automatic dtype promotion', () => {
    expect(run(`${T}tf.constant(1.0) + tf.constant(1)`).error).toMatchObject({ type: 'InvalidArgumentError', message: 'cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:AddV2]' })
    expect(run(`${T}print(tf.reduce_mean(tf.constant([1, 2])))`).stdout).toBe('tf.Tensor(1, shape=(), dtype=int32)\n')
  })
  it('GradientTape and Variable.assign_sub: one gradient-descent step', () => {
    const r = run(`${T}w = tf.Variable(3.0)
with tf.GradientTape() as tape:
    loss = (w - 1.0) ** 2
g = tape.gradient(loss, w)
print(g)
w.assign_sub(0.1 * g)
print(w.numpy())`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('tf.Tensor(4.0, shape=(), dtype=float32)\n2.6\n')
  })
})
