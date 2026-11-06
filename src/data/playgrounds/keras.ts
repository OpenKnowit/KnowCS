// Keras 实验台：第 6、8 讲的模型搭建、summary、compile / fit / evaluate / predict；数据都在代码里现场生成（沙盒没有 MNIST）
import { KERAS } from '../../lib/pyKeras'
import { MATPLOTLIB } from '../../lib/pyPlot'
import type { PlayConfig, PlayEntry } from './types'

const H = 'import numpy as np\nimport keras\nfrom keras.models import Sequential\nfrom keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input\n\n'

const ENTRIES: PlayEntry[] = [
  // ---- build
  {
    id: 'cnn_summary', cat: 'build', label: 'lecture 8 CNN: summary()', focus: ['flow'],
    code: `${H}num_classes = 10
model = Sequential()                       # lecture 8, layer for layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()`,
  },
  {
    id: 'mlp_summary', cat: 'build', label: 'lecture 6 MLP', focus: ['flow'],
    code: `${H}from keras import regularizers, activations

model = Sequential()                       # lecture 6's model
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(Dense(units=128, activation=activations.relu, kernel_regularizer=regularizers.l2(0.002)))
model.add(Dense(units=10, activation='softmax'))
model.summary()
print((784 + 1) * 128, (128 + 1) * 128, (128 + 1) * 10)   # params of each Dense layer`,
  },
  {
    id: 'functional', cat: 'build', label: 'keras.Model(inputs, outputs)', focus: ['flow'],
    code: `${H}from keras import layers

inputs = keras.Input(shape=(784,))           # a placeholder: no batch dimension
x = layers.Dense(64, activation='relu')(inputs)
print(x.shape)                               # (None, 64): None is the batch
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_mlp')
model.summary()`,
  },
  {
    id: 'params', cat: 'build', label: 'count the parameters',
    code: `${H}model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(16, (5, 5), activation='relu'),     # (5*5*3 + 1) * 16
    MaxPooling2D((2, 2)),
    Conv2D(8, (3, 3), padding='same'),         # (3*3*16 + 1) * 8
    Flatten(),
    Dense(10, activation='softmax'),
])
for layer in model.layers:
    print(f"{layer.name:16} {layer.count_params():>6}")
print(model.count_params())`,
  },

  // ---- train
  {
    id: 'fit_xor', cat: 'train', label: 'compile → fit → evaluate', focus: ['history'],
    code: `${H}import matplotlib.pyplot as plt
keras.utils.set_random_seed(0)

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 1, 1, 0])                     # XOR
model = Sequential([Input(shape=(2,)), Dense(8, activation='tanh'), Dense(1, activation='sigmoid')])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05),
              loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=150, batch_size=4, verbose=0)
loss, acc = model.evaluate(X, y, verbose=0)
print(f"loss {loss:.4f}, accuracy {acc}")
print(model.predict(X, verbose=0).round(3))
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('binary cross-entropy')`,
  },
  {
    id: 'blobs', cat: 'train', label: 'softmax classifier (3 blobs)',
    code: `${H}import matplotlib.pyplot as plt
np.random.seed(3)
keras.utils.set_random_seed(3)

centres = np.array([[0, 0], [3, 3], [0, 4]])
X = np.vstack([c + np.random.randn(30, 2) * 0.7 for c in centres])
y = np.repeat(np.arange(3), 30)                    # labels 0, 1, 2

model = Sequential([Input(shape=(2,)), Dense(16, activation='relu'), Dense(3, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=60, batch_size=16, verbose=0)

probs = model.predict(X, verbose=0)                 # one probability per class
pred = np.argmax(probs, axis=1)                     # the predicted class
print('accuracy', (pred == y).mean())
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='viridis', s=30)
plt.title('predicted class')`,
  },
  {
    id: 'overfit', cat: 'train', label: 'validation_split + EarlyStopping', focus: ['history'], focusFirst: true,
    code: `${H}np.random.seed(0)
keras.utils.set_random_seed(0)
X = np.random.rand(60, 5)
y = (X[:, 0] + 0.3 * np.random.randn(60) > 0.5).astype(int)   # a noisy rule

model = Sequential([Input(shape=(5,)), Dense(64, activation='relu'), Dense(64, activation='relu'), Dense(2, activation='softmax')])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=60, batch_size=8, validation_split=0.3, verbose=0)
best = int(np.argmin(history.history['val_loss'])) + 1
print('lowest validation loss at epoch', best)

stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model2 = Sequential([Input(shape=(5,)), Dense(64, activation='relu'), Dense(64, activation='relu'), Dense(2, activation='softmax')])
model2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h2 = model2.fit(X, y, epochs=60, batch_size=8, validation_split=0.3, callbacks=[stop], verbose=0)
print('with EarlyStopping it ran', len(h2.history['loss']), 'epochs')`,
  },
  {
    id: 'tiny_cnn', cat: 'train', label: 'a tiny CNN on 8×8 images',
    code: `${H}import matplotlib.pyplot as plt
np.random.seed(1)
keras.utils.set_random_seed(1)

# 40 noisy 8 x 8 images: a vertical bar (label 0) or a horizontal bar (label 1)
X = np.random.rand(40, 8, 8) * 0.3
y = np.arange(40) % 2
for i in range(40):
    k = np.random.randint(1, 7)
    if y[i] == 0:
        X[i, :, k] = 1
    else:
        X[i, k, :] = 1
X = X.reshape(40, 8, 8, 1)                     # Keras wants a channel axis

model = Sequential([Input(shape=(8, 8, 1)), Conv2D(4, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Flatten(), Dense(2, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=30, batch_size=8, verbose=0)
pred = np.argmax(model.predict(X[:6], verbose=0), axis=1)

fig, axes = plt.subplots(1, 6, figsize=(9, 2))
for i, ax in enumerate(axes):
    ax.imshow(X[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'pred {pred[i]}')
    ax.axis('off')`,
  },
  {
    id: 'confusion', cat: 'train', label: 'predict → confusion_matrix',
    code: `${H}from tensorflow.math import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
np.random.seed(2)
keras.utils.set_random_seed(2)

X = np.vstack([np.random.randn(25, 2) + [0, 0], np.random.randn(25, 2) + [2, 2]])
y = np.repeat([0, 1], 25)
model = Sequential([Input(shape=(2,)), Dense(2, activation='softmax')])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=40, verbose=0)

prediction_results = np.argmax(model.predict(X, verbose=0), axis=1)
cm = confusion_matrix(y, prediction_results)      # as in lecture 6
print(cm)
f, ax = plt.subplots(figsize=(4, 4))
sn.heatmap(cm.numpy(), annot=True, fmt='d', square=True, ax=ax)`,
  },

  // ---- pitfalls
  {
    id: 'sparse_vs_cat', cat: 'pitfalls', label: 'categorical vs sparse', expectError: 'ValueError',
    code: `${H}from keras.utils import to_categorical
X = np.random.rand(6, 4)
y = np.array([0, 2, 1, 2, 0, 1])                 # integer labels
print(to_categorical(y, 3))                      # the one-hot version

model = Sequential([Input(shape=(4,)), Dense(3, activation='softmax')])
# integer labels need 'sparse_categorical_crossentropy';
# 'categorical_crossentropy' needs to_categorical(y) first
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1, verbose=0)`,
  },
  {
    id: 'no_flatten', cat: 'pitfalls', label: 'Dense without Flatten', focus: ['flow'],
    code: `${H}model = Sequential([
    Input(shape=(28, 28, 1)),
    Dense(10, activation='softmax'),     # acts on the LAST axis only
])
model.summary()      # (None, 28, 28, 10): ten "predictions" per pixel row
# fix: put Flatten() before the Dense layer`,
  },
  {
    id: 'no_channel', cat: 'pitfalls', label: 'images need a channel axis', expectError: 'ValueError',
    code: `${H}x_train = np.random.rand(10, 28, 28)       # grayscale images, no channel axis
print(x_train.shape)
# fix: x_train = x_train.reshape(-1, 28, 28, 1)
model = Sequential([Input(shape=x_train.shape[1:]), Conv2D(8, (3, 3), activation='relu')])`,
  },
  {
    id: 'same_stride', cat: 'build', label: "padding='same', strides=2", focus: ['flow'],
    code: `${H}model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(16, 3, padding='same'),             # same: 28 -> 28
    Conv2D(32, 3, strides=2, padding='same'),  # same + stride 2: ceil(28 / 2) = 14
    Conv2D(32, 3, strides=2),                  # valid: floor((14 - 3) / 2) + 1 = 6
    Flatten(),
    Dense(10, activation='softmax'),
])
model.summary()`,
  },
  {
    id: 'onehot', cat: 'train', label: "to_categorical",
    code: `${H}from keras.utils import to_categorical

y = np.array([0, 2, 1, 2])
Y = to_categorical(y, num_classes=3)
print(Y)            # one row per label: the target for categorical_crossentropy
print(Y.shape)`,
  },
  // ---- what the sandbox gained later; each one was run on the real library
  {
    id: 'residual', cat: 'functional', label: 'skip connection',
    code: `import numpy as np
import keras
from keras import layers

x = keras.Input((8,))
h = layers.Dense(8, activation='relu')(x)
h = layers.Dense(8)(h)
out = layers.Add()([x, h])
model = keras.Model(x, out, name='residual')
model.summary()`,
  },
  {
    id: 'two_inputs', cat: 'functional', label: 'several inputs',
    code: `import numpy as np
import keras
from keras import layers

a = keras.Input((4,), name='image')
b = keras.Input((2,), name='meta')
h = layers.Concatenate()([layers.Dense(3, activation='relu')(a), b])
model = keras.Model([a, b], layers.Dense(1)(h))
model.summary()
print(model.predict({'image': np.ones((2, 4)), 'meta': np.ones((2, 2))}, verbose=0).shape)`,
  },
  {
    id: 'features', cat: 'functional', label: 'a feature extractor',
    code: `import numpy as np
import keras
from keras import layers

model = keras.Sequential([layers.Input((8, 8, 1)), layers.Conv2D(4, 3, activation='relu', name='c1'), layers.Flatten(), layers.Dense(2)])
fe = keras.Model(model.inputs, model.get_layer('c1').output)
print(fe.predict(np.zeros((1, 8, 8, 1)), verbose=0).shape)
print([l.name for l in fe.layers])`,
  },
  {
    id: 'custom_layer', cat: 'custom', label: 'class MyLayer(Layer)',
    code: `import numpy as np
import keras
from keras import layers

class Scale(layers.Layer):
    def build(self, input_shape):
        self.k = self.add_weight(shape=(input_shape[-1],), initializer='ones', name='k')
    def call(self, x):
        return x * self.k

model = keras.Sequential([layers.Input((3,)), Scale(), layers.Dense(1)])
model.summary()
print(model.layers[0].k.shape, model.count_params())`,
  },
  {
    id: 'subclass', cat: 'custom', label: 'class Net(keras.Model)',
    code: `import numpy as np
import keras
from keras import layers

class Net(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = layers.Dense(4, activation='relu')
        self.out = layers.Dense(1)
    def call(self, x):
        return self.out(self.hidden(x))

net = Net()
print(net(np.ones((2, 3))).shape, [l.name for l in net.layers], net.count_params())
net.compile('adam', 'mse')
print(list(net.fit(np.ones((4, 3)), np.ones((4, 1)), verbose=0).history))`,
  },
  {
    id: 'early_stop', cat: 'callbacks', label: 'EarlyStopping',
    code: `import numpy as np
import keras
from keras import layers

X = np.ones((4, 1))
m = keras.Sequential([layers.Input((1,)), layers.Dense(1, kernel_initializer='ones', use_bias=False)])
m.compile(keras.optimizers.SGD(0.05), 'mse')
stop = keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=0.5, verbose=1)
h = m.fit(X, np.zeros((4, 1)), epochs=20, verbose=0, callbacks=[stop])
print('ran', len(h.history['loss']), 'of 20 epochs')`,
  },
  {
    id: 'lr_schedule', cat: 'callbacks', label: 'LearningRateScheduler',
    code: `import numpy as np
import keras
from keras import layers

X = np.ones((4, 1))
m = keras.Sequential([layers.Input((1,)), layers.Dense(1, kernel_initializer='ones', use_bias=False)])
m.compile(keras.optimizers.SGD(0.1), 'mse')
sched = keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * 0.5)
h = m.fit(X, np.zeros((4, 1)), epochs=3, verbose=0, callbacks=[sched])
print([round(x, 4) for x in h.history['learning_rate']])
print([round(x, 4) for x in h.history['loss']])`,
  },
  {
    id: 'my_callback', cat: 'callbacks', label: 'class MyCallback',
    code: `import numpy as np
import keras
from keras import layers

class Show(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('epoch', epoch, 'loss', round(logs['loss'], 4))
        if logs['loss'] < 0.7:
            self.model.stop_training = True

m = keras.Sequential([layers.Input((1,)), layers.Dense(1, kernel_initializer='ones', use_bias=False)])
m.compile(keras.optimizers.SGD(0.05), 'mse')
m.fit(np.ones((4, 1)), np.zeros((4, 1)), epochs=9, verbose=0, callbacks=[Show()])`,
  },
  {
    id: 'saving', cat: 'callbacks', label: 'model.save',
    code: `import numpy as np
import keras
from keras import layers

import os
m = keras.Sequential([layers.Input((2,)), layers.Dense(1, kernel_initializer='ones', name='d')], name='net')
m.compile('adam', 'mse')
m.save('net.keras')
print(os.path.exists('net.keras'))
back = keras.models.load_model('net.keras')
print(back.name, back.optimizer.name, back.get_weights()[0].ravel())
print(m.get_layer('d').kernel.path)`,
  },
  {
    id: 'rnn_summary', cat: 'sequences', label: 'LSTM / GRU',
    code: `import numpy as np
import keras
from keras import layers

model = keras.Sequential([layers.Input((10,)), layers.Embedding(100, 8), layers.Bidirectional(layers.LSTM(4, return_sequences=True)), layers.GRU(4), layers.Dense(1, activation='sigmoid')])
model.summary()`,
  },
  {
    id: 'text', cat: 'sequences', label: 'TextVectorization',
    code: `import numpy as np
import keras
from keras import layers

v = layers.TextVectorization(output_sequence_length=4)
v.adapt(['the cat sat', 'the dog'])
print(sorted(v.get_vocabulary()))
print(v.vocabulary_size(), v(['the cat ran']).shape)
print(keras.utils.pad_sequences([[1, 2, 3], [4]], maxlen=3))`,
  },
  {
    id: 'dataset_fit', cat: 'sequences', label: 'fit(tf.data)',
    code: `import numpy as np
import keras
from keras import layers

import tensorflow as tf
X = np.arange(8.).reshape(8, 1)
ds = tf.data.Dataset.from_tensor_slices((X, 2 * X)).shuffle(8, seed=0).batch(4)
m = keras.Sequential([layers.Input((1,)), layers.Dense(1)])
m.compile(keras.optimizers.SGD(0.01), 'mse')
h = m.fit(ds, epochs=2, verbose=0)
print(ds.element_spec, h.params)
print(m.predict(ds, verbose=0).shape)`,
  },
  // ---- the everyday APIs; each one was run on the real library
  {
    id: 'cnn_layers', cat: 'layers', label: 'Conv2D / MaxPool / BatchNorm',
    code: `import numpy as np
import keras
from keras import layers

m = keras.Sequential([
    keras.Input((28, 28, 1)),
    layers.Rescaling(1 / 255),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Conv2D(16, 3, activation='relu'),
    layers.AveragePooling2D(2),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),
])
m.summary()
for l in m.layers:
    print(l.name, l.output.shape)`,
  },
  {
    id: 'shape_layers', cat: 'layers', label: 'Reshape / Permute / UpSampling2D',
    code: `import numpy as np
import keras
from keras import layers

x = np.arange(24.).reshape(1, 2, 3, 4)
print(layers.Reshape((4, 6))(x).shape)
print(layers.Permute((3, 1, 2))(x).shape)
print(layers.Flatten()(x).shape)
print(layers.UpSampling2D(2)(x).shape)
print(layers.ZeroPadding2D(1)(x).shape)
print(layers.Cropping2D(((0, 1), (1, 1)))(x).shape)
print(layers.Resizing(4, 6)(x).shape)
print(layers.RepeatVector(3)(np.ones((1, 2))).shape)`,
  },
  {
    id: 'activation_layers', cat: 'layers', label: 'ReLU / LeakyReLU / Softmax',
    code: `import numpy as np
import keras
from keras import layers

x = np.array([[-2., 0., 2.]])
print(layers.ReLU()(x))
print(layers.LeakyReLU(negative_slope=0.1)(x))
print(layers.ELU()(x))
print(layers.Softmax()(x))
print(layers.Activation('tanh')(x))
print(layers.Lambda(lambda t: t * 2)(x))
print(keras.activations.relu(x), keras.activations.sigmoid(x), keras.activations.gelu(x))`,
  },
  {
    id: 'merge_layers', cat: 'layers', label: 'Add / Concatenate / Multiply',
    code: `import numpy as np
import keras
from keras import layers

a = np.array([[1., 2.]])
b = np.array([[3., 4.]])
print(layers.Add()([a, b]), layers.Subtract()([a, b]), layers.Multiply()([a, b]))
print(layers.Average()([a, b]), layers.Maximum()([a, b]), layers.Minimum()([a, b]))
print(layers.Concatenate()([a, b]), layers.Dot(axes=1)([a, b]))
print(layers.add([a, b]), layers.concatenate([a, b], axis=-1))`,
  },
  {
    id: 'rnn_layers', cat: 'layers', label: 'SimpleRNN / Embedding / Bidirectional',
    code: `import numpy as np
import keras
from keras import layers

m = keras.Sequential([
    keras.Input((5,), dtype='int32'),
    layers.Embedding(100, 8),
    layers.Bidirectional(layers.SimpleRNN(4, return_sequences=True)),
    layers.TimeDistributed(layers.Dense(2)),
    layers.Conv1D(3, 2),
    layers.MaxPooling1D(2),
    layers.GlobalMaxPooling1D(),
])
m.summary()
print(m(np.zeros((2, 5), 'int32')).shape)`,
  },
  {
    id: 'attention_norm', cat: 'layers', label: 'LayerNormalization / Normalization',
    code: `import numpy as np
import keras
from keras import layers

x = np.ones((1, 4, 8), 'float32')
print(layers.LayerNormalization()(x).shape)
print(layers.GaussianNoise(0.1)(x).shape, layers.GaussianDropout(0.1)(x).shape, layers.SpatialDropout1D(0.5)(x).shape)
norm = layers.Normalization(axis=-1)
norm.adapt(np.array([[1.], [2.], [3.]]))
print(norm(np.array([[2.]])))`,
  },
  {
    id: 'augment', cat: 'layers', label: 'RandomFlip / RandomRotation',
    code: `import numpy as np
import keras
from keras import layers

aug = keras.Sequential([layers.RandomFlip('horizontal'), layers.RandomRotation(0.1), layers.RandomZoom(0.1), layers.RandomCrop(4, 4)])
x = np.random.rand(2, 8, 8, 3)
print(aug(x, training=True).shape)
print(layers.RandomFlip('horizontal')(x, training=False).shape)
print(np.allclose(layers.RandomFlip('horizontal')(x, training=False), x))`,
  },
  {
    id: 'losses_fn', cat: 'evaluate', label: 'keras.losses',
    code: `import numpy as np
import keras
from keras import layers

y = np.array([[1., 0.], [0., 1.]])
p = np.array([[.9, .1], [.2, .8]])
print(keras.losses.MeanSquaredError()(y, p))
print(keras.losses.MeanAbsoluteError()(y, p))
print(keras.losses.CategoricalCrossentropy()(y, p))
print(keras.losses.BinaryCrossentropy()(y, p))
print(keras.losses.SparseCategoricalCrossentropy()(np.array([0, 1]), p))
print(keras.losses.Huber()(y, p), keras.losses.Hinge()(y, p), keras.losses.KLDivergence()(y, p), keras.losses.CosineSimilarity()(y, p))
print(keras.losses.get('mse').__name__)`,
  },
  {
    id: 'metrics_fn', cat: 'evaluate', label: 'keras.metrics',
    code: `import numpy as np
import keras
from keras import layers

m = keras.metrics.BinaryAccuracy()
m.update_state([1, 0, 1, 1], [0.9, 0.2, 0.4, 0.8])
print(m.result(), m.name)
m.reset_state()
print(m.result())
for cls in [keras.metrics.Precision, keras.metrics.Recall, keras.metrics.AUC]:
    x = cls()
    x.update_state([1, 0, 1, 1], [0.9, 0.2, 0.4, 0.8])
    print(x.name, np.round(float(x.result()), 3))
c = keras.metrics.SparseCategoricalAccuracy()
c.update_state([1, 0], [[.1, .9], [.3, .7]])
print(c.result())
t = keras.metrics.TopKCategoricalAccuracy(k=2)
t.update_state([[0, 0, 1]], [[.5, .3, .2]])
print(t.result())
print(keras.metrics.MeanSquaredError().name, keras.metrics.RootMeanSquaredError().name, keras.metrics.Mean().name)`,
  },
  {
    id: 'optimizers', cat: 'evaluate', label: 'keras.optimizers',
    code: `import numpy as np
import keras
from keras import layers

for cls in [keras.optimizers.SGD, keras.optimizers.Adam, keras.optimizers.AdamW, keras.optimizers.RMSprop, keras.optimizers.Adagrad, keras.optimizers.Adadelta, keras.optimizers.Adamax, keras.optimizers.Nadam]:
    o = cls(learning_rate=0.01)
    print(cls.__name__, round(float(o.learning_rate), 4))
print(keras.optimizers.SGD(0.1, momentum=0.9).momentum)
print(type(keras.optimizers.get('adam')).__name__)
m = keras.Sequential([keras.Input((1,)), layers.Dense(1)])
m.compile(keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
print(m.metrics_names, m.get_compile_config()['loss'])`,
  },
  {
    id: 'initializers', cat: 'evaluate', label: 'initializers / regularizers',
    code: `import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(0)
print(keras.initializers.Zeros()((2,)), keras.initializers.Ones()((2,)), keras.initializers.Constant(3.)((2,)))
for init in ['glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'random_normal', 'orthogonal']:
    print(init, keras.initializers.get(init)((3, 3)).shape)
d = layers.Dense(2, kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=keras.regularizers.L2(0.01))
print(d(np.ones((1, 2))).shape, len(d.losses))
w = np.array([1., 2.], 'float32')
print(keras.regularizers.L2(0.5)(w), keras.regularizers.L1(0.5)(w), keras.regularizers.L1L2(0.5, 0.5)(w))`,
  },
  {
    id: 'model_attrs', cat: 'evaluate', label: 'weights / get_config / to_json',
    code: `import numpy as np
import keras
from keras import layers

m = keras.Sequential([keras.Input((3,)), layers.Dense(2, name='a'), layers.Dropout(0.5), layers.BatchNormalization()])
print(len(m.layers), [l.name for l in m.layers], m.count_params())
print([w.shape for w in m.weights])
print(len(m.trainable_weights), len(m.non_trainable_weights), len(m.trainable_variables))
print(m.get_layer('a').units, m.get_layer(index=0).name, m.input_shape, m.output_shape)
m.layers[0].trainable = False
print(len(m.trainable_weights), m.layers[0].trainable)
w = m.get_layer('a').get_weights()
m.get_layer('a').set_weights([np.ones_like(w[0]), w[1]])
print(m.get_layer('a').kernel.numpy())
print(m.get_config()['layers'][1]['class_name'], keras.models.model_from_json(m.to_json()).count_params())
print(keras.models.clone_model(m).count_params())`,
  },
  {
    id: 'batch_api', cat: 'train', label: 'train_on_batch / predict_on_batch',
    code: `import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(0)
X = np.array([[0.], [1.]])
y = np.array([[0.], [2.]])
m = keras.Sequential([keras.Input((1,)), layers.Dense(1)])
m.compile('sgd', 'mse')
first = m.train_on_batch(X, y)
for _ in range(50):
    last = m.train_on_batch(X, y)
print(float(last) < float(first))
print(m.predict_on_batch(X).shape, type(m.test_on_batch(X, y)).__name__)
m.save_weights('w.weights.h5')
m.load_weights('w.weights.h5')
print(m.predict(X, verbose=0).shape, m.evaluate(X, y, verbose=0) >= 0)`,
  },
  {
    id: 'callbacks_more', cat: 'callbacks', label: 'ReduceLROnPlateau / ModelCheckpoint',
    code: `import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(0)
X = np.linspace(0, 1, 8).reshape(-1, 1)
y = 2 * X
m = keras.Sequential([keras.Input((1,)), layers.Dense(1)])
m.compile(keras.optimizers.SGD(0.1), 'mse')
log = []
cbs = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-3),
    keras.callbacks.ModelCheckpoint('best.keras', monitor='loss', save_best_only=True),
    keras.callbacks.TerminateOnNaN(),
    keras.callbacks.LambdaCallback(on_epoch_end=lambda e, logs: log.append(e)),
]
h = m.fit(X, y, epochs=3, verbose=0, callbacks=cbs)
print(log, sorted(h.history), h.params['epochs'], h.epoch)
print(keras.models.load_model('best.keras').count_params())`,
  },
  {
    id: 'ops', cat: 'custom', label: 'keras.ops',
    code: `import numpy as np
import keras
from keras import layers

from keras import ops
x = ops.convert_to_tensor([[1., 2.], [3., 4.]])
print(ops.shape(x), ops.sum(x), ops.mean(x, axis=0), ops.max(x), ops.min(x), ops.prod(x))
print(ops.reshape(x, (-1,)), ops.transpose(x), ops.expand_dims(x, 0).shape, ops.squeeze(ops.ones((1, 2))).shape)
print(ops.concatenate([x, x]).shape, ops.stack([x, x]).shape, ops.tile(x, 2).shape, ops.zeros_like(x), ops.eye(2), ops.linspace(0, 1, 3))
print(ops.matmul(x, x), ops.multiply(x, x), ops.subtract(x, x), ops.divide(x, x), ops.power(x, 2), ops.sqrt(x), ops.exp(ops.zeros(1)), ops.abs(-x))
print(ops.argmax(x), ops.argsort(x), ops.sort(-x), ops.where(x > 2, x, 0.), ops.clip(x, 2, 3), ops.one_hot(np.array([0, 1]), 2))
print(ops.cast(x, 'int32'), ops.equal(x, x), ops.greater(x, 2), ops.less(x, 2), ops.norm(x), ops.stop_gradient(x).shape)
print(ops.softmax(x), ops.relu(x - 2), ops.sigmoid(x))`,
  },
  {
    id: 'utils', cat: 'custom', label: 'keras.utils',
    code: `import numpy as np
import keras
from keras import layers

print(keras.utils.to_categorical([0, 2], 3))
print(keras.utils.normalize(np.array([[3., 4.]])))
keras.utils.set_random_seed(0)
keras.utils.clear_session()
a, b = keras.utils.split_dataset(np.arange(10), left_size=0.8)
print(len(list(a)), len(list(b)))
print(keras.utils.img_to_array(np.zeros((2, 2, 3))).shape)`,
  },
]

export const KERAS_PLAYGROUND: PlayConfig = {
  id: 'keras',
  libs: [KERAS, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 5000,
  debounce: 700,
  cats: ['build', 'train', 'pitfalls', 'functional', 'custom', 'callbacks', 'sequences', 'layers', 'evaluate'],
  entries: ENTRIES,
  prelude: H,
  callApis: [],
}
