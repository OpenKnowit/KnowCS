# Keras Core API Study Notes

> Compiled from the official keras.io docs (Keras 3, v3.14)
> Outline: **API landscape → classification framework → 25 core APIs in detail → complete example → learning path**

---

## 1. API Landscape: How Many APIs Are in the Docs?

The core Keras 3 docs contain on the order of **600–800 APIs** (200+ doc pages with 1–10 symbols each).

| Area | Approx. count | Notes |
| --- | --- | --- |
| `keras.ops` (numpy-style ops + nn ops) | ~250 | The largest part, but **no need to learn it**: use it like NumPy |
| Layers (all kinds of network layers) | ~110 | Important, but you only need to master 15 |
| Metrics + Losses | ~70 | Remember 3–5 of each |
| Applications (pretrained models) | ~40 | ResNet / EfficientNet, etc. |
| Optimizers + learning-rate schedules | ~30 | Adam / AdamW are enough to start |
| Initializers / Regularizers / Constraints | ~30 | Look up when needed |
| Activations / Callbacks / Datasets / Random | 10–20 each | Remember 4 callbacks |
| Model / saving / mixed precision / distribution, etc. | ~50 | Know how to use them |

Ecosystem extensions (not counted in the core): KerasHub with several hundred and KerasTuner with ~50, for 1,000+ in total.

---

## 2. Classification Framework: 6 Groups Along the "Model Lifecycle"

Every Keras API is organised around the lifecycle of training a model:

```
data → build model → configure training → train → evaluate / predict → save & deploy
```

| Stage | API area | How well to know it |
| --- | --- | --- |
| ① Data preparation | `keras.datasets`, `data_loading`, preprocessing layers | Know how to use them |
| ② Building the model | Layers, `Model`/`Sequential`, Applications | **Key: master 15 layers** |
| ③ Configuring training | Losses, Metrics, Optimizers | 3–5 of each |
| ④ Controlling training | Callbacks, learning-rate schedules | Remember 4 |
| ⑤ Evaluation / inference | `evaluate` / `predict` | Know how to use them |
| ⑥ Saving & deployment | saving / export | Remember 2 |
| ⑦ Low-level ops | `keras.ops` | **No need to learn; look up when needed** |

> **Tip: key insight**
> The largest area, `keras.ops`, is precisely the one you least need to learn: it is just numpy-style functions (`ops.matmul`, `ops.reshape`…) that you look up while writing custom layers.

### The 80/20 rule: 25 core APIs cover 90% of cases

```python
# model skeleton (3)
keras.Sequential, keras.Model, keras.Input

# layers (15, by frequency)
Dense, Conv2D, MaxPooling2D, Flatten, Dropout,
BatchNormalization, Embedding, LSTM, GRU,
MultiHeadAttention, LayerNormalization,
GlobalAveragePooling2D, Concatenate, Rescaling, TextVectorization

# the training trio
optimizers.Adam / AdamW
losses.SparseCategoricalCrossentropy / BinaryCrossentropy / MeanSquaredError
metrics.Accuracy / AUC

# workflow (4 methods)
model.compile() / fit() / evaluate() / predict()

# callbacks (4)
EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
```

---

## 3. Model Skeleton (3)

### 1. `keras.Sequential`: a pipeline model

Use when layers are **chained in a single straight line** with no branches.

```python
import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),     # declaring the input explicitly is recommended
    layers.Conv2D(32, 3, activation="relu"),
    layers.Flatten(),
    layers.Dense(10, activation="softmax"),
])
model.summary()   # print the structure any time; make it a habit
```

> **⚠️ Pitfall**
> It still runs without `Input` (the model is built the first time it sees data), but `summary()` then complains the model isn't built. Always write `Input` explicitly.

### 2. `keras.Input`: a placeholder for the data entry point

```python
inputs = keras.Input(shape=(784,))   # shape excludes the batch dimension!
```

> **⚠️ Pitfall**
> `shape=(784,)` means each sample has 784 features; the actual data is `(batch, 784)`. Including the batch dimension is the most common beginner mistake.

### 3. `keras.Model`: the Functional API (the real essence)

Use for branches, multiple inputs, multiple outputs and skip connections. The idea: **call layers like functions, with tensors flowing between them**.

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation="relu")(inputs)   # layer(input) → output
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

A branching example (impossible with Sequential):

```python
inputs = keras.Input(shape=(128,))
branch_a = layers.Dense(32, activation="relu")(inputs)
branch_b = layers.Dense(32, activation="tanh")(inputs)
merged = layers.Concatenate()([branch_a, branch_b])
outputs = layers.Dense(1, activation="sigmoid")(merged)
model = keras.Model(inputs, outputs)
```

> **Tip: rule of thumb**
> Use Sequential to try ideas quickly and Functional for real work. Once you know Functional you rarely go back.

---

## 4. The 15 Core Layers

### Dense and convolution group (4)

#### `Dense(units, activation)`: fully connected layer

One line of maths: `output = activation(input @ W + b)`

```python
layers.Dense(64, activation="relu")    # hidden layer
layers.Dense(10, activation="softmax") # multi-class output layer
layers.Dense(1, activation="sigmoid")  # binary output layer
layers.Dense(1)                        # regression output layer (no activation!)
```

#### `Conv2D(filters, kernel_size)`: image feature extractor

```python
layers.Conv2D(32, 3, activation="relu", padding="same")
```

- `filters`: how many kinds of features to learn (edges, textures…); conventionally doubled layer by layer, 32 → 64 → 128
- `padding="valid"` (the default) shrinks the image; `"same"` keeps width and height
- The input must be 4-D: `(batch, height, width, channels)`

#### `MaxPooling2D(pool_size=2)`: downsampling

```python
layers.MaxPooling2D(2)   # halves width and height, taking the max of each 2×2 region
```

Purpose: shrink feature maps, reduce computation, add translation invariance. The classic CNN rhythm: repeat `Conv → Conv → Pool`.

#### `Flatten()`: flatten

```python
layers.Flatten()   # (batch, 7, 7, 64) → (batch, 3136)
```

The bridge from the convolutional world (4-D) to the fully connected world (2-D). No parameters, just a reshape.

### Regularisation group (2)

#### `Dropout(rate)`: randomly drop units to fight overfitting

```python
layers.Dropout(0.5)   # randomly drops 50% of the units during training
```

- Only active during training; turned off automatically in `predict` (Keras handles it)
- Usually placed between Dense layers; rate is typically 0.2–0.5
- **Train the model until it overfits, then add Dropout**. Don't reverse the order

#### `BatchNormalization()`: normalise the output distribution of each layer

```python
layers.Conv2D(64, 3)
layers.BatchNormalization()
layers.Activation("relu")   # classic order: Conv → BN → ReLU
```

Purpose: more stable training, faster convergence, larger learning rates allowed. Almost always used in deep CNNs.

### Sequence / NLP group (5)

#### `TextVectorization`: text → integer sequences

```python
vectorizer = layers.TextVectorization(max_tokens=20000, output_sequence_length=200)
vectorizer.adapt(train_texts)   # you must adapt first to learn the vocabulary!
# "I love keras" → [12, 845, 1932, 0, 0, ...]
```

#### `Embedding(input_dim, output_dim)`: integers → dense vectors

```python
layers.Embedding(input_dim=20000, output_dim=128)
# word ID 845 → a learnable 128-d vector; words with similar meanings end up close together
```

The standard first layer of an NLP model. `input_dim` = vocabulary size, `output_dim` = vector size (64–300 is common).

#### `LSTM(units)` / `GRU(units)`: recurrent networks

```python
layers.LSTM(64)                         # returns only the last step → for classification
layers.LSTM(64, return_sequences=True)  # returns every step → for stacking / sequence labelling
```

- GRU is a simplified LSTM: fewer parameters, faster, usually similar results. **Try GRU first by default**

> **⚠️ The biggest pitfall**
> When stacking two RNN layers, the first must have `return_sequences=True`; otherwise the second receives no sequence and you get a shape error.

#### `MultiHeadAttention(num_heads, key_dim)`: the heart of the Transformer

```python
attn = layers.MultiHeadAttention(num_heads=8, key_dim=64)
output = attn(query=x, value=x, key=x)   # self-attention: all three are the same
```

Lets every position in a sequence directly "see" all other positions. Unlike an LSTM, it doesn't process step by step but globally in parallel.

### Normalisation and utility group (4)

#### `LayerNormalization()`: standard in Transformers

```python
x = layers.LayerNormalization()(x + attn_output)   # residual structure
```

The difference from BatchNorm in one sentence: **BatchNorm normalises across samples (good for CNNs); LayerNorm normalises within a single sample (good for sequences / Transformers and unaffected by batch size)**.

#### `GlobalAveragePooling2D()`: how modern CNNs finish

```python
layers.GlobalAveragePooling2D()   # (batch, 7, 7, 512) → (batch, 512)
```

Averages each channel over the whole image. Far fewer parameters than `Flatten` and more resistant to overfitting; **almost always used in transfer learning**.

#### `Concatenate()`: join several branches

```python
merged = layers.Concatenate()([branch_a, branch_b])  # concatenates along the last axis
```

#### `Rescaling(scale)`: put preprocessing inside the model

```python
layers.Rescaling(1./255)   # pixels 0–255 → 0–1
```

Benefit: normalisation becomes part of the model, so you can't forget the preprocessing at deployment. Such "preprocessing layers" are the recommended style in Keras 3.

---

## 5. The Training Trio

### Optimizer: how to update the weights

```python
keras.optimizers.Adam(learning_rate=1e-3)    # the all-round default; start with it
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)  # standard for training Transformers
```

**The only parameter you need to tune is `learning_rate`**:
- Default `1e-3`
- Not learning → increase it; loss oscillates / explodes → decrease it (try `1e-4`)
- Use an even smaller one (`1e-5`) when fine-tuning a pretrained model

### Loss: how to measure "how wrong" (the easiest one to get wrong)

Choose by just two things: **task type + label format**.

| Task | Label format | Loss | Output layer |
| --- | --- | --- | --- |
| Multi-class | integer `3` | `SparseCategoricalCrossentropy` | `Dense(N, "softmax")` |
| Multi-class | one-hot `[0,0,0,1,0]` | `CategoricalCrossentropy` | `Dense(N, "softmax")` |
| Binary | `0` or `1` | `BinaryCrossentropy` | `Dense(1, "sigmoid")` |
| Regression | continuous `3.72` | `MeanSquaredError` | `Dense(1)` with no activation |

> Sparse just means "the labels are integers, not one-hot". MNIST labels are integers like `5`, so use the Sparse version. Typical symptoms of the wrong choice: a shape error, or a loss that never goes down.

#### Advanced pitfall: `from_logits`

```python
# Style A: softmax in the output layer (intuitive)
layers.Dense(10, activation="softmax")
loss = keras.losses.SparseCategoricalCrossentropy()

# Style B: no activation in the output layer; the loss handles it (more stable numerically, recommended)
layers.Dense(10)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

Both are correct, but **don't mix them**: softmax + `from_logits=True` is a classic bug. The model trains, but noticeably worse.

### Metric: numbers for humans (not used in training)

```python
metrics=["accuracy"]              # the basic one for classification
metrics=[keras.metrics.AUC()]     # more honest than accuracy when classes are imbalanced
```

> When accuracy alone isn't enough: if 99% of samples are negative, always guessing negative also scores 99%. Look at AUC instead.

---

## 6. The Four Workflow Methods

```python
# ① compile: assemble the trio (configuration only, no computation)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ② fit: train
history = model.fit(
    x_train, y_train,
    batch_size=32,            # samples per step, commonly 32/64/128
    epochs=20,                # full passes over the data
    validation_split=0.2,     # automatically hold out 20% as a validation set
    callbacks=[...],
)
# history.history is a dict: {"loss": [...], "val_loss": [...], ...}
# plotting the val_loss curve is the first tool for diagnosing training

# ③ evaluate: compute the final score on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)

# ④ predict: real use
probs = model.predict(x_new)        # softmax output → probability per class
preds = probs.argmax(axis=-1)       # take the most likely class
```

### Reading the fit output (a core skill)

| What you see | Diagnosis |
| --- | --- |
| `loss` falls and `val_loss` falls too | Healthy, keep going |
| `loss` falls but `val_loss` starts rising | **Overfitting**: stop at the turning point (EarlyStopping's job) |
| Neither falls | Wrong learning rate / loss, or a data problem |

> **⚠️ Pitfall**
> `validation_split` simply takes the **last** 20% of the data. If the data is sorted by class, the validation set will be all one class. Shuffle first.

---

## 7. Four Callbacks

Callbacks = hooks that run automatically during training, triggered at the end of each epoch. **The first three together are the standard setup for production training**:

```python
callbacks = [
    # ① stop automatically when validation stops improving, and roll back to the best weights
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,                  # tolerate 5 epochs without improvement
        restore_best_weights=True,   # ⚠️ always add this! otherwise you keep the worse weights
    ),
    # ② save the best model to disk automatically
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    # ③ lower the learning rate automatically when stuck (big steps first, then fine steps)
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,        # halve the learning rate
        patience=3,        # smaller than EarlyStopping's patience: lower lr before giving up
    ),
    # ④ visualise training
    keras.callbacks.TensorBoard(log_dir="./logs"),
    # terminal: tensorboard --logdir ./logs
]

model.fit(..., epochs=100, callbacks=callbacks)
# with ① in place you can safely set a large epochs; training stops by itself
```

---

## 8. Complete Example: MNIST (all 25 APIs together)

```python
import keras
from keras import layers

# data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[..., None]   # (60000,28,28) → (60000,28,28,1)
x_test = x_test[..., None]

# model (Functional API)
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10)(x)              # no activation → logits
model = keras.Model(inputs, outputs)

# the trio
model.compile(
    optimizer=keras.optimizers.AdamW(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# training
model.fit(
    x_train, y_train,
    batch_size=128, epochs=50, validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ],
)

# evaluation
print(model.evaluate(x_test, y_test))
```

---

## 9. Learning Path

### Three stages

1. **Stage 1: get the main line running (1–2 days)**: run the full MNIST workflow, then change parameters to experiment (see the exercise list below). Build muscle memory for the lifecycle.
2. **Stage 2: unlock flexibility (3–5 days)**
   - Sequential → **Functional API** (multiple inputs / outputs, branches)
   - Do one round of **transfer learning** with `keras.applications.ResNet50` (freeze + fine-tune)
   - Write a custom `Layer` (subclass `keras.Layer` and implement `call()`), which naturally brings in `keras.ops`
3. **Stage 3: go deeper as needed (long term)**: custom training loops (override `train_step`), custom Loss / Callback / Metric. The pattern is always the same: **subclass a base class + override 1–2 methods**.

### Hands-on exercise list

- [ ] Run the MNIST example from section 8
- [ ] Remove BatchNorm and watch how val_loss changes
- [ ] Set lr to `1e-1` and see what an exploding loss looks like
- [ ] Switch to the non-Sparse loss and see what error you get (**having seen the error is more useful than memorising the right answer**)
- [ ] Write a two-branch model with the Functional API (practise `Concatenate`)
- [ ] Walk through an NLP pipeline with `keras.datasets.imdb`: `TextVectorization → Embedding → GRU → Dense`
- [ ] Hand-build a Transformer block with `MultiHeadAttention + LayerNormalization`

### Three study tips

1. **Remember the "entry points", not the APIs**: only remember namespaces like `keras.layers.` and `keras.losses.`; IDE completion + [keras.io/api](https://keras.io/api/) are your external memory. Nobody can memorise 600 of them.
2. **The official examples are the best textbook**: [keras.io/examples](https://keras.io/examples/) has 100+ end-to-end examples (CV / NLP / generative), ~100 lines each. Pick a direction and study 3–5 closely.
3. **Read the source to remove the mystery**: the Keras source is unusually clean. The core of `Dense` is 10 lines (matmul + bias + activation). When an API feels mysterious, click `[source]` on its doc page.

> **One-sentence summary**: the framework splits into 6 groups along the lifecycle → master the 25 core APIs → connect everything with the Functional API + custom Layers → look up the other 575 when you need them.
