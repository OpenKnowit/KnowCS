# Keras 核心 API 学习笔记

> 基于 keras.io 官方文档（Keras 3，v3.14）整理
> 主线：**API 全景 → 分类框架 → 25 个核心 API 详解 → 完整示例 → 学习路径**

---

## 一、API 全景：官方文档有多少 API？

Keras 3 核心文档 API 数量级：**约 600–800 个**（文档页面 200+ 个，每页 1–10 个符号）。

| 分区 | 大约数量 | 说明 |
| --- | --- | --- |
| `keras.ops`（numpy 风格算子 + nn 算子） | ~250 | 最大头，但**不用学**，当 NumPy 用 |
| Layers（各类网络层） | ~110 | 重点，但只需精通 15 个 |
| Metrics + Losses | ~70 | 每类记 3–5 个 |
| Applications（预训练模型） | ~40 | ResNet / EfficientNet 等 |
| Optimizers + 学习率调度 | ~30 | 记 Adam / AdamW 即可起步 |
| Initializers / Regularizers / Constraints | ~30 | 用时查 |
| Activations / Callbacks / Datasets / Random | 各 10–20 | 回调记 4 个 |
| Model / 保存 / 混合精度 / 分布式等 | ~50 | 会用即可 |

生态扩展（不算在核心内）：KerasHub ~几百个、KerasTuner ~50 个，总量 1000+。

---

## 二、分类框架：按"模型生命周期"分 6 大类

所有 Keras API 都围绕训练一个模型的生命周期组织：

```
数据 → 搭模型 → 配置训练 → 训练 → 评估/推理 → 保存部署
```

| 阶段 | API 分区 | 掌握程度 |
| --- | --- | --- |
| ① 数据准备 | `keras.datasets`、`data_loading`、预处理层 | 会用即可 |
| ② 搭模型 | Layers、`Model`/`Sequential`、Applications | **重点：精通 15 个层** |
| ③ 配置训练 | Losses、Metrics、Optimizers | 每类 3–5 个 |
| ④ 训练控制 | Callbacks、学习率调度 | 记 4 个 |
| ⑤ 评估/推理 | `evaluate` / `predict` | 会用即可 |
| ⑥ 保存部署 | saving / export | 记 2 个 |
| ⑦ 底层算子 | `keras.ops` | **不用学，用时查** |

> [!tip] 关键洞察
> 最大的一块 `keras.ops` 恰恰最不用学——它就是 numpy 风格函数（`ops.matmul`、`ops.reshape`…），写自定义层时现查现用。

### 二八法则：25 个核心 API 覆盖 90% 场景

```python
# 模型骨架（3个）
keras.Sequential, keras.Model, keras.Input

# 层（15个，按出现频率排）
Dense, Conv2D, MaxPooling2D, Flatten, Dropout,
BatchNormalization, Embedding, LSTM, GRU,
MultiHeadAttention, LayerNormalization,
GlobalAveragePooling2D, Concatenate, Rescaling, TextVectorization

# 训练三件套
optimizers.Adam / AdamW
losses.SparseCategoricalCrossentropy / BinaryCrossentropy / MeanSquaredError
metrics.Accuracy / AUC

# 流程（4个方法）
model.compile() / fit() / evaluate() / predict()

# 回调（4个）
EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
```

---

## 三、模型骨架（3个）

### 1. `keras.Sequential` — 流水线模型

适用：层与层**一条线串到底**，没有分支。

```python
import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),     # 推荐显式声明输入
    layers.Conv2D(32, 3, activation="relu"),
    layers.Flatten(),
    layers.Dense(10, activation="softmax"),
])
model.summary()   # 随时打印结构，养成习惯
```

> [!warning] 坑
> 不写 `Input` 也能跑（首次喂数据时才 build），但 `summary()` 会报模型未构建。始终显式写 `Input`。

### 2. `keras.Input` — 数据入口的占位符

```python
inputs = keras.Input(shape=(784,))   # shape 不含 batch 维！
```

> [!warning] 坑
> `shape=(784,)` 表示每个样本 784 维，实际数据是 `(batch, 784)`。新手最常见错误就是把 batch 维写进去。

### 3. `keras.Model` — Functional API（真正的精髓）

适用：分支、多输入、多输出、跳跃连接。思想：**把层当函数调用，张量在层之间流动**。

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation="relu")(inputs)   # 层(输入) → 输出
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

分支示例（Sequential 做不到）：

```python
inputs = keras.Input(shape=(128,))
branch_a = layers.Dense(32, activation="relu")(inputs)
branch_b = layers.Dense(32, activation="tanh")(inputs)
merged = layers.Concatenate()([branch_a, branch_b])
outputs = layers.Dense(1, activation="sigmoid")(merged)
model = keras.Model(inputs, outputs)
```

> [!tip] 心法
> Sequential 用来快速验证想法，Functional 用来干正事。学会 Functional 后基本不会回头。

---

## 四、15 个核心层

### 全连接与卷积组（4个）

#### `Dense(units, activation)` — 全连接层

本质一行数学：`output = activation(input @ W + b)`

```python
layers.Dense(64, activation="relu")    # 隐藏层
layers.Dense(10, activation="softmax") # 多分类输出层
layers.Dense(1, activation="sigmoid")  # 二分类输出层
layers.Dense(1)                        # 回归输出层（不要激活！）
```

#### `Conv2D(filters, kernel_size)` — 图像特征提取器

```python
layers.Conv2D(32, 3, activation="relu", padding="same")
```

- `filters`：学多少种特征（边缘、纹理…），惯例逐层翻倍 32 → 64 → 128
- `padding="valid"`（默认）会缩小图像；`"same"` 保持宽高不变
- 输入必须 4D：`(batch, height, width, channels)`

#### `MaxPooling2D(pool_size=2)` — 降采样

```python
layers.MaxPooling2D(2)   # 宽高各减半，取每 2×2 区域最大值
```

作用：缩小特征图、降低计算量、增加平移不变性。经典 CNN 节奏：`Conv → Conv → Pool` 重复。

#### `Flatten()` — 拍平

```python
layers.Flatten()   # (batch, 7, 7, 64) → (batch, 3136)
```

卷积世界（4D）到全连接世界（2D）的桥梁。无参数，纯 reshape。

### 正则化组（2个）

#### `Dropout(rate)` — 随机失活，防过拟合

```python
layers.Dropout(0.5)   # 训练时随机丢弃 50% 神经元
```

- 只在训练时生效，`predict` 时自动关闭（Keras 自动处理）
- 常放在 Dense 层之间；rate 通常 0.2–0.5
- **先把模型练到过拟合，再加 Dropout**，顺序别反

#### `BatchNormalization()` — 标准化每层输出分布

```python
layers.Conv2D(64, 3)
layers.BatchNormalization()
layers.Activation("relu")   # 经典顺序：Conv → BN → ReLU
```

作用：训练更稳、收敛更快、允许更大学习率。深层 CNN 几乎必用。

### 序列 / NLP 组（5个）

#### `TextVectorization` — 文本 → 整数序列

```python
vectorizer = layers.TextVectorization(max_tokens=20000, output_sequence_length=200)
vectorizer.adapt(train_texts)   # 必须先 adapt 学词表！
# "I love keras" → [12, 845, 1932, 0, 0, ...]
```

#### `Embedding(input_dim, output_dim)` — 整数 → 稠密向量

```python
layers.Embedding(input_dim=20000, output_dim=128)
# 词 ID 845 → 一个 128 维可学习向量，意思相近的词向量会靠近
```

NLP 模型标准第一层。`input_dim` = 词表大小，`output_dim` = 向量维度（64–300 常见）。

#### `LSTM(units)` / `GRU(units)` — 循环网络

```python
layers.LSTM(64)                         # 只返回最后一步 → 用于分类
layers.LSTM(64, return_sequences=True)  # 返回每一步 → 用于堆叠/序列标注
```

- GRU 是 LSTM 简化版：参数少、更快，效果通常接近。**默认先试 GRU**

> [!warning] 最大的坑
> 堆叠两层 RNN 时，第一层必须 `return_sequences=True`，否则第二层收不到序列，报 shape 错误。

#### `MultiHeadAttention(num_heads, key_dim)` — Transformer 的心脏

```python
attn = layers.MultiHeadAttention(num_heads=8, key_dim=64)
output = attn(query=x, value=x, key=x)   # 自注意力：三者相同
```

让序列中每个位置直接"看到"所有其他位置。与 LSTM 区别：不按顺序逐步处理，全局并行。

### 归一化与工具组（4个）

#### `LayerNormalization()` — Transformer 标配

```python
x = layers.LayerNormalization()(x + attn_output)   # 残差结构
```

与 BatchNorm 区别一句话：**BatchNorm 跨样本归一（适合 CNN），LayerNorm 在单个样本内归一（适合序列/Transformer，不受 batch size 影响）**。

#### `GlobalAveragePooling2D()` — 现代 CNN 的收尾

```python
layers.GlobalAveragePooling2D()   # (batch, 7, 7, 512) → (batch, 512)
```

每个通道整张图取平均。比 `Flatten` 参数少得多、更抗过拟合，**迁移学习时几乎总用它**。

#### `Concatenate()` — 拼接多个分支

```python
merged = layers.Concatenate()([branch_a, branch_b])  # 沿最后一维拼接
```

#### `Rescaling(scale)` — 把预处理写进模型

```python
layers.Rescaling(1./255)   # 像素 0–255 → 0–1
```

好处：归一化成为模型的一部分，部署时不会忘记预处理。这类"预处理层"是 Keras 3 的推荐风格。

---

## 五、训练三件套

### Optimizer：怎么更新权重

```python
keras.optimizers.Adam(learning_rate=1e-3)    # 万能默认，先用它
keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)  # 训 Transformer 标配
```

**唯一需要调的参数就是 `learning_rate`**：
- 默认 `1e-3`
- 训不动 → 调大；loss 震荡/爆炸 → 调小（试 `1e-4`）
- 微调预训练模型用更小的（`1e-5`）

### Loss：怎么衡量"错得多离谱"（最容易选错）

选择只看两件事：**任务类型 + 标签格式**。

| 任务 | 标签格式 | Loss | 输出层 |
| --- | --- | --- | --- |
| 多分类 | 整数 `3` | `SparseCategoricalCrossentropy` | `Dense(N, "softmax")` |
| 多分类 | one-hot `[0,0,0,1,0]` | `CategoricalCrossentropy` | `Dense(N, "softmax")` |
| 二分类 | `0` 或 `1` | `BinaryCrossentropy` | `Dense(1, "sigmoid")` |
| 回归 | 连续值 `3.72` | `MeanSquaredError` | `Dense(1)` 无激活 |

> Sparse 的意思就是"标签是整数而非 one-hot"。MNIST 标签是 `5` 这样的整数，所以用 Sparse 版。选错的典型症状：shape 报错或 loss 完全不降。

#### 进阶坑：`from_logits`

```python
# 写法A：输出层带 softmax（直观）
layers.Dense(10, activation="softmax")
loss = keras.losses.SparseCategoricalCrossentropy()

# 写法B：输出层不带激活，loss 内部算（数值更稳定，推荐）
layers.Dense(10)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

两种都对，但**别混搭**——softmax + `from_logits=True` 是经典 bug：模型能训但效果差一截。

### Metric：给人看的指标（不参与训练）

```python
metrics=["accuracy"]              # 分类基本款
metrics=[keras.metrics.AUC()]     # 类别不平衡时比 accuracy 诚实
```

> 什么时候不能只看 accuracy：99% 样本是负类时，全猜负类也有 99% 准确率。这时看 AUC。

---

## 六、流程四方法

```python
# ① compile：组装三件套（只是配置，不计算）
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ② fit：训练
history = model.fit(
    x_train, y_train,
    batch_size=32,            # 每次喂多少样本，常用 32/64/128
    epochs=20,                # 完整过几遍数据
    validation_split=0.2,     # 自动切 20% 做验证集
    callbacks=[...],
)
# history.history 是 dict：{"loss": [...], "val_loss": [...], ...}
# 画 val_loss 曲线是诊断训练的第一手段

# ③ evaluate：测试集上算最终成绩
test_loss, test_acc = model.evaluate(x_test, y_test)

# ④ predict：实际使用
probs = model.predict(x_new)        # softmax 输出 → 每类概率
preds = probs.argmax(axis=-1)       # 取概率最大的类别
```

### 读懂 fit 输出（核心技能）

| 现象 | 诊断 |
| --- | --- |
| `loss` 降、`val_loss` 也降 | 健康，继续 |
| `loss` 降、`val_loss` 开始升 | **过拟合**，从转折点该停（EarlyStopping 的工作） |
| 两者都不降 | 学习率/loss 选错，或数据有问题 |

> [!warning] 坑
> `validation_split` 直接切走**最后** 20% 数据。若数据按类别排过序，验证集会全是同一类——先 shuffle。

---

## 七、四个回调（Callbacks）

回调 = 训练过程中自动执行的钩子，每个 epoch 结束时触发。**前三个组合是生产级训练的标准配置**：

```python
callbacks = [
    # ① 验证集不再进步就自动停，并回滚到最佳权重
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,                  # 容忍 5 个 epoch 没进步
        restore_best_weights=True,   # ⚠️ 必加！否则拿到的是变差后的权重
    ),
    # ② 自动保存最佳模型到磁盘
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    # ③ 卡住时自动降学习率（先大步走，后小步精调）
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,        # 学习率减半
        patience=3,        # 要比 EarlyStopping 的 patience 小：先降 lr 再放弃
    ),
    # ④ 可视化训练过程
    keras.callbacks.TensorBoard(log_dir="./logs"),
    # 终端: tensorboard --logdir ./logs
]

model.fit(..., epochs=100, callbacks=callbacks)
# 有了 ①，epochs 可以放心设大，反正会自动停
```

---

## 八、完整示例：MNIST（25 个 API 的合奏）

```python
import keras
from keras import layers

# 数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[..., None]   # (60000,28,28) → (60000,28,28,1)
x_test = x_test[..., None]

# 模型（Functional API）
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10)(x)              # 无激活 → logits
model = keras.Model(inputs, outputs)

# 三件套
model.compile(
    optimizer=keras.optimizers.AdamW(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 训练
model.fit(
    x_train, y_train,
    batch_size=128, epochs=50, validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ],
)

# 评估
print(model.evaluate(x_test, y_test))
```

---

## 九、学习路径

### 三阶段

1. **阶段 1：跑通主线（1–2 天）** — 跑通 MNIST 全流程，然后改参数做实验（见下面的练习清单）。建立"生命周期"肌肉记忆。
2. **阶段 2：解锁灵活性（3–5 天）**
   - Sequential → **Functional API**（多输入/多输出/分支）
   - 用 `keras.applications.ResNet50` 做一次**迁移学习**（冻结 + 微调）
   - 写一个自定义 `Layer`（继承 `keras.Layer`，实现 `call()`），自然用到 `keras.ops`
3. **阶段 3：按需深入（长期）** — 自定义训练循环（重写 `train_step`）、自定义 Loss / Callback / Metric。模式都一样：**继承基类 + 重写 1–2 个方法**。

### 动手练习清单

- [ ] 跑通第八节的 MNIST 示例
- [ ] 删掉 BatchNorm，观察 val_loss 变化
- [ ] 把 lr 改成 `1e-1`，看 loss 爆炸长什么样
- [ ] 把 loss 换成不带 Sparse 的版本，看报什么错（**见过错误比记住正确更有用**）
- [ ] 用 Functional API 写一个双分支模型（练 `Concatenate`）
- [ ] 用 `keras.datasets.imdb` 走一遍 NLP 流程：`TextVectorization → Embedding → GRU → Dense`
- [ ] 用 `MultiHeadAttention + LayerNormalization` 手搭一个 Transformer block

### 三个学习技巧

1. **记"入口"不记 API**：只记 `keras.layers.`、`keras.losses.` 这些命名空间，IDE 补全 + [keras.io/api](https://keras.io/api/) 是外部记忆。没人背得住 600 个。
2. **官方 Examples 是最好的教材**：[keras.io/examples](https://keras.io/examples/) 有 100+ 个端到端示例（CV/NLP/生成式），每个 ~100 行，挑方向精读 3–5 个。
3. **读源码祛魅**：Keras 源码异常干净。`Dense` 核心就 10 行（matmul + bias + activation）。觉得某个 API 神秘时，点文档页的 `[source]` 看一眼。

> **一句话总结**：框架按生命周期分 6 类 → 精通 25 个核心 API → 用 Functional API + 自定义 Layer 打通任督二脉 → 其余 575 个用时再查。
