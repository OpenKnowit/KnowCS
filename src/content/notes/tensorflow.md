# TensorFlow API 数量与模块结构

> 基于官方文档 [All symbols 页面](https://www.tensorflow.org/api_docs/python/tf/all_symbols)（TensorFlow v2.16.1）统计，整理于 2026-06-07

## 一、API 数量统计

| 口径 | 数量 |
|---|---|
| 全部公开 API 符号名（含别名，如 `tf.add` 和 `tf.math.add` 算两个） | **6,571** |
| 去重后的独立 API 文档页（同一 API 的别名只算一个） | **约 4,821** |
| 剔除 `compat` 兼容层和 `raw_ops` 底层算子后的「核心 API」 | **2,956** |
| 直接挂在 `tf.` 顶层的符号（如 `tf.constant`、`tf.GradientTape`） | **273** |

## 二、按顶层模块分布

### 三大头部模块（占总量 ~75%）

| 模块 | 符号数 | 说明 |
|---|---|---|
| `tf.compat.*` | 2,178 | TF1 兼容层（以 `compat.v1` 为主），新代码不应使用 |
| `tf.raw_ops.*` | 1,437 | 底层原始算子，自动生成，一般不直接调用 |
| `tf.keras.*` | 1,306 | Keras 高层 API（层、模型、优化器、损失等），日常建模主力 |

### 其余主要模块

| 模块 | 符号数 | 模块 | 符号数 |
|---|---|---|---|
| `tf.experimental` | 307 | `tf.train` | 36 |
| `tf.math` | 150 | `tf.signal` | 35 |
| `tf.nn` | 93 | `tf.debugging` | 35 |
| `tf.data` | 76 | `tf.sparse` | 31 |
| `tf.image` | 72 | `tf.strings` | 27 |
| `tf.linalg` | 71 | `tf.tpu` | 26 |
| `tf.distribute` | 66 | `tf.summary` | 22 |
| `tf.io` | 62 | `tf.quantization` | 21 |
| `tf.config` | 52 | `tf.errors` | 19 |
| `tf.random` | 39 | `tf.lite` | 15 |

其他小模块：`saved_model`、`profiler`、`ragged`、`lookup`、`autograph`、`dtypes`、`bitwise`、`sets`、`xla`、`audio` 等，各 < 15 个。

## 三、结论

- 官方文档口径下，TensorFlow Python API 共 **约 6,600 个符号 / 约 4,800 个独立 API**
- 但其中 1/3 是 TF1 兼容层、1/5 是底层 raw_ops，**实际日常会用到的核心 API 约 3,000 个**，且高度集中在 `tf.keras`（建模）、`tf.math`/`tf.nn`（运算）、`tf.data`（数据管道）几个模块
- 对比：NumPy 顶层约 500 个（见 [[numpy]]），TensorFlow 的 API 面大一个数量级，主要因为深度学习栈覆盖了从底层算子到高层建模的全链路

## 四、统计口径说明

- 仅统计 **Python API**；TensorFlow 还有 C++、Java、JavaScript（TF.js）、TFLite 等其他语言绑定，不在此列
- "符号"含别名：很多函数同时挂在多个命名空间（如 `tf.add` = `tf.math.add`），按文档页去重后约少 1,750 个
- 统计方法：抓取官方 all_symbols 页面 HTML，正则提取 `tf.*` 符号名后去重计数
