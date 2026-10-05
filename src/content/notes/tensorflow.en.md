# TensorFlow API Counts & Module Structure

> Counted from the official [All symbols page](https://www.tensorflow.org/api_docs/python/tf/all_symbols) (TensorFlow v2.16.1), compiled 2026-06-07

## 1. API Counts

| Measure | Count |
|---|---|
| All public API symbol names (aliases included, e.g. `tf.add` and `tf.math.add` count as two) | **6,571** |
| Distinct API doc pages after deduplication (aliases of one API count once) | **about 4,821** |
| "Core API" after removing the `compat` layer and `raw_ops` low-level ops | **2,956** |
| Symbols directly under the top-level `tf.` (e.g. `tf.constant`, `tf.GradientTape`) | **273** |

## 2. Distribution by Top-Level Module

### The three largest modules (~75% of the total)

| Module | Symbols | Notes |
|---|---|---|
| `tf.compat.*` | 2,178 | TF1 compatibility layer (mostly `compat.v1`); new code should not use it |
| `tf.raw_ops.*` | 1,437 | Low-level raw ops, auto-generated, rarely called directly |
| `tf.keras.*` | 1,306 | Keras high-level API (layers, models, optimizers, losses…), the workhorse for everyday modelling |

### Other major modules

| Module | Symbols | Module | Symbols |
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

Other small modules: `saved_model`, `profiler`, `ragged`, `lookup`, `autograph`, `dtypes`, `bitwise`, `sets`, `xla`, `audio`, etc., each with fewer than 15.

## 3. Takeaways

- By the official docs, the TensorFlow Python API has **about 6,600 symbols / about 4,800 distinct APIs**
- But a third of them are the TF1 compatibility layer and a fifth are low-level raw_ops, so **the core API you would actually use is about 3,000**, concentrated in `tf.keras` (modelling), `tf.math` / `tf.nn` (computation) and `tf.data` (data pipelines)
- For comparison: NumPy has about 500 top-level APIs (see the NumPy note). TensorFlow's API surface is an order of magnitude larger because a deep-learning stack covers everything from low-level ops to high-level modelling

## 4. How the Counts Were Made

- Only the **Python API** is counted; TensorFlow's other bindings (C++, Java, JavaScript / TF.js, TFLite) are excluded
- "Symbols" include aliases: many functions live in several namespaces (e.g. `tf.add` = `tf.math.add`); deduplicating by doc page removes about 1,750
- Method: scrape the HTML of the official all_symbols page, extract `tf.*` symbol names with a regex, deduplicate and count
