# NumPy API 与模块结构

> 基于本机 NumPy 2.2.6 实测 + 官方文档 [Module structure](https://numpy.org/doc/stable/reference/module_structure.html)，整理于 2026-06-07

## 一、API 数量统计

| 维度 | 数量 |
|---|---|
| 顶层公开 API（`np.__all__`） | **499** |
| 其中：函数/可调用对象 | 392（含 106 个 ufunc） |
| 其中：类（如 `ndarray`、`dtype`） | 70 |
| 其中：常量（如 `pi`、`e`、`inf`） | 18 |
| `ndarray` 公开方法/属性 | 74 |

### 子模块 API 数

| 子模块 | API 数 | 子模块 | API 数 |
|---|---|---|---|
| `np.ma`（掩码数组） | 226 | `np.linalg` | 32 |
| `np.random` | 60 | `np.fft` | 18 |
| `np.char` | 53 | `np.polynomial` | 13 |
| `np.testing` | 49 | `np.emath` | 9 |
| `np.strings` | 45 | `np.rec` | 9 |
| `np.dtypes` | 33 | `np.exceptions` | 6 |

### 结论

- 顶层 API ≈ **500 个**
- 加上子模块约 **1070 个**（有重复，`np.ma` 大量镜像顶层函数）
- 算上 `ndarray` 方法、`random.Generator` 等类方法，完整 API 面约 **1200+**
- 常见口径「NumPy 约 600 个公开函数」= 去重后的顶层函数 + 主要子模块函数

## 二、官方模块结构（20 个公开命名空间，分三层）

### ① Main namespaces — 推荐日常使用（9 个）

| 模块 | 用途 |
|---|---|
| `numpy` | 主命名空间：数组创建、数学运算、索引等核心功能 |
| `numpy.exceptions` | NumPy 专属的异常和警告类 |
| `numpy.fft` | 快速傅里叶变换 |
| `numpy.linalg` | 线性代数（求逆、特征值、SVD…） |
| `numpy.polynomial` | 多项式表示与运算 |
| `numpy.random` | 随机数生成、概率分布 |
| `numpy.strings` | 字符串数组操作（2.0 新增，替代 `np.char`） |
| `numpy.testing` | 测试断言工具（`assert_allclose` 等） |
| `numpy.typing` | 类型注解支持（`NDArray`、`ArrayLike`） |

### ② Special-purpose namespaces — 特定场景（6 个）

| 模块 | 用途 |
|---|---|
| `numpy.ctypeslib` | 和 C 库 / ctypes 互操作 |
| `numpy.dtypes` | dtype 类的定义（一般不直接用） |
| `numpy.emath` | 自动切换定义域的数学函数（`sqrt(-1)` 返回复数而非 nan） |
| `numpy.lib` | 放不进主命名空间的杂项工具 |
| `numpy.rec` | 记录数组（基本被 pandas 取代） |
| `numpy.version` | 详细版本信息 |

### ③ Legacy namespaces — 官方不建议新代码使用（5 个）

| 模块 | 状态 |
|---|---|
| `numpy.char` | 定宽字符串遗留功能 → 用 `np.strings` 替代 |
| `numpy.distutils` | 已弃用的构建系统（Python 3.12 起移除） |
| `numpy.f2py` | Fortran 绑定生成器，一般只在命令行用 |
| `numpy.ma` | 掩码数组，官方自评「不太可靠，需要重构」 |
| `numpy.matlib` | 给 `matrix` 类用的，待弃用 |

### 关于「30 个子模块」的说法

安装目录里实际有 **34 个**子模块/子包，但多出来的不算公开 API：

- **私有模块**（11 个）：`_core`、`_typing`、`_utils` 等下划线开头 = 内部实现，随时可能变
- **兼容性空壳**：`core`、`compat`、`matrixlib` —— 2.0 后只是指向 `_core` 的别名，导入有 deprecation 警告
- **非 API 文件**：`conftest`、`tests`、`__config__` 等测试和构建产物

记忆口径：**对用户有意义的是那 20 个，日常高频的只有 `numpy` 本身 + `fft` / `linalg` / `random` / `polynomial`**。

## 三、`numpy.polynomial` 详解

把多项式当作**对象**处理——定义、求值、求根、求导积分、运算、拟合，一行搞定。

### 核心用法

```python
from numpy.polynomial import Polynomial

p = Polynomial([1, 2, 3])   # 系数从低次到高次 → p(x) = 1 + 2x + 3x²
p(2)          # 代入求值 → 17.0
p.roots()     # 求根（可返回复根）
p.deriv()     # 求导 → 2 + 6x
p.integ()     # 积分 → x + x² + x³
p * q         # 多项式乘法，直接用运算符
```

### 最常用场景：曲线拟合

```python
fit = Polynomial.fit(x, y, deg=2)   # 二次多项式拟合数据点
fit.convert()                        # 还原成标准系数形式
```

实测：对带噪声的 `0.5x² - 2x + 3` 数据拟合，得到 `0.52x² - 2.15x + 3.12`。

### 六种多项式类

| 类 | 基函数 | 用途 |
|---|---|---|
| `Polynomial` | 普通幂级数 1, x, x² | 日常用这个就够 |
| `Chebyshev` | 切比雪夫多项式 | 数值逼近标准选择，区间端点不震荡 |
| `Legendre` | 勒让德多项式 | 物理、球谐函数相关 |
| `Hermite` / `HermiteE` | 埃尔米特多项式 | 量子力学、概率论 |
| `Laguerre` | 拉盖尔多项式 | 氢原子径向方程等 |

正交多项式在**数值积分（高斯求积）和函数逼近**中很重要——`Chebyshev.fit()` 拟合高次多项式比普通幂级数数值稳定得多。

### 历史注意点

老代码常见 `np.polyfit()` / `np.poly1d` 是旧 API（系数顺序从高次到低次，是反的）。官方现推荐 `numpy.polynomial`，新代码别用 `poly1d`。
