# PyTorch API 统计与学习路线

> 基于 PyTorch 2.12 官方文档 Sphinx 索引（`objects.inv`）统计，整理于 2026-06-07
> 文档入口：[docs.pytorch.org](https://docs.pytorch.org/docs/stable/) · 教程：[Learn the Basics](https://docs.pytorch.org/tutorials/)

## 一、API 数量统计

| 维度 | 数量 |
|---|---|
| 官方文档收录的 Python API 总数 | **7,282** |
| 其中：方法（如 `Tensor.view()`） | 2,633 |
| 其中：函数（如 `torch.matmul()`） | 1,908 |
| 其中：类（如 `nn.Linear`） | 1,273 |
| 其中：模块（如 `torch.nn`） | 818 |
| 其中：属性/特性（如 `Tensor.shape`） | 627 |
| 其中：异常、数据常量 | 23 |

### 按模块分布（Top 10）

| 模块 | API 数 | 用途 |
|---|---|---|
| `torch.nn` | 1,078 | 神经网络层、损失函数 |
| `torch.distributed` | 707 | 分布式训练 |
| `torch.Tensor` | 547 | 张量方法 |
| `torch.distributions` | 547 | 概率分布 |
| `torch.optim` | 522 | 优化器 |
| `torch`（顶层） | 518 | 张量创建与数学运算 |
| `torch.fx` | 461 | 图变换/编译 |
| `torch.ao` | 448 | 量化 |
| `torch.cuda` | 221 | GPU 管理 |
| `torch.utils` | 156 | DataLoader 等工具 |

### 结论

- 文档收录 Python API 共 **7,282 个**，但日常开发 90% 的时间只用其中不到 1%
- **学透 PyTorch = 精通 ~70 个 API + 理解 1 个核心机制（autograd）**，其余是"字典"，会查就行

## 二、API 分类：9 层"积木"金字塔

不按模块背，按**"训练一个模型的生命周期"**分层：

```
┌─────────────────────────────────────────────────┐
│  ⑨ 专业领域    distributions / ao量化 / fx图变换    │  ← 用到再学
│  ⑧ 规模化      distributed / FSDP / DDP           │  ← 多卡才学
│  ⑦ 工程部署    compile / export / onnx / profiler  │  ← 上线才学
├─────────────────────────────────────────────────┤
│  ⑥ 硬件管理    cuda / mps / device                 │  ← 会 3 个 API 就够
│  ⑤ 训练引擎    optim / lr_scheduler / 损失函数      │  ★ 必学
│  ④ 建模组件    nn / nn.functional / nn.init        │  ★ 必学
│  ③ 求导引擎    autograd / backward / no_grad       │  ★ 必学(理解>记忆)
│  ② 数据管道    utils.data / Dataset / DataLoader   │  ★ 必学
│  ① 张量底座    torch顶层 / Tensor 方法              │  ★ 必学(地基)
└─────────────────────────────────────────────────┘
```

| 层 | 涉及模块 | API 总量 | 真正常用 | 学习策略 |
|---|---|---|---|---|
| ① 张量底座 | `torch` + `torch.Tensor` | ~1,065 | **~30 个** | 精学，肌肉记忆 |
| ② 数据管道 | `torch.utils.data` | ~156 | **~5 个** | 记模板 |
| ③ 求导引擎 | `torch.autograd` | ~133 | **~4 个** | 理解原理 |
| ④ 建模组件 | `torch.nn` | ~1,078 | **~20 个** | 学一个会一类 |
| ⑤ 训练引擎 | `torch.optim` | ~522 | **~5 个** | 记模板 |
| ⑥ 硬件管理 | `cuda/mps` | ~250 | **~3 个** | 知道就行 |
| ⑦⑧⑨ | 其余 | ~4,000 | 0 | 入门阶段完全不看 |

> [!tip] 为什么 `torch.nn` 的 1,078 个只需学 20 个？
> 因为它们是**同构的**——都遵循同一个协议：*继承 `nn.Module`、参数自动注册、调用即 forward*。学会 `nn.Linear` 就等于学会了 `nn.Conv2d`、`nn.LSTM`……**学一个，会一类**。

## 三、核心心智模型

> **Tensor 在流动，autograd 在记账，Module 在组装，optim 在还账。**

- 数据是 `Tensor`，它流过运算时 autograd 偷偷记下每一步（计算图）
- `nn.Module` 只是把"带参数的运算"打包成乐高积木
- `loss.backward()` 让 autograd 沿着账本反向算出梯度
- `optimizer.step()` 按梯度更新参数

所有 7,282 个 API 都只是这四个角色的具体化。

## 四、必学代码模板

### ① 张量基础

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])   # 创建
torch.zeros(2, 3); torch.randn(2, 3)      # 常用初始化
x.shape, x.dtype, x.device                # 三大属性
x.view(4), x.reshape(-1)                  # 变形
x @ x.T, x.sum(), x.mean(dim=0)           # 运算
x.to('mps')                               # Mac 上用 MPS 加速（N 卡用 'cuda'）
```

### ③ 自动求导（PyTorch 的灵魂）

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # dy/dx = 2x+3 = 7
```

### ④ 搭网络（万能模板）

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### ⑤ 训练循环（万能五行）

```python
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for x, y in dataloader:
    optimizer.zero_grad()        # 1. 清空梯度
    loss = loss_fn(model(x), y)  # 2. 前向 + 算损失
    loss.backward()              # 3. 反向传播
    optimizer.step()             # 4. 更新参数
```

### ② 数据加载

```python
from torch.utils.data import Dataset, DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 造轮子：手搓线性回归（学透 ①③⑤ 层的试金石）

```python
# 不用 nn、不用 optim，只用 Tensor + autograd
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for _ in range(100):
    loss = ((w * x + b - y) ** 2).mean()   # 前向：autograd 在记账
    loss.backward()                         # 反向：算出 w.grad, b.grad
    with torch.no_grad():                   # 更新时不许记账
        w -= 0.01 * w.grad; b -= 0.01 * b.grad
        w.grad.zero_(); b.grad.zero_()      # 这就是 optimizer.zero_grad() 的本质
```

写完这 10 行就明白了 `nn.Module`、`optim.SGD`、`zero_grad()` 都只是它的封装——**祛魅了，就学透了**。

## 五、三遍学习法

| 遍数 | 做什么 | 检验标准 |
|---|---|---|
| **第一遍：跑通** | 复制官方教程代码，改参数看变化 | 代码能跑，知道每行干嘛 |
| **第二遍：手写** | 合上文档，从空白文件默写训练循环 | 5 分钟内不查文档写出 MNIST 训练 |
| **第三遍：造轮子** | 用裸 Tensor + autograd 手搓线性层和 SGD | 能解释 `zero_grad()` 不写会发生什么 |

第三遍是大多数人跳过、但**拉开差距**的一步。

## 六、四周通关路线

| 周 | 主题 | 产出物（必须动手） |
|---|---|---|
| **W1** | ①张量 + ③autograd | 手搓线性回归 |
| **W2** | ④nn + ⑤optim + ②data | 默写 MNIST 全连接网络，准确率 >97% |
| **W3** | 真实项目 | CIFAR-10 卷积网络 + 数据增强 + 学习率调度，>85% |
| **W4** | 迁移学习 | 用预训练 ResNet 微调自己的数据集 |

### 每周自检问题

- **W1**：解释 `view` vs `reshape`；为什么 `a += b` 有时报 autograd 错
- **W2**：解释 `model.train()` vs `model.eval()`；CrossEntropyLoss 为什么不要先 softmax
- **W3**：解释过拟合曲线长什么样；Dropout 在 eval 时为什么自动关闭

## 七、三条反模式

1. ❌ **按文档目录顺序学** → 会在 `torch.special` 的 56 个贝塞尔函数里迷路。✅ 按 9 层金字塔，只走 ①→⑤
2. ❌ **背 API 签名** → 没人记得住 `Conv2d` 的 9 个参数。✅ 记"它解决什么问题"，参数现查（`help(nn.Conv2d)`）
3. ❌ **只看不写** → 视频看 10 小时不如手写 1 小时。✅ 每个概念必须落到能跑的代码

## 相关笔记

- [[numpy]] — Tensor 的 API 设计大量借鉴 NumPy（`reshape`、`sum(dim=…)` 对应 `axis=…`）
- [[pandas]]
