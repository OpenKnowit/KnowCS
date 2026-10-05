# PyTorch API Stats & Learning Path

> Counted from the Sphinx index (`objects.inv`) of the official PyTorch 2.12 docs, compiled 2026-06-07
> Docs: [docs.pytorch.org](https://docs.pytorch.org/docs/stable/) · Tutorial: [Learn the Basics](https://docs.pytorch.org/tutorials/)

## 1. API Counts

| Dimension | Count |
|---|---|
| Total Python APIs in the official docs | **7,282** |
| of which: methods (e.g. `Tensor.view()`) | 2,633 |
| of which: functions (e.g. `torch.matmul()`) | 1,908 |
| of which: classes (e.g. `nn.Linear`) | 1,273 |
| of which: modules (e.g. `torch.nn`) | 818 |
| of which: attributes / properties (e.g. `Tensor.shape`) | 627 |
| of which: exceptions, data constants | 23 |

### Distribution by module (top 10)

| Module | APIs | Purpose |
|---|---|---|
| `torch.nn` | 1,078 | Neural-network layers, loss functions |
| `torch.distributed` | 707 | Distributed training |
| `torch.Tensor` | 547 | Tensor methods |
| `torch.distributions` | 547 | Probability distributions |
| `torch.optim` | 522 | Optimizers |
| `torch` (top level) | 518 | Tensor creation and math |
| `torch.fx` | 461 | Graph transforms / compilation |
| `torch.ao` | 448 | Quantization |
| `torch.cuda` | 221 | GPU management |
| `torch.utils` | 156 | DataLoader and other utilities |

### Takeaways

- The docs list **7,282** Python APIs, yet 90% of everyday work uses less than 1% of them
- **Mastering PyTorch = knowing ~70 APIs well + understanding 1 core mechanism (autograd)**. The rest is a "dictionary": you just need to know how to look things up

## 2. Classifying the API: a 9-Layer Pyramid of Building Blocks

Don't memorise by module; layer it by **the lifecycle of training a model**:

```
┌─────────────────────────────────────────────────────┐
│  ⑨ Specialist   distributions / ao quant / fx graph   │  ← learn when needed
│  ⑧ Scale-out    distributed / FSDP / DDP              │  ← only for multi-GPU
│  ⑦ Deployment   compile / export / onnx / profiler    │  ← only for production
├─────────────────────────────────────────────────────┤
│  ⑥ Hardware     cuda / mps / device                   │  ← 3 APIs are enough
│  ⑤ Training     optim / lr_scheduler / loss functions │  ★ must learn
│  ④ Modelling    nn / nn.functional / nn.init          │  ★ must learn
│  ③ Autograd     autograd / backward / no_grad         │  ★ must learn (understand > memorise)
│  ② Data         utils.data / Dataset / DataLoader     │  ★ must learn
│  ① Tensors      torch top level / Tensor methods      │  ★ must learn (the foundation)
└─────────────────────────────────────────────────────┘
```

| Layer | Modules | Total APIs | Actually used | Strategy |
|---|---|---|---|---|
| ① Tensors | `torch` + `torch.Tensor` | ~1,065 | **~30** | Learn thoroughly, build muscle memory |
| ② Data | `torch.utils.data` | ~156 | **~5** | Memorise the template |
| ③ Autograd | `torch.autograd` | ~133 | **~4** | Understand the principle |
| ④ Modelling | `torch.nn` | ~1,078 | **~20** | Learn one, know the whole family |
| ⑤ Training | `torch.optim` | ~522 | **~5** | Memorise the template |
| ⑥ Hardware | `cuda/mps` | ~250 | **~3** | Just know they exist |
| ⑦⑧⑨ | the rest | ~4,000 | 0 | Ignore completely as a beginner |

> **Tip: why only 20 of the 1,078 `torch.nn` APIs?**
> Because they are **isomorphic**: they all follow the same protocol (*subclass `nn.Module`, parameters register automatically, calling the module runs forward*). Once you know `nn.Linear`, you effectively know `nn.Conv2d`, `nn.LSTM`… **learn one, know the whole family**.

## 3. The Core Mental Model

> **Tensors flow, autograd keeps the books, Modules assemble, optim settles the account.**

- Data is a `Tensor`; as it flows through operations, autograd quietly records every step (the computation graph)
- `nn.Module` just packages "operations with parameters" into Lego bricks
- `loss.backward()` makes autograd walk the ledger backwards to compute gradients
- `optimizer.step()` updates the parameters using those gradients

All 7,282 APIs are just concrete forms of these four roles.

## 4. Must-Know Code Templates

### ① Tensor basics

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])   # create
torch.zeros(2, 3); torch.randn(2, 3)      # common initialisers
x.shape, x.dtype, x.device                # the three key attributes
x.view(4), x.reshape(-1)                  # reshape
x @ x.T, x.sum(), x.mean(dim=0)           # arithmetic
x.to('mps')                               # MPS acceleration on a Mac ('cuda' on NVIDIA)
```

### ③ Autograd (the soul of PyTorch)

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # dy/dx = 2x+3 = 7
```

### ④ Building a network (the universal template)

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

### ⑤ Training loop (the universal five lines)

```python
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for x, y in dataloader:
    optimizer.zero_grad()        # 1. clear gradients
    loss = loss_fn(model(x), y)  # 2. forward pass + loss
    loss.backward()              # 3. backpropagate
    optimizer.step()             # 4. update parameters
```

### ② Loading data

```python
from torch.utils.data import Dataset, DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Build it yourself: linear regression by hand (the litmus test for layers ①③⑤)

```python
# no nn, no optim: only Tensor + autograd
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for _ in range(100):
    loss = ((w * x + b - y) ** 2).mean()   # forward: autograd keeps the books
    loss.backward()                         # backward: computes w.grad, b.grad
    with torch.no_grad():                   # no bookkeeping during the update
        w -= 0.01 * w.grad; b -= 0.01 * b.grad
        w.grad.zero_(); b.grad.zero_()      # this is all optimizer.zero_grad() really does
```

After writing these 10 lines you'll see that `nn.Module`, `optim.SGD` and `zero_grad()` are just wrappers around it. **Once the magic is gone, you've truly learned it.**

## 5. The Three-Pass Method

| Pass | What to do | How to check |
|---|---|---|
| **Pass 1: run it** | Copy the official tutorial code, change parameters and watch what happens | The code runs and you know what each line does |
| **Pass 2: write it** | Close the docs and write the training loop from a blank file | Write MNIST training in under 5 minutes without the docs |
| **Pass 3: build it** | Hand-roll a linear layer and SGD with bare Tensors + autograd | Explain what happens if you forget `zero_grad()` |

Pass 3 is the step most people skip, and the one that **sets you apart**.

## 6. A Four-Week Roadmap

| Week | Topic | Deliverable (you must build it) |
|---|---|---|
| **W1** | ① tensors + ③ autograd | Hand-rolled linear regression |
| **W2** | ④ nn + ⑤ optim + ② data | Fully-connected MNIST network from memory, accuracy > 97% |
| **W3** | A real project | CIFAR-10 CNN + data augmentation + LR scheduling, > 85% |
| **W4** | Transfer learning | Fine-tune a pretrained ResNet on your own dataset |

### Weekly self-check questions

- **W1**: explain `view` vs `reshape`; why does `a += b` sometimes raise an autograd error?
- **W2**: explain `model.train()` vs `model.eval()`; why shouldn't you apply softmax before CrossEntropyLoss?
- **W3**: what does an overfitting curve look like? Why is Dropout turned off automatically in eval mode?

## 7. Three Anti-Patterns

1. ❌ **Learning in the order of the docs' table of contents** → you'll get lost among the 56 Bessel functions in `torch.special`. ✅ Follow the 9-layer pyramid and only walk ①→⑤
2. ❌ **Memorising API signatures** → nobody remembers the 9 parameters of `Conv2d`. ✅ Remember "what problem it solves" and look up parameters when needed (`help(nn.Conv2d)`)
3. ❌ **Watching without writing** → 10 hours of videos are worth less than 1 hour of writing code. ✅ Every concept must end up as code that runs

## Related Notes

- NumPy: tensor APIs borrow heavily from NumPy (`reshape`, and `sum(dim=…)` corresponds to `axis=…`)
- pandas
