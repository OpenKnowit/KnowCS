// PyTorch 实验台：第 9 讲的张量、自动求导、nn.Module 与训练循环；示例数据都在代码里现场生成（沙盒没有 MNIST）
import { MATPLOTLIB } from '../../lib/pyPlot'
import { TORCH } from '../../lib/pyTorch'
import type { PlayConfig, PlayEntry } from './types'

const H = 'import torch\n\n'
const NN = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'

const ENTRIES: PlayEntry[] = [
  // ---- tensors
  {
    id: 'attrs', cat: 'tensors', label: 'shape / dtype / device',
    code: `${H}tensor = torch.tensor([1, 3, 5])          # lecture 9's first example
print(f"Shape of tensor: {tensor.shape}")
print(f"Data type of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

data = torch.tensor([[1, 2], [3, 4]])
print(torch.ones_like(data))                 # keeps shape and dtype (int64)
print(torch.zeros(2, 3).dtype)               # factory functions default to float32
print(torch.cuda.is_available())`,
  },
  {
    id: 'matmul', cat: 'tensors', label: 'tensor @ tensor.T vs *', focus: ['op:@'],
    code: `${H}tensor = torch.tensor([[1., 2.],
                       [3., 4.]])
y1 = tensor @ tensor.T       # matrix product: row · column
z1 = tensor * tensor         # element-wise product
print(y1)
print(z1)
print(torch.matmul(tensor, tensor.T))   # the same as @`,
  },
  {
    id: 'indexing', cat: 'tensors', label: 'tensor[:, 1] = 0',
    code: `${H}tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0                       # in place: column 1 becomes 0
print(tensor)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1.shape)`,
  },
  {
    id: 'view', cat: 'tensors', label: 'view / reshape / unsqueeze',
    code: `${H}x = torch.arange(12)
a = x.view(3, 4)            # same memory, new shape
b = x.view(2, -1)           # -1: "work it out" -> 6
print(a.shape, b.shape)

img = torch.zeros(28, 28)
batch = img.unsqueeze(0).unsqueeze(0)   # (1, 1, 28, 28): batch, channel, H, W
print(batch.shape)

t = a.T                     # a transpose is not contiguous ...
print(t.is_contiguous())
print(t.reshape(12).shape)  # reshape copies when it must; t.view(12) would fail`,
  },
  {
    id: 'bridge', cat: 'tensors', label: 'torch.from_numpy (shared)',
    code: `${H}import numpy as np

n = np.ones(5)
t = torch.from_numpy(n)     # shares memory with n
n += 1                      # change the array ...
print(t)                    # ... and the tensor sees it
t.add_(1)                   # in-place on the tensor ...
print(n)                    # ... changes the array

c = torch.tensor(n)         # torch.tensor copies
n += 1
print(c)`,
  },
  {
    id: 'dtype', cat: 'tensors', label: 'int64 vs float32',
    code: `${H}a = torch.tensor([1, 2, 3])      # Python ints -> int64
b = torch.tensor([1.0, 2.0])     # Python floats -> float32
print(a.dtype, b.dtype)
print(a / 2)                     # true division always gives floats
print((a * 2).dtype, (a * 2.5).dtype)
print(torch.tensor(0.1).item())  # float32 cannot store 0.1 exactly
print(a.float().mean())          # mean needs a float tensor`,
  },

  // ---- autograd
  {
    id: 'grad_basic', cat: 'autograd', label: 'y.backward()', focus: ['graph'],
    code: `${H}x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x          # dy/dx = 2x + 3 = 7 at x = 2
print(y)                    # grad_fn: y remembers how it was made
y.backward()
print(x.grad)`,
  },
  {
    id: 'grad_lecture', cat: 'autograd', label: 'loss.backward() → w.grad', focus: ['graph'],
    code: `${H}torch.manual_seed(0)
x = torch.ones(5)            # input tensor
y = torch.zeros(3)           # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()              # fills w.grad and b.grad
print(w.grad)
print(b.grad)`,
  },
  {
    id: 'accumulate', cat: 'autograd', label: 'grads accumulate',
    code: `${H}w = torch.tensor(1.0, requires_grad=True)
loss = 3 * w
loss.backward()
print(w.grad)              # 3

loss = 3 * w
loss.backward()            # no zeroing in between ...
print(w.grad)              # ... 6: gradients ADD UP

w.grad.zero_()             # what optimizer.zero_grad() does
loss = 3 * w
loss.backward()
print(w.grad)              # 3 again`,
  },
  {
    id: 'no_grad', cat: 'autograd', label: 'with torch.no_grad()',
    code: `${H}x = torch.ones(5)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w) + b
print(z.requires_grad)       # True: tracked for backward

with torch.no_grad():        # inference: no graph is built
    z = torch.matmul(x, w) + b
print(z.requires_grad)       # False

z_det = (torch.matmul(x, w) + b).detach()
print(z_det.requires_grad)   # False`,
  },
  {
    id: 'linreg', cat: 'autograd', label: 'gradient descent by hand',
    code: `${H}import matplotlib.pyplot as plt

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])     # y = 2x + 1
w = torch.zeros(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.05
losses = []
for epoch in range(200):
    loss = ((X @ w + b - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():          # update the leaves without recording it
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    losses.append(loss.item())

print(f"w = {w.item():.3f}, b = {b.item():.3f}")
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Loss while learning y = 2x + 1')`,
  },

  // ---- nn and shapes
  {
    id: 'mnist_cnn', cat: 'nn', label: 'class MNIST_CNN(nn.Module)', focus: ['flow'],
    code: `${NN}class MNIST_CNN(nn.Module):              # lecture 9's model
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # (batch, 32, 28, 28)
        x = F.relu(self.conv2(x))      # (batch, 64, 28, 28)
        x = self.pool(x)               # (batch, 64, 14, 14)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)      # (batch, 12544)
        x = F.relu(self.fc1(x))        # (batch, 128)
        x = self.dropout2(x)
        return self.fc2(x)             # raw logits: no softmax here

model = MNIST_CNN()
print(model)
print(sum(p.numel() for p in model.parameters()), 'parameters')
image = torch.randn(1, 1, 28, 28)       # one fake grey 28 x 28 image
print(model(image).shape)`,
  },
  {
    id: 'flatten_bug', cat: 'nn', label: 'wrong in_features', expectError: 'RuntimeError',
    code: `${NN}conv = nn.Conv2d(1, 16, kernel_size=3)       # no padding: 28 -> 26
pool = nn.MaxPool2d(2)                        # 26 -> 13
fc = nn.Linear(16 * 14 * 14, 10)              # BUG: assumes 14 x 14

x = torch.zeros(1, 1, 28, 28)
x = pool(F.relu(conv(x)))
print(x.shape)                                # the real shape
x = x.view(x.size(0), -1)
print(x.shape)
out = fc(x)                                   # fix: nn.Linear(16 * 13 * 13, 10)`,
  },
  {
    id: 'sequential', cat: 'nn', label: 'nn.Sequential + params', focus: ['flow'],
    code: `${NN}model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),     # (784 + 1) * 128 parameters
    nn.ReLU(),
    nn.Linear(128, 10),          # (128 + 1) * 10
)
print(model)
for name, p in model.named_parameters():
    print(name, tuple(p.shape), p.numel())
out = model(torch.zeros(4, 1, 28, 28))     # a batch of 4 images
print(out.shape)`,
  },
  {
    id: 'logits', cat: 'nn', label: 'CrossEntropyLoss(logits)',
    code: `${NN}logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.2]])      # raw scores from the last layer
labels = torch.tensor([0, 1])

loss = nn.CrossEntropyLoss()(logits, labels)  # applies log_softmax itself
print(loss)
probs = F.softmax(logits, dim=1)              # only for reading off probabilities
print(probs)
_, predicted = torch.max(logits, 1)           # lecture 9's way to pick the class
print(predicted, (predicted == labels).sum().item(), 'correct')`,
  },

  // ---- training
  {
    id: 'train_xor', cat: 'train', label: 'training loop (XOR)', focus: ['figure'],
    code: `${NN}import matplotlib.pyplot as plt
torch.manual_seed(0)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 0])                    # XOR: not linearly separable
model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

losses = []
model.train()
for epoch in range(150):
    optimizer.zero_grad()           # 1. reset gradients
    outputs = model(X)              # 2. forward pass
    loss = lossfunc(outputs, y)     # 3. loss
    loss.backward()                 # 4. gradients
    optimizer.step()                # 5. update the weights
    losses.append(loss.item())

model.eval()
with torch.no_grad():
    print(torch.argmax(model(X), 1))
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('cross-entropy')`,
  },
  {
    id: 'zero_grad_bug', cat: 'train', label: 'forgot zero_grad()',
    code: `${H}import matplotlib.pyplot as plt

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = 2 * X + 1

def train(zero_grad):
    w = torch.zeros(1, 1, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.02)
    losses = []
    for epoch in range(40):
        if zero_grad:
            opt.zero_grad()
        loss = ((X @ w - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

plt.plot(train(True), label='with zero_grad()')
plt.plot(train(False), label='without: gradients pile up')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()`,
  },
  {
    id: 'train_eval', cat: 'train', label: 'model.train() vs eval()',
    code: `${NN}torch.manual_seed(1)
model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
x = torch.ones(1, 4)

model.train()                 # dropout ON: random units zeroed, the rest x2
with torch.no_grad():
    print(model(x))
    print(model(x))           # different every call

model.eval()                  # dropout OFF: deterministic
with torch.no_grad():
    print(model(x))
    print(model(x))`,
  },
  {
    id: 'dataloader', cat: 'train', label: 'DataLoader batches',
    code: `${H}from torch.utils.data import TensorDataset, DataLoader

X = torch.arange(20.).view(10, 2)     # 10 samples, 2 features
y = torch.arange(10) % 2              # labels 0 / 1
loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)
print(len(loader), 'batches')         # ceil(10 / 4) = 3
for i, (inputs, labels) in enumerate(loader):
    print(i, inputs.shape, labels.tolist())`,
  },
  {
    id: 'cat_stack', cat: 'tensors', label: "torch.cat vs torch.stack",
    code: `${H}a = torch.zeros(2, 3)
b = torch.ones(2, 3)
print(torch.cat([a, b], dim=0).shape)    # joins along an existing axis
print(torch.cat([a, b], dim=1).shape)
print(torch.stack([a, b]).shape)         # adds a new axis in front`,
  },
  {
    id: 'conv_edge', cat: 'nn', label: "F.conv2d (Sobel edge)", focus: ['F.conv2d'],
    code: `${NN}img = torch.tensor([[0., 0., 0., 9., 9., 9.],
                    [0., 0., 0., 9., 9., 9.],
                    [0., 0., 0., 9., 9., 9.],
                    [0., 0., 0., 9., 9., 9.]]).reshape(1, 1, 4, 6)   # (batch, channels, H, W)
sobel_x = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).reshape(1, 1, 3, 3)
out = F.conv2d(img, sobel_x)        # no padding: (4 - 3 + 1) x (6 - 3 + 1)
print(out.shape)
print(out[0, 0])                    # strong response where dark meets bright`,
  },
  {
    id: 'softmax_acc', cat: 'train', label: "softmax → argmax → accuracy",
    code: `${H}logits = torch.tensor([[2.0, 0.5, -1.0],
                       [0.1, 1.5, 0.3],
                       [1.2, 0.2, 2.2]])
y = torch.tensor([0, 1, 1])
probs = torch.softmax(logits, dim=1)       # each row now sums to 1
pred = probs.argmax(dim=1)
print(probs)
print(pred)
print((pred == y).float().mean())          # accuracy: 2 of 3`,
  },
  // ---- what the sandbox gained later; each one was run on the real library
  {
    id: 'einsum', cat: 'maths', label: 'torch.einsum',
    code: `import torch

A = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([1., 1.])
print(torch.einsum('ij,j->i', A, b))     # A @ b
print(torch.einsum('ij,kj->ik', A, A))   # A @ A.T
print(torch.einsum('ii->', A))           # trace`,
  },
  {
    id: 'solve', cat: 'maths', label: 'torch.linalg',
    code: `import torch

A = torch.tensor([[2., 1.], [1., 3.]])
b = torch.tensor([3., 5.])
print(torch.linalg.solve(A, b))
print(torch.linalg.det(A), torch.linalg.inv(A))
print(torch.linalg.norm(A), torch.linalg.matrix_rank(A))
print(torch.linalg.svdvals(A), torch.linalg.eigvalsh(A))`,
  },
  {
    id: 'distributions', cat: 'maths', label: 'torch.distributions',
    code: `import torch

from torch.distributions import Normal, Categorical
n = Normal(torch.tensor(0.), torch.tensor(1.))
print(n, n.log_prob(torch.tensor(0.)), n.sample((3,)).shape)
c = Categorical(logits=torch.tensor([0., 1., 2.]))
print(c.probs, c.entropy(), c.log_prob(torch.tensor(2)))`,
  },
  {
    id: 'create_graph', cat: 'autograd', label: 'create_graph=True',
    code: `import torch

x = torch.tensor(2., requires_grad=True)
y = x ** 3
g, = torch.autograd.grad(y, x, create_graph=True)
print(g)            # 3x² = 12
h, = torch.autograd.grad(g, x)
print(h)            # 6x = 12`,
  },
  {
    id: 'rnn', cat: 'nn', label: 'nn.LSTM',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

lstm = nn.LSTM(input_size=4, hidden_size=8, num_layers=2, batch_first=True)
x = torch.zeros(3, 5, 4)      # batch, time, features
out, (h, c) = lstm(x)
print(out.shape, h.shape, c.shape)
print(lstm)
print(sum(p.numel() for p in lstm.parameters()))`,
  },
  {
    id: 'attention', cat: 'nn', label: 'TransformerEncoderLayer',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
x = torch.zeros(2, 5, 16)
print(layer(x).shape)
encoder = nn.TransformerEncoder(layer, num_layers=2)
print(encoder(x).shape, sum(p.numel() for p in encoder.parameters()))`,
  },
  {
    id: 'block', cat: 'nn', label: 'class Block(nn.Sequential)',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())

net = nn.Sequential(Block(3, 8), nn.MaxPool2d(2), Block(8, 16), nn.Flatten())
print(net)
print(net(torch.zeros(1, 3, 32, 32)).shape)`,
  },
  {
    id: 'scheduler', cat: 'train', label: 'lr_scheduler',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

w = nn.Parameter(torch.zeros(1))
opt = torch.optim.SGD([w], lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
for epoch in range(5):
    opt.step()
    sched.step()
    print(epoch, sched.get_last_lr())`,
  },
  {
    id: 'save_load', cat: 'scripts', label: 'torch.save / load',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from pathlib import Path
net = nn.Linear(2, 1)
out = Path('runs')
out.mkdir(exist_ok=True)
torch.save(net.state_dict(), out / 'net.pt')
print(os.path.exists(out / 'net.pt'), list(net.state_dict().keys()))
again = nn.Linear(2, 1)
print(again.load_state_dict(torch.load(out / 'net.pt')))
print(torch.equal(again.weight, net.weight))`,
  },
  {
    id: 'skeleton', cat: 'scripts', label: 'a training script',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse, random
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--epochs', type=int, default=3)
args = p.parse_args([])
random.seed(0); np.random.seed(0); torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(args, device in ('cuda', 'mps', 'cpu'))
model = nn.Linear(1, 1).to(device)
print(next(model.parameters()).shape, torch.get_num_threads() > 0)
with torch.inference_mode():
    print(model(torch.zeros(1, 1, device=device)).shape, torch.is_inference_mode_enabled())`,
  },
  // ---- the everyday APIs; each one was run on the real library
  {
    id: 'create', cat: 'tensors', label: 'zeros / arange / randn / like',
    code: `import torch

torch.manual_seed(0)
print(torch.zeros(2, 3).shape, torch.ones(2), torch.full((1, 2), 7.), torch.eye(2), torch.empty(2).shape)
print(torch.arange(0, 6, 2), torch.linspace(0, 1, 3), torch.logspace(0, 2, 3))
print(torch.rand(2).shape, torch.randn(2, 2).shape, torch.randint(0, 5, (3,)).shape, torch.randperm(4).shape)
x = torch.tensor([[1., 2.]])
print(torch.zeros_like(x), torch.ones_like(x), torch.full_like(x, 3.), torch.rand_like(x).shape, x.new_zeros(2))
print(torch.as_tensor([1, 2]), x.clone(), x.detach().requires_grad, x.tolist(), x[0, 1].item(), x.numpy())`,
  },
  {
    id: 'reduce', cat: 'tensors', label: 'sum / mean / max / argmax',
    code: `import torch

x = torch.tensor([[1., 5.], [3., 2.]])
print(x.sum(), x.sum(dim=0), x.mean(dim=1), x.mean(dim=1, keepdim=True), x.prod(), x.std(), x.var())
print(x.max(), x.max(dim=0), x.min(dim=1).values, x.argmax(), x.argmax(dim=1), x.argmin(dim=0))
print(x.cumsum(dim=0), x.median(), torch.topk(x.flatten(), 2), torch.sort(x, descending=True).indices)
print(torch.unique(torch.tensor([1, 1, 2])), torch.bincount(torch.tensor([0, 1, 1])), (x > 2).sum(), (x > 2).float().mean(), (x > 0).all(), (x > 4).any())`,
  },
  {
    id: 'joining', cat: 'tensors', label: 'cat / stack / chunk / split',
    code: `import torch

a = torch.arange(6).reshape(2, 3)
print(torch.cat([a, a], dim=0).shape, torch.cat([a, a], dim=1).shape, torch.stack([a, a]).shape)
print([p.shape for p in a.chunk(3, dim=1)], [p.shape for p in a.split([1, 2], dim=1)], [p.shape for p in a.unbind(0)])
print(a.flatten(), a.permute(1, 0).shape, a.transpose(0, 1).shape, a.T.shape, a.squeeze().shape, a.unsqueeze(0).shape)
print(a.repeat(2, 1).shape, a[:1].expand(3, -1).shape, a.flip(0), a.roll(1, 0), a.tril(), a.triu(), a.diag())
print(torch.where(a > 2, a, 0), a.clamp(1, 3), a.gather(1, torch.tensor([[0], [1]])), a.index_select(1, torch.tensor([0])), a.masked_fill(a > 2, -1), a.masked_select(a > 3), a.nonzero().shape)`,
  },
  {
    id: 'losses', cat: 'nn', label: 'nn.MSELoss / BCEWithLogitsLoss',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

p = torch.tensor([0.5, 2.0])
y = torch.tensor([1.0, 0.0])
print(nn.MSELoss()(p, y), nn.L1Loss()(p, y), nn.HuberLoss()(p, y), nn.SmoothL1Loss()(p, y))
print(nn.BCEWithLogitsLoss()(p, y), nn.BCELoss()(torch.sigmoid(p), y))
logits = torch.tensor([[2., 0.5], [0.1, 3.]])
t = torch.tensor([0, 1])
print(nn.CrossEntropyLoss()(logits, t), nn.NLLLoss()(F.log_softmax(logits, dim=1), t))
print(F.mse_loss(p, y), F.l1_loss(p, y), F.cross_entropy(logits, t), F.binary_cross_entropy_with_logits(p, y))
print(nn.MSELoss(reduction='none')(p, y), nn.MSELoss(reduction='sum')(p, y), nn.KLDivLoss(reduction='sum')(F.log_softmax(logits, 1), F.softmax(logits, 1)))`,
  },
  {
    id: 'activations', cat: 'nn', label: 'ReLU / LeakyReLU / F.softmax',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-2., 0., 2.])
for m in [nn.ReLU(), nn.LeakyReLU(0.1), nn.ELU(), nn.GELU(), nn.Sigmoid(), nn.Tanh(), nn.Softplus(), nn.Mish(), nn.Hardtanh()]:
    print(type(m).__name__, m(x))
print(F.relu(x), F.leaky_relu(x, 0.1), F.elu(x), F.gelu(x), F.softplus(x))
print(nn.Softmax(dim=0)(x), F.softmax(x, dim=0), nn.LogSoftmax(dim=0)(x), F.log_softmax(x, dim=0))
print(nn.Identity()(x), nn.Flatten()(torch.zeros(2, 3, 4)).shape, nn.Unflatten(1, (2, 2))(torch.zeros(1, 4)).shape)`,
  },
  {
    id: 'layers_cnn', cat: 'nn', label: 'Conv2d / BatchNorm2d / AdaptiveAvgPool2d',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.zeros(4, 3, 32, 32)
net = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, stride=2), nn.AvgPool2d(2), nn.Dropout2d(0.1),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
)
print(net(x).shape)
for l in net:
    x = l(x)
    print(type(l).__name__, tuple(x.shape))
print(nn.ConvTranspose2d(1, 1, 2, stride=2)(torch.zeros(1, 1, 4, 4)).shape, nn.Upsample(scale_factor=2)(torch.zeros(1, 1, 4, 4)).shape)
print(F.interpolate(torch.zeros(1, 1, 4, 4), size=(8, 8)).shape, F.max_pool2d(torch.zeros(1, 1, 4, 4), 2).shape, F.adaptive_avg_pool2d(torch.zeros(1, 1, 4, 4), 1).shape, F.pad(torch.zeros(1, 2), (1, 1)).shape)`,
  },
  {
    id: 'layers_seq', cat: 'nn', label: 'Embedding / GRU / LayerNorm',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

tokens = torch.tensor([[1, 2, 3]])
e = nn.Embedding(10, 4, padding_idx=0)
x = e(tokens)
print(x.shape, e.weight.shape)
out, h = nn.GRU(4, 5, batch_first=True)(x)
print(out.shape, h.shape)
out, (h, c) = nn.LSTM(4, 5, num_layers=2, bidirectional=True, batch_first=True)(x)
print(out.shape, h.shape, c.shape)
print(nn.LayerNorm(4)(x).shape, nn.GroupNorm(2, 4)(x.permute(0, 2, 1)).shape, nn.BatchNorm1d(4)(x.permute(0, 2, 1)).shape)
print(nn.MultiheadAttention(4, 2, batch_first=True)(x, x, x)[0].shape, F.cosine_similarity(x[0, 0], x[0, 0], dim=0).shape, F.one_hot(tokens, 4), F.normalize(torch.tensor([3., 4.]), dim=0))`,
  },
  {
    id: 'containers', cat: 'nn', label: 'ModuleList / parameters / state_dict',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
        self.head = nn.ModuleDict({'a': nn.Linear(2, 1)})
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer('mean', torch.zeros(2))
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head['a'](x - self.mean) * self.scale

n = Net()
print(n(torch.zeros(1, 2)).shape)
print([k for k, _ in n.named_parameters()])
print(sum(p.numel() for p in n.parameters()), sum(p.numel() for p in n.parameters() if p.requires_grad))
print([k for k, _ in n.named_children()], len(list(n.modules())), [k for k, _ in n.named_buffers()], list(n.state_dict())[:2])
n.blocks[0].requires_grad_(False)
print(sum(p.numel() for p in n.parameters() if p.requires_grad), n.training, n.eval().training)
print(n.get_submodule('head.a'), n.get_parameter('scale').shape, n.apply(lambda m: None) is n)`,
  },
  {
    id: 'init', cat: 'nn', label: 'nn.init',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
l = nn.Linear(4, 4)
nn.init.zeros_(l.bias)
nn.init.constant_(l.weight, 0.5)
print(l.weight[0], l.bias)
nn.init.ones_(l.bias)
nn.init.xavier_uniform_(l.weight)
nn.init.xavier_normal_(l.weight)
nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
nn.init.kaiming_uniform_(l.weight)
nn.init.normal_(l.weight, mean=0, std=0.01)
nn.init.uniform_(l.weight, -1, 1)
nn.init.orthogonal_(l.weight)
print(l.weight.shape, nn.init.calculate_gain('relu'), round(nn.init.calculate_gain('tanh'), 4))
def init(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
print(nn.Sequential(nn.Linear(1, 1)).apply(init)[0].bias)`,
  },
  {
    id: 'optimizers', cat: 'train', label: 'torch.optim',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

m = nn.Linear(1, 1)
for cls in [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW, torch.optim.RMSprop, torch.optim.Adagrad, torch.optim.Adadelta, torch.optim.Adamax, torch.optim.NAdam, torch.optim.RAdam]:
    o = cls(m.parameters(), lr=0.01)
    print(cls.__name__, o.param_groups[0]['lr'], len(o.param_groups[0]['params']))
o = torch.optim.SGD([{'params': [m.weight], 'lr': 0.1}, {'params': [m.bias]}], lr=0.01, momentum=0.9, weight_decay=1e-4)
print([g['lr'] for g in o.param_groups], o.defaults['momentum'], list(o.state_dict()))
print(torch.optim.Adam(m.parameters(), betas=(0.9, 0.99)).param_groups[0]['betas'])`,
  },
  {
    id: 'schedulers', cat: 'train', label: 'CosineAnnealingLR / OneCycleLR',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

def run(make, n=4):
    o = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=1.0)
    s = make(o)
    out = []
    for _ in range(n):
        o.step()
        s.step()
        out.append(round(s.get_last_lr()[0], 4))
    return out

S = torch.optim.lr_scheduler
print(run(lambda o: S.StepLR(o, step_size=2, gamma=0.1)))
print(run(lambda o: S.MultiStepLR(o, [1, 3], gamma=0.5)))
print(run(lambda o: S.ExponentialLR(o, 0.5)))
print(run(lambda o: S.CosineAnnealingLR(o, T_max=4)))
print(run(lambda o: S.LambdaLR(o, lambda e: 1 / (e + 1))))
print(run(lambda o: S.LinearLR(o, start_factor=0.5, total_iters=4)))
print(run(lambda o: S.OneCycleLR(o, max_lr=1.0, total_steps=5)))
o = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=1.0)
p = S.ReduceLROnPlateau(o, patience=0, factor=0.5)
for v in [1., 1., 1.]:
    p.step(v)
print(o.param_groups[0]['lr'])`,
  },
  {
    id: 'clip_grad', cat: 'train', label: 'clip_grad_norm_ / nn.utils',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

m = nn.Linear(2, 1)
(m(torch.ones(1, 2)) * 100).sum().backward()
print(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0) > 1)
print(round(float(torch.nn.utils.get_total_norm([p.grad for p in m.parameters()])), 4))
torch.nn.utils.clip_grad_value_(m.parameters(), 0.1)
print(m.weight.grad.abs().max() <= 0.1)
v = torch.nn.utils.parameters_to_vector(m.parameters())
print(v.shape)
torch.nn.utils.vector_to_parameters(torch.zeros(3), m.parameters())
print(m.weight, m.bias)`,
  },
  {
    id: 'dataset_class', cat: 'train', label: 'class Dataset / random_split',
    code: `import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset, random_split

class Squares(Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return torch.tensor([float(i)]), torch.tensor([float(i * i)])

d = Squares(10)
print(len(d), d[3])
train, val = random_split(d, [8, 2], generator=torch.Generator().manual_seed(0))
print(len(train), len(val), type(train).__name__)
print(len(Subset(d, [0, 1, 2])), len(ConcatDataset([d, d])), len(TensorDataset(torch.zeros(4), torch.ones(4))))
loader = DataLoader(train, batch_size=3, shuffle=True, drop_last=True)
print(len(loader))
for x, y in loader:
    print(x.shape, y.shape)
    break`,
  },
  {
    id: 'samplers', cat: 'train', label: 'WeightedRandomSampler / default_collate',
    code: `import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler, TensorDataset, WeightedRandomSampler, default_collate

print(list(SequentialSampler(range(3))), sorted(RandomSampler(range(3))))
print(list(BatchSampler(SequentialSampler(range(5)), 2, drop_last=False)))
print(sorted(SubsetRandomSampler([1, 3])))
w = WeightedRandomSampler([0., 0., 1.], 4, replacement=True)
print(list(w))
d = TensorDataset(torch.arange(4.))
print(len(DataLoader(d, batch_size=2, sampler=SequentialSampler(d))))
print(default_collate([(1, 'a'), (2, 'b')]))
print(default_collate([torch.zeros(2), torch.ones(2)]).shape)`,
  },
  {
    id: 'linalg_more', cat: 'maths', label: 'linalg.norm / svd / qr',
    code: `import torch

A = torch.tensor([[2., 1.], [1., 3.]])
print(torch.linalg.norm(A), torch.linalg.vector_norm(A[0]), torch.linalg.matrix_norm(A, ord=1), A.norm(), torch.dist(A[0], A[1]))
print(torch.linalg.det(A), torch.det(A), torch.linalg.inv(A), torch.inverse(A), A.trace(), torch.linalg.matrix_rank(A), torch.linalg.pinv(A))
print(torch.linalg.solve(A, torch.tensor([3., 5.])), torch.linalg.lstsq(A, torch.tensor([[3.], [5.]])).solution.shape)
print(torch.linalg.cholesky(A), torch.linalg.eigh(A).eigenvalues, torch.linalg.matrix_power(A, 2), torch.linalg.matrix_exp(torch.zeros(2, 2)))
U, S, Vh = torch.linalg.svd(A)
print(S, torch.allclose(U @ torch.diag(S) @ Vh, A), torch.linalg.qr(A).R.shape, torch.dot(A[0], A[1]), torch.mm(A, A).shape, torch.mv(A, A[0]), torch.outer(A[0], A[1]))`,
  },
  {
    id: 'math_ops', cat: 'maths', label: 'exp / log / clamp / isnan',
    code: `import torch

x = torch.tensor([1., 4., 9.])
print(x.sqrt(), x.exp(), x.log(), x.log2(), x.log10(), x.pow(2), x ** 2, x.abs(), x.neg(), x.reciprocal(), x.sign())
print(x.floor(), x.ceil(), x.round(), x.trunc(), x.frac(), x.clamp(min=2), x.clamp(max=5), torch.fmod(x, 2), x % 2, x // 2)
print(x.sin(), x.cos(), x.tan(), x.tanh(), x.sigmoid(), x.softmax(0), x.lerp(torch.zeros(3), 0.5))
n = torch.tensor([1., float('nan'), float('inf')])
print(n.isnan(), n.isinf(), n.isfinite(), n.nan_to_num(), torch.nansum(n[:2]), torch.nanmean(n[:2]))
print(torch.eq(x, x), x.ne(x), x.gt(2), x.lt(2), x.ge(4), x.le(4), torch.equal(x, x), torch.isclose(x, x), torch.allclose(x, x), torch.maximum(x, x - 1), torch.minimum(x, x - 1))
print(x.add(1), x.sub(1), x.mul(2), x.div(2), x.add_(1), x)`,
  },
  {
    id: 'autograd_more', cat: 'autograd', label: 'torch.autograd.grad / requires_grad_',
    code: `import torch

x = torch.tensor([2.], requires_grad=True)
y = x ** 3
print(torch.autograd.grad(y, x), y.requires_grad, y.grad_fn is not None, x.is_leaf, y.is_leaf)
z = torch.ones(1)
print(z.requires_grad, z.requires_grad_().requires_grad, z.detach().requires_grad)
with torch.enable_grad():
    print((x * 2).requires_grad)
with torch.set_grad_enabled(False):
    print((x * 2).requires_grad)
print(torch.is_grad_enabled())
@torch.no_grad()
def f(a):
    return a * 2
print(f(x).requires_grad)
with torch.inference_mode():
    print((x * 2).requires_grad)
v = torch.tensor([1., 2.], requires_grad=True)
(v ** 2).backward(torch.tensor([1., 1.]))
print(v.grad)
v.grad = None
print(v.grad, torch.autograd.gradcheck(lambda t: t * 2, torch.tensor([1.], dtype=torch.float64, requires_grad=True)))`,
  },
  {
    id: 'dist_more', cat: 'maths', label: 'Normal / Categorical / kl_divergence',
    code: `import torch
from torch.distributions import Bernoulli, Categorical, Normal, Uniform, kl_divergence

torch.manual_seed(0)
n = Normal(0., 1.)
print(n.sample((2,)).shape, n.rsample().shape, n.log_prob(torch.tensor(0.)), n.mean, n.stddev, n.entropy(), n.cdf(torch.tensor(0.)))
c = Categorical(probs=torch.tensor([0.25, 0.75]))
print(c.sample((3,)).shape, c.log_prob(torch.tensor(1)), c.probs, c.entropy())
print(Bernoulli(0.5).sample().shape, Bernoulli(probs=torch.tensor(0.5)).mean, Uniform(0., 2.).mean, Uniform(0., 2.).log_prob(torch.tensor(1.)))
print(kl_divergence(Normal(0., 1.), Normal(1., 1.)), kl_divergence(Categorical(probs=torch.tensor([.5, .5])), Categorical(probs=torch.tensor([.9, .1]))))`,
  },
]

export const PYTORCH_PLAYGROUND: PlayConfig = {
  id: 'pytorch',
  libs: [TORCH, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 4000,
  debounce: 600,
  cats: ['tensors', 'autograd', 'nn', 'train', 'maths', 'scripts'],
  entries: ENTRIES,
  prelude: NN,
  callApis: ['torch.', 'Tensor.', 'op:', 'F.'],
}
