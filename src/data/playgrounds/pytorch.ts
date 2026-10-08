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
]

export const PYTORCH_PLAYGROUND: PlayConfig = {
  id: 'pytorch',
  libs: [TORCH, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 4000,
  debounce: 600,
  cats: ['tensors', 'autograd', 'nn', 'train'],
  entries: ENTRIES,
  prelude: NN,
  callApis: ['torch.', 'Tensor.', 'op:'],
}
