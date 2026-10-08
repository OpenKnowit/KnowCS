import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { TORCH } from './pyTorch'

const run = (src: string) => runPython(src, { libs: [TORCH], maxSize: 2_000_000, timeBudget: 20_000 })

// Expected stdout produced by PyTorch 2.12 on the same source (deterministic snippets only).
const CASES: Record<string, { src: string; stdout: string }> = {
  "lecture_attrs": {
    "src": "import torch\ntensor = torch.tensor([1, 3, 5])\nprint(f\"Shape of tensor: {tensor.shape}\")\nprint(f\"Data type of tensor: {tensor.dtype}\")\nprint(f\"Device tensor is stored on: {tensor.device}\")\n",
    "stdout": "Shape of tensor: torch.Size([3])\nData type of tensor: torch.int64\nDevice tensor is stored on: cpu\n"
  },
  "lecture_ops": {
    "src": "import torch\ntensor = torch.ones(4, 4)\nprint(f\"First row: {tensor[0]}\")\nprint(f\"First column: {tensor[:, 0]}\")\nprint(f\"Last column: {tensor[..., -1]}\")\ntensor[:, 1] = 0\nprint(tensor)\nt1 = torch.cat([tensor, tensor, tensor], dim=1)\nprint(t1.shape)\ny1 = tensor @ tensor.T\nz1 = tensor * tensor\nprint(y1)\nprint(z1)\nagg = tensor.sum()\nprint(agg.item(), type(agg.item()))\ntensor.add_(5)\nprint(tensor)\n",
    "stdout": "First row: tensor([1., 1., 1., 1.])\nFirst column: tensor([1., 1., 1., 1.])\nLast column: tensor([1., 1., 1., 1.])\ntensor([[1., 0., 1., 1.],\n        [1., 0., 1., 1.],\n        [1., 0., 1., 1.],\n        [1., 0., 1., 1.]])\ntorch.Size([4, 12])\ntensor([[3., 3., 3., 3.],\n        [3., 3., 3., 3.],\n        [3., 3., 3., 3.],\n        [3., 3., 3., 3.]])\ntensor([[1., 0., 1., 1.],\n        [1., 0., 1., 1.],\n        [1., 0., 1., 1.],\n        [1., 0., 1., 1.]])\n12.0 <class 'float'>\ntensor([[6., 5., 6., 6.],\n        [6., 5., 6., 6.],\n        [6., 5., 6., 6.],\n        [6., 5., 6., 6.]])\n"
  },
  "bridge": {
    "src": "import torch\nimport numpy as np\nn = np.ones(5)\nt = torch.from_numpy(n)\nn += 1\nprint(t)\nt.add_(1)\nprint(n)\nt2 = torch.tensor(n)\nn += 1\nprint(t2)\nprint(torch.tensor([1, 2]).numpy())\n",
    "stdout": "tensor([2., 2., 2., 2., 2.], dtype=torch.float64)\n[3. 3. 3. 3. 3.]\ntensor([3., 3., 3., 3., 3.], dtype=torch.float64)\n[1 2]\n"
  },
  "dtypes": {
    "src": "import torch\na = torch.tensor([1, 2])\nprint((a / 2).dtype, (a * 2).dtype, (a * 2.5).dtype, (a + torch.tensor([0.5, 0.5])).dtype)\nprint(torch.arange(5), torch.arange(0, 1, 0.25))\nprint(torch.zeros(2, 3).dtype, torch.tensor(3.5).item())\nprint(torch.tensor([[1, 2], [3, 4]]).float().mean())\n",
    "stdout": "torch.float32 torch.int64 torch.float32 torch.float32\ntensor([0, 1, 2, 3, 4]) tensor([0.0000, 0.2500, 0.5000, 0.7500])\ntorch.float32 3.5\ntensor(2.5000)\n"
  },
  "autograd_scalar": {
    "src": "import torch\nx = torch.tensor(2.0, requires_grad=True)\ny = x ** 2 + 3 * x\nprint(y)\ny.backward()\nprint(x.grad)\ny2 = x ** 2\ny2.backward()\nprint(x.grad)\n",
    "stdout": "tensor(10., grad_fn=<AddBackward0>)\ntensor(7.)\ntensor(11.)\n"
  },
  "autograd_lecture": {
    "src": "import torch\nx = torch.ones(5)\ny = torch.zeros(3)\nw = torch.full((5, 3), 0.1, requires_grad=True)\nb = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)\nz = torch.matmul(x, w) + b\nloss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)\nprint(z)\nprint(loss)\nloss.backward()\nprint(w.grad)\nprint(b.grad)\nprint(z.requires_grad)\nwith torch.no_grad():\n    z = torch.matmul(x, w) + b\nprint(z.requires_grad)\nz_det = (torch.matmul(x, w) + b).detach()\nprint(z_det.requires_grad)\n",
    "stdout": "tensor([0.6000, 0.7000, 0.8000], grad_fn=<AddBackward0>)\ntensor(1.1039, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)\ntensor([[0.2152, 0.2227, 0.2300],\n        [0.2152, 0.2227, 0.2300],\n        [0.2152, 0.2227, 0.2300],\n        [0.2152, 0.2227, 0.2300],\n        [0.2152, 0.2227, 0.2300]])\ntensor([0.2152, 0.2227, 0.2300])\nTrue\nFalse\nFalse\n"
  },
  "mnist_cnn": {
    "src": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass MNIST_CNN(nn.Module):\n    def __init__(self):\n        super(MNIST_CNN, self).__init__()\n        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)\n        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)\n        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)\n        self.dropout1 = nn.Dropout(0.25)\n        self.fc1 = nn.Linear(64 * 14 * 14, 128)\n        self.dropout2 = nn.Dropout(0.5)\n        self.fc2 = nn.Linear(128, 10)\n\n    def forward(self, x):\n        x = F.relu(self.conv1(x))\n        x = F.relu(self.conv2(x))\n        x = self.pool(x)\n        x = self.dropout1(x)\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.dropout2(x)\n        x = self.fc2(x)\n        return x\n\nmodel = MNIST_CNN()\nprint(model)\nprint(sum(p.numel() for p in model.parameters()))\nfor name, p in model.named_parameters():\n    print(name, tuple(p.shape))\nout = model(torch.zeros(2, 1, 28, 28))\nprint(out.shape)\n",
    "stdout": "MNIST_CNN(\n  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n  (dropout1): Dropout(p=0.25, inplace=False)\n  (fc1): Linear(in_features=12544, out_features=128, bias=True)\n  (dropout2): Dropout(p=0.5, inplace=False)\n  (fc2): Linear(in_features=128, out_features=10, bias=True)\n)\n1625866\nconv1.weight (32, 1, 3, 3)\nconv1.bias (32,)\nconv2.weight (64, 32, 3, 3)\nconv2.bias (64,)\nfc1.weight (128, 12544)\nfc1.bias (128,)\nfc2.weight (10, 128)\nfc2.bias (10,)\ntorch.Size([2, 10])\n"
  },
  "losses": {
    "src": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nlogits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]])\ntarget = torch.tensor([0, 1])\nprint(F.cross_entropy(logits, target))\nprint(nn.CrossEntropyLoss()(logits, target))\nprint(F.softmax(logits, dim=1))\n_, predicted = torch.max(logits, 1)\nprint(predicted, (predicted == target).sum().item())\nprint(torch.argmax(F.softmax(logits, dim=1), 1))\nprint(F.mse_loss(torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 5.])))\nprint(nn.BCELoss()(torch.tensor([0.9, 0.2]), torch.tensor([1., 0.])))\n",
    "stdout": "tensor(0.3143)\ntensor(0.3143)\ntensor([[0.6590, 0.2424, 0.0986],\n        [0.1095, 0.8093, 0.0811]])\ntensor([0, 1]) 2\ntensor([0, 1])\ntensor(1.3333)\ntensor(0.1643)\n"
  },
  "errors_view": {
    "src": "import torch\nx = torch.ones(2, 3)\nprint(x.T.reshape(6))\nprint(x.T.contiguous().view(6))\n",
    "stdout": "tensor([1., 1., 1., 1., 1., 1.])\ntensor([1., 1., 1., 1., 1., 1.])\n"
  },
  "sgd_step": {
    "src": "import torch\nw = torch.tensor([1.0, -2.0], requires_grad=True)\nopt = torch.optim.SGD([w], lr=0.1)\nloss = (w ** 2).sum()\nloss.backward()\nprint(w.grad)\nopt.step()\nprint(w)\nopt.zero_grad()\nprint(w.grad)\n",
    "stdout": "tensor([ 2., -4.])\ntensor([ 0.8000, -1.6000], requires_grad=True)\nNone\n"
  },
  "linreg_manual": {
    "src": "import torch\nX = torch.tensor([[1.0], [2.0], [3.0], [4.0]])\ny = torch.tensor([[3.0], [5.0], [7.0], [9.0]])\nw = torch.zeros(1, 1, requires_grad=True)\nb = torch.zeros(1, requires_grad=True)\nfor epoch in range(200):\n    y_pred = X @ w + b\n    loss = ((y_pred - y) ** 2).mean()\n    loss.backward()\n    with torch.no_grad():\n        w -= 0.05 * w.grad\n        b -= 0.05 * b.grad\n        w.grad.zero_()\n        b.grad.zero_()\nprint(round(w.item(), 3), round(b.item(), 3), round(loss.item(), 5))\n",
    "stdout": "2.005 0.986 3e-05\n"
  }
}

describe('torch sandbox matches PyTorch', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(name, () => {
      const r = run(c.src)
      expect(r.error).toBeNull()
      expect(r.stdout).toBe(c.stdout)
    })
  }
})

const CNN = `import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4 * 4 * 4, 3)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
model = Net()
`

describe('torch events', () => {
  it('a forward pass records each layer with shapes, params and the size formula', () => {
    const r = run(`${CNN}out = model(torch.zeros(2, 1, 8, 8))`)
    expect(r.error).toBeNull()
    const flow = r.events.find((e) => e.type === 'flow')
    expect(flow && flow.type === 'flow' && flow.rows.map((x) => [x.name, x.input, x.output, x.params])).toEqual([
      ['conv1', [2, 1, 8, 8], [2, 4, 8, 8], 40],
      ['F.relu(self.conv1(x))', [2, 4, 8, 8], [2, 4, 8, 8], 0],
      ['pool', [2, 4, 8, 8], [2, 4, 4, 4], 0],
      ['x.view(x.size(0), -1)', [2, 4, 4, 4], [2, 64], 0],
      ['fc', [2, 64], [2, 3], 195],
    ])
    expect(flow && flow.type === 'flow' && flow.rows[0].note).toMatch('⌊(8 + 2·1 − 3) / 1⌋ + 1 = 8')
    expect(flow && flow.type === 'flow' && flow.total).toBe(235)
  })

  it('backward() records the graph with parameter names and gradients', () => {
    const r = run(`${CNN}loss = nn.CrossEntropyLoss()(model(torch.ones(1, 1, 8, 8)), torch.tensor([2]))\nloss.backward()`)
    expect(r.error).toBeNull()
    const g = r.events.find((e) => e.type === 'graph')
    expect(g && g.type === 'graph' && g.nodes.map((n) => n.label)).toEqual(expect.arrayContaining(['loss', 'fc.weight', 'fc.bias', 'conv1.weight', 'conv1.bias']))
    expect(g && g.type === 'graph' && g.nodes.find((n) => n.label === 'fc.bias')?.grad).toMatch(/^\[/)
  })

  it('teaching notes: gradients accumulating, dropout left on, softmax before CrossEntropyLoss', () => {
    const notes = (src: string) => run(src).events.filter((e) => e.type === 'note').map((e) => (e as { key: string }).key)
    expect(notes('import torch\nw = torch.tensor(1.0, requires_grad=True)\n(w * 2).backward()\n(w * 2).backward()')).toEqual(['grad_accumulate'])
    expect(notes('import torch\nimport torch.nn as nn\nd = nn.Dropout(0.5)\nwith torch.no_grad():\n    y = d(torch.ones(4))')).toEqual(['dropout_eval'])
    expect(notes('import torch\nimport torch.nn.functional as F\np = F.softmax(torch.tensor([[2., 1.]]), dim=1)\nF.cross_entropy(p, torch.tensor([0]))')).toEqual(['softmax_ce'])
  })
})

describe('torch errors students meet', () => {
  const msg = (src: string) => run(`import torch\nimport torch.nn as nn\n${src}`).error?.message
  it('wrong flatten size into Linear', () => expect(msg('nn.Linear(6272, 128)(torch.zeros(1, 12544))')).toBe('mat1 and mat2 shapes cannot be multiplied (1x12544 and 6272x128)'))
  it('view after transpose', () => expect(msg('torch.ones(2, 3).T.view(6)')).toMatch("view size is not compatible with input tensor's size and stride"))
  it('in-place update of a leaf outside no_grad', () => expect(msg('w = torch.ones(2, requires_grad=True)\nw -= 0.1')).toBe('a leaf Variable that requires grad is being used in an in-place operation.'))
  it('numpy() on a tensor that requires grad', () => expect(msg('torch.ones(2, requires_grad=True).numpy()')).toMatch("Can't call numpy() on Tensor that requires grad"))
  it('module attribute before super().__init__()', () => expect(msg('class B(nn.Module):\n    def __init__(self):\n        self.fc = nn.Linear(2, 2)\nB()')).toBe('cannot assign module before Module.__init__() call'))
  it('conv channels', () => expect(msg('nn.Conv2d(1, 8, 3)(torch.zeros(1, 3, 8, 8))')).toBe('Given groups=1, weight of size [8, 1, 3, 3], expected input[1, 3, 8, 8] to have 1 channels, but got 3 channels instead'))
})

describe('training', () => {
  it('an nn.Module learns XOR with Adam, batches from a DataLoader', () => {
    const r = run(`import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 0])
loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=True)
model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
opt = torch.optim.Adam(model.parameters(), lr=0.05)
lossfunc = nn.CrossEntropyLoss()
losses = []
for epoch in range(150):
    for xb, yb in loader:
        opt.zero_grad()
        loss = lossfunc(model(xb), yb)
        loss.backward()
        opt.step()
    losses.append(loss.item())
with torch.no_grad():
    pred = model(X).argmax(dim=1)
print(pred.tolist(), losses[-1] < losses[0] / 10)`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('[0, 1, 1, 0] True\n')
  })
})
