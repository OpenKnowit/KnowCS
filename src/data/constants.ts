import type { AbNode, BayesData, CheatGroup, KernelPresetName, KnnPoint, PerceptronPoint, Vec2 } from '../types'

// --- Constants & Helper Data ---
export const INITIAL_MATRIX: number[][] = Array.from({ length: 4 }, (_, r) =>
  Array.from({ length: 4 }, (_, c) => r * 4 + c)
)

export const KERNEL_PRESETS: Record<KernelPresetName, number[][]> = {
  'Sobel-X': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  'Sobel-Y': [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
  'Laplacian': [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
  'Identity': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
}

export const BAYES_DATA: BayesData = {
  priors: { yes: 9 / 14, no: 5 / 14 },
  counts: {
    BP: { High: { yes: 2, no: 3 }, Normal: { yes: 3, no: 2 }, Low: { yes: 4, no: 0 } },
    Fever: { High: { yes: 2, no: 2 }, Mild: { yes: 4, no: 2 }, No: { yes: 3, no: 1 } },
    Diabetes: { Yes: { yes: 3, no: 4 }, No: { yes: 6, no: 1 } },
    Vomit: { Yes: { yes: 3, no: 3 }, No: { yes: 6, no: 2 } },
  },
  m_values: { BP: 3, Fever: 3, Diabetes: 2, Vomit: 2 },
}

// --- Alpha-Beta 剪枝：固定 3 层满二叉树（MAX→MIN→MAX→叶），坐标用于 SVG 布局 ---
export const AB_NODES: Record<string, AbNode> = {
  A: { id: 'A', type: 'max', parent: null, children: ['B', 'C'], x: 500, y: 60 },
  B: { id: 'B', type: 'min', parent: 'A', children: ['D', 'E'], x: 260, y: 170 },
  C: { id: 'C', type: 'min', parent: 'A', children: ['F', 'G'], x: 740, y: 170 },
  D: { id: 'D', type: 'max', parent: 'B', children: ['L1', 'L2'], x: 140, y: 280 },
  E: { id: 'E', type: 'max', parent: 'B', children: ['L3', 'L4'], x: 380, y: 280 },
  F: { id: 'F', type: 'max', parent: 'C', children: ['L5', 'L6'], x: 620, y: 280 },
  G: { id: 'G', type: 'max', parent: 'C', children: ['L7', 'L8'], x: 860, y: 280 },
  L1: { id: 'L1', type: 'leaf', parent: 'D', value: 3, x: 75, y: 400 },
  L2: { id: 'L2', type: 'leaf', parent: 'D', value: 5, x: 205, y: 400 },
  L3: { id: 'L3', type: 'leaf', parent: 'E', value: 2, x: 315, y: 400 },
  L4: { id: 'L4', type: 'leaf', parent: 'E', value: 4, x: 445, y: 400 },
  L5: { id: 'L5', type: 'leaf', parent: 'F', value: 6, x: 555, y: 400 },
  L6: { id: 'L6', type: 'leaf', parent: 'F', value: 1, x: 685, y: 400 },
  L7: { id: 'L7', type: 'leaf', parent: 'G', value: 0, x: 795, y: 400 },
  L8: { id: 'L8', type: 'leaf', parent: 'G', value: 7, x: 925, y: 400 },
}

export const AB_LEAF_IDS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'] as const

export const AB_PRESETS: Record<string, number[]> = {
  user: [3, 5, 2, 4, 6, 1, 0, 7], // 默认示例
  beta: [3, 5, 6, 2, 1, 4, 8, 9], // 演示单分支 Beta 剪枝
  alpha: [6, 7, 8, 9, 1, 2, 0, 4], // 演示子树级 Alpha 剪枝
}

// --- K-Means：三簇二维教学数据 ---
export const KMEANS_DATA: Vec2[] = [
  // 左下簇
  { x: 1.2, y: 1.5 }, { x: 1.8, y: 1.1 }, { x: 1.0, y: 2.2 }, { x: 2.1, y: 1.8 }, { x: 1.5, y: 1.0 }, { x: 2.3, y: 2.4 },
  // 右下簇
  { x: 7.2, y: 1.8 }, { x: 7.8, y: 1.2 }, { x: 8.1, y: 2.1 }, { x: 7.5, y: 2.6 }, { x: 8.4, y: 1.5 }, { x: 6.9, y: 1.1 },
  // 上中簇
  { x: 4.2, y: 7.5 }, { x: 4.8, y: 8.1 }, { x: 5.1, y: 7.2 }, { x: 4.5, y: 6.8 }, { x: 5.4, y: 8.0 }, { x: 3.9, y: 7.8 },
]

// 固定初始质心（教学可复现，刻意放偏以展示收敛过程）
export const KMEANS_INIT_CENTROIDS: Vec2[] = [
  { x: 2.0, y: 6.0 },
  { x: 5.0, y: 3.0 },
  { x: 6.0, y: 6.0 },
]

export const KNN_RAW_DATA: KnnPoint[] = [
  { h: 158, w: 58, s: 'M' }, { h: 158, w: 59, s: 'M' }, { h: 158, w: 63, s: 'M' },
  { h: 160, w: 59, s: 'M' }, { h: 160, w: 60, s: 'M' }, { h: 163, w: 60, s: 'M' },
  { h: 163, w: 61, s: 'M' }, { h: 160, w: 64, s: 'L' }, { h: 163, w: 64, s: 'L' },
  { h: 165, w: 61, s: 'L' }, { h: 165, w: 62, s: 'L' }, { h: 165, w: 65, s: 'L' },
  { h: 168, w: 62, s: 'L' }, { h: 168, w: 63, s: 'L' }, { h: 168, w: 66, s: 'L' },
  { h: 170, w: 63, s: 'L' }, { h: 170, w: 64, s: 'L' }, { h: 170, w: 68, s: 'L' },
]

// --- 感知机：线性可分的二维数据（label = ±1） ---
export const PERCEPTRON_DATA: PerceptronPoint[] = [
  { x: 1.0, y: 1.0, label: -1 }, { x: 1.5, y: 2.0, label: -1 }, { x: 2.0, y: 1.5, label: -1 },
  { x: 1.2, y: 2.6, label: -1 }, { x: 2.4, y: 2.2, label: -1 },
  { x: 4.0, y: 4.5, label: 1 }, { x: 4.6, y: 5.2, label: 1 }, { x: 5.2, y: 4.1, label: 1 },
  { x: 5.6, y: 5.6, label: 1 }, { x: 4.4, y: 5.8, label: 1 },
]

// --- PyTorch cheatsheet（L9/Lab9）：按主题分组，代码语言无关、描述走 i18n ---
export const PYTORCH_CHEATSHEET: CheatGroup[] = [
  {
    titleKey: 'pytorch.group.tensor',
    items: [
      { api: 'torch.tensor', code: "x = torch.tensor([1., 2.], requires_grad=True)", descKey: 'pytorch.api.tensor' },
      { api: 'torch.arange / zeros / ones', code: "torch.arange(3); torch.zeros(2, 3)", descKey: 'pytorch.api.create' },
      { api: 'x.reshape / view', code: "x = x.reshape(x.size(0), -1)  # flatten", descKey: 'pytorch.api.reshape' },
      { api: 'x.to(device)', code: "model = model.to('cuda')", descKey: 'pytorch.api.device' },
    ],
  },
  {
    titleKey: 'pytorch.group.autograd',
    items: [
      { api: 'requires_grad=True', code: "x = torch.tensor(2.0, requires_grad=True)", descKey: 'pytorch.api.requires_grad' },
      { api: 'loss.backward()', code: "loss.backward()  # 反向传播填充 .grad", descKey: 'pytorch.api.backward' },
      { api: 'x.grad', code: "print(x.grad)  # ∂loss/∂x", descKey: 'pytorch.api.grad' },
    ],
  },
  {
    titleKey: 'pytorch.group.nn',
    items: [
      { api: 'nn.Conv2d', code: "nn.Conv2d(3, 32, kernel_size=3)", descKey: 'pytorch.api.conv' },
      { api: 'nn.MaxPool2d', code: "nn.MaxPool2d(kernel_size=2, stride=2)", descKey: 'pytorch.api.pool' },
      { api: 'nn.Linear', code: "nn.Linear(1600, 128)", descKey: 'pytorch.api.linear' },
      { api: 'nn.ReLU / Dropout', code: "nn.ReLU(); nn.Dropout(0.5)", descKey: 'pytorch.api.relu' },
      { api: 'nn.CrossEntropyLoss', code: "criterion = nn.CrossEntropyLoss()", descKey: 'pytorch.api.loss' },
    ],
  },
  {
    titleKey: 'pytorch.group.train',
    items: [
      { api: 'optim.SGD', code: "opt = torch.optim.SGD(model.parameters(), lr=0.01)", descKey: 'pytorch.api.optim' },
      { api: 'optimizer.zero_grad()', code: "opt.zero_grad()  # 梯度清零（否则累加！）", descKey: 'pytorch.api.zero_grad' },
      { api: 'optimizer.step()', code: "opt.step()  # 用梯度更新参数", descKey: 'pytorch.api.step' },
    ],
  },
]

export const PYTORCH_TRAIN_LOOP = `for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)          # 前向
        loss = criterion(outputs, labels)

        optimizer.zero_grad()            # ① 清零
        loss.backward()                  # ② 反传
        optimizer.step()                 # ③ 更新
    print(f'Epoch {epoch+1}: loss={loss.item():.4f}')`

export const PYTORCH_CNN_CODE = `class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)`
