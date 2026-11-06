#!/usr/bin/env python3
"""Trains the small CNN of the /cnn-live/ lab on CIFAR-10 and writes what the page and its test need.

The layout is the one CS231n's front page shows: (conv relu conv relu pool) three times and one dense layer, ten 3x3
filters per conv, so every channel of every layer fits on the screen. 6,440 parameters.

    ~/repo/minimind/.venv/bin/python scripts/train-cnn-live.py path/to/cifar-10-batches-py

Writes:
    src/data/cnnLive/weights.bin    little-endian float32, each layer's weight then bias, in PyTorch's order
    src/data/cnnLive/samples.png    60 test images, 6 per class in test-set order, not picked for looking good
    src/data/cnnLive/meta.json      accuracy measured on all 10,000 test images
    src/lib/cnnLive.cases.json      what real PyTorch computed on the samples, for cnnLive.test.ts
"""
import json
import pathlib
import pickle
import struct
import sys
import zlib

import numpy as np
import torch
import torch.nn as nn

EPOCHS = 150
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
here = pathlib.Path(__file__).parent
out_dir = here.parent / 'src/data/cnnLive'
out_dir.mkdir(parents=True, exist_ok=True)
data_dir = pathlib.Path(sys.argv[1])


def load(name):
    with open(data_dir / name, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data'].reshape(-1, 3, 32, 32), np.array(d[b'labels'])


parts = [load(f'data_batch_{i}') for i in range(1, 6)]
X = np.concatenate([p[0] for p in parts])
Y = np.concatenate([p[1] for p in parts])
X_test, Y_test = load('test_batch')
to_input = lambda a: a.astype(np.float32) / 255 - 0.5

torch.manual_seed(0)
np.random.seed(0)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


def block(a, b):
    return [nn.Conv2d(a, 10, 3, padding=1), nn.ReLU(), nn.Conv2d(10, 10, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]


net = nn.Sequential(*block(3, 10), *block(10, 10), *block(10, 10), nn.Flatten(), nn.Linear(160, 10)).to(device)
print('parameters', sum(p.numel() for p in net.parameters()))
x_all = torch.from_numpy(to_input(X)).to(device)
y_all = torch.from_numpy(Y).long().to(device)
x_test = torch.from_numpy(to_input(X_test)).to(device)
y_test = torch.from_numpy(Y_test).long().to(device)
opt = torch.optim.AdamW(net.parameters(), lr=4e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=EPOCHS * (len(X) // 128))
loss_fn = nn.CrossEntropyLoss()


def augment(b):
    """a flip and a shift of up to 4 pixels, per image"""
    n = len(b)
    pad = nn.functional.pad(b, (4, 4, 4, 4), mode='reflect')
    dy, dx = torch.randint(0, 9, (n,), device=device), torch.randint(0, 9, (n,), device=device)
    ys = (torch.arange(32, device=device)[None] + dy[:, None])[:, None, :, None].expand(n, 3, 32, 40)
    out = pad.gather(2, ys)
    xs = (torch.arange(32, device=device)[None] + dx[:, None])[:, None, None, :].expand(n, 3, 32, 32)
    out = out.gather(3, xs)
    flip = torch.rand(n, device=device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def accuracy():
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(x_test[i:i + 1000]).argmax(1) for i in range(0, 10000, 1000)])
    net.train()
    return pred


for epoch in range(EPOCHS):
    order = torch.randperm(len(X), device=device)
    total = 0.0
    for i in range(len(X) // 128):
        ix = order[i * 128:(i + 1) * 128]
        loss = loss_fn(net(augment(x_all[ix])), y_all[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item()
    if epoch % 10 == 9 or epoch == EPOCHS - 1:
        print(epoch + 1, round(total / (len(X) // 128), 4), round((accuracy() == y_test).float().mean().item(), 4), flush=True)

net = net.to('cpu').eval()
pred = torch.cat([net(torch.from_numpy(to_input(X_test[i:i + 1000]))).argmax(1) for i in range(0, 10000, 1000)]).numpy()
per_class = {c: round(float((pred[Y_test == k] == k).mean()), 4) for k, c in enumerate(CLASSES)}
accuracy_all = float((pred == Y_test).mean())
print('test accuracy', accuracy_all, per_class)

# ---- the weights
blob = b''
for p in net.parameters():
    blob += p.detach().numpy().astype('<f4').tobytes()
(out_dir / 'weights.bin').write_bytes(blob)

# ---- the samples: the first 6 of each class in the test set, in the order they come
picks = [i for k in range(10) for i in np.where(Y_test == k)[0][:6]]
picks.sort()
sprite = np.zeros((6 * 32, 10 * 32, 3), np.uint8)
for n, i in enumerate(picks):
    r, c = n // 10, n % 10
    sprite[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = X_test[i].transpose(1, 2, 0)


def png(a):
    h, w, _ = a.shape
    raw = b''.join(b'\0' + a[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')


(out_dir / 'samples.png').write_bytes(png(sprite))
(out_dir / 'meta.json').write_text(json.dumps({
    'classes': CLASSES,
    'labels': [int(Y_test[i]) for i in picks],
    'testIndex': [int(i) for i in picks],
    'testAccuracy': round(accuracy_all, 4),
    'perClass': per_class,
    'params': sum(p.numel() for p in net.parameters()),
    'epochs': EPOCHS,
}, indent=1) + '\n')

# ---- what real PyTorch computes, layer by layer, on the samples
cases = []
with torch.no_grad():
    for i in picks:
        x = torch.from_numpy(to_input(X_test[i:i + 1]))
        stats, a = [], x
        for layer in net:
            a = layer(a)
            if isinstance(layer, nn.Flatten):
                continue
            stats.append({'shape': list(a.shape[1:]), 'mean': round(a.mean().item(), 6), 'max': round(a.max().item(), 6), 'min': round(a.min().item(), 6), 'first': round(a.flatten()[0].item(), 6)})
        cases.append({'probs': [round(v, 6) for v in a.softmax(1)[0].tolist()], 'layers': stats})
(here.parent / 'src/lib/cnnLive.cases.json').write_text(json.dumps({'torch': torch.__version__, 'cases': cases}, separators=(',', ':')) + '\n')
print('wrote', len(blob), 'bytes of weights,', (out_dir / 'samples.png').stat().st_size, 'bytes of samples')
