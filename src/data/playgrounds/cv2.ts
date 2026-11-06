// OpenCV 实验台：第 7 讲的颜色、滤波和几何变换；图片是代码里生成的数组
import { CV2 } from '../../lib/pyCv2'
import { MATPLOTLIB } from '../../lib/pyPlot'
import type { PlayConfig, PlayEntry } from './types'

const ENTRIES: PlayEntry[] = [
  {
    id: 'colours', cat: 'colour', label: 'cv2.cvtColor',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape, gray.shape, gray.dtype, img[0, 0], rgb[0, 0], gray[0, 0])
fig, (a, b) = plt.subplots(1, 2)
a.imshow(rgb); a.set_title('RGB')
b.imshow(gray, cmap='gray'); b.set_title('gray')`,
  },
  {
    id: 'threshold', cat: 'colour', label: 'cv2.threshold',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t1, fixed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
t2, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(t1, t2, np.unique(otsu))
fig, (a, b) = plt.subplots(1, 2)
a.imshow(fixed, cmap='gray'); a.set_title('fixed 100')
b.imshow(otsu, cmap='gray'); b.set_title(f'Otsu {t2}')`,
  },
  {
    id: 'histogram', cat: 'colour', label: 'calcHist / equalizeHist',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h = cv2.calcHist([gray], [0], None, [8], [0, 256]).ravel()
print(h, int(h.sum()), gray.size)
eq = cv2.equalizeHist(gray)
print(gray.min(), gray.max(), eq.min(), eq.max())
plt.bar(range(8), h)
plt.title('grey levels in 8 bins')`,
  },
  {
    id: 'filter', cat: 'filter', label: 'cv2.filter2D',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
edges = cv2.filter2D(gray.astype(np.float32), -1, k)
print(edges.shape, edges.dtype, edges.min(), edges.max())
plt.imshow(edges, cmap='gray')
plt.title('Sobel x by hand')`,
  },
  {
    id: 'blur', cat: 'filter', label: 'blur / median',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
noisy = img.copy()
noisy[::7, ::9] = 255
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for a, (name, out) in zip(ax, [('noisy', noisy), ('box', cv2.blur(noisy, (3, 3))), ('median', cv2.medianBlur(noisy, 3))]):
    a.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    a.set_title(name)
print(cv2.getGaussianKernel(3, 1).ravel().round(4))`,
  },
  {
    id: 'morphology', cat: 'filter', label: 'erode / dilate',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((20, 20), np.uint8)
img[5:15, 5:15] = 255
img[10, 2] = 255
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
for name, out in [('erode', cv2.erode(img, k)), ('dilate', cv2.dilate(img, k)), ('open', cv2.morphologyEx(img, cv2.MORPH_OPEN, k))]:
    print(name, cv2.countNonZero(out))
plt.imshow(cv2.morphologyEx(img, cv2.MORPH_OPEN, k), cmap='gray')
plt.title('open removes the lone dot')`,
  },
  {
    id: 'resize', cat: 'geometry', label: 'resize / flip',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
small = cv2.resize(img, (20, 15), interpolation=cv2.INTER_AREA)
big = cv2.resize(small, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
print(img.shape, small.shape, big.shape)
print(cv2.flip(img, 1).shape, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE).shape)
plt.imshow(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
plt.title('down, then up')`,
  },
  {
    id: 'warp', cat: 'geometry', label: 'warpAffine',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
print(M.round(3))
rot = cv2.warpAffine(img, M, (w, h))
move = cv2.warpAffine(img, np.float32([[1, 0, 15], [0, 1, 5]]), (w, h))
fig, (a, b) = plt.subplots(1, 2)
a.imshow(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)); a.set_title('rotate 30°')
b.imshow(cv2.cvtColor(move, cv2.COLOR_BGR2RGB)); b.set_title('translate')`,
  },
  // ---- the everyday APIs; each one was run on the real library
  {
    id: 'split_merge', cat: 'colour', label: 'cv2.split / cv2.merge',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

b, g, r = cv2.split(img)
print(b.shape, b[0, 0], g[0, 0], r[0, 0])
swapped = cv2.merge([r, g, b])
print(swapped.shape, swapped[0, 0])
print(np.array_equal(swapped, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
plt.imshow(swapped)
plt.show()`,
  },
  {
    id: 'hsv', cat: 'colour', label: 'COLOR_BGR2HSV / inRange',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsv[0, 0], hsv[20, 25])
mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
print(mask.shape, mask.dtype, np.unique(mask), cv2.countNonZero(mask))
only = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(only, cv2.COLOR_BGR2RGB))
plt.show()`,
  },
  {
    id: 'saturate', cat: 'arith', label: 'cv2.add vs +',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

a = np.array([[200, 100]], np.uint8)
b = np.array([[100, 100]], np.uint8)
print(cv2.add(a, b), a + b)
print(cv2.subtract(b, a), b - a)
print(cv2.absdiff(a, b))
print(cv2.multiply(a, 2), cv2.divide(a, 2))`,
  },
  {
    id: 'blend', cat: 'arith', label: 'addWeighted / convertScaleAbs',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

flat = np.full(img.shape, 255, np.uint8)
mixed = cv2.addWeighted(img, 0.7, flat, 0.3, 0)
print(img[0, 0], mixed[0, 0])
brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
print(brighter[0, 0], brighter.dtype)
fig, ax = plt.subplots(1, 2, figsize=(6, 2.4))
ax[0].imshow(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(brighter, cv2.COLOR_BGR2RGB))
plt.show()`,
  },
  {
    id: 'bitwise', cat: 'arith', label: 'bitwise_and / or / xor / not',
    code: `import numpy as np
import cv2

a = np.array([[0, 255, 255, 0]], np.uint8)
b = np.array([[0, 0, 255, 255]], np.uint8)
print(cv2.bitwise_and(a, b))
print(cv2.bitwise_or(a, b))
print(cv2.bitwise_xor(a, b))
print(cv2.bitwise_not(a))`,
  },
  {
    id: 'measure', cat: 'arith', label: 'mean / minMaxLoc / norm',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(cv2.mean(img))
print(cv2.minMaxLoc(gray))
print(cv2.sumElems(gray)[0], gray.sum())
print(cv2.norm(gray.astype(np.float32)), np.round(np.linalg.norm(gray.astype(np.float32)), 2))
stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
print(gray.min(), gray.max(), stretched.min(), stretched.max())`,
  },
  {
    id: 'adaptive', cat: 'colour', label: 'adaptiveThreshold / THRESH_OTSU',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ramp = np.tile(np.linspace(40, 220, 80, dtype=np.uint8), (60, 1))
ramp[20:40, 10:30] = ramp[20:40, 10:30] - 30
t, otsu = cv2.threshold(ramp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(t)
_, inv = cv2.threshold(ramp, 128, 255, cv2.THRESH_BINARY_INV)
adaptive = cv2.adaptiveThreshold(ramp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
print(otsu.mean().round(1), inv.mean().round(1), adaptive.shape)
fig, ax = plt.subplots(1, 3, figsize=(8, 2.2))
for a, im in zip(ax, [otsu, inv, adaptive]):
    a.imshow(im, cmap='gray')
plt.show()`,
  },
  {
    id: 'canny', cat: 'filter', label: 'cv2.Canny / GaussianBlur',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smooth = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(smooth, 50, 150)
print(edges.shape, edges.dtype, np.unique(edges), cv2.countNonZero(edges))
fig, ax = plt.subplots(1, 3, figsize=(8, 2.2))
for a, im in zip(ax, [gray, smooth, edges]):
    a.imshow(im, cmap='gray')
plt.show()`,
  },
  {
    id: 'gradients', cat: 'filter', label: 'Sobel / Scharr / Laplacian',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

f = gray.astype(np.float32)
gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
print(gx.shape, gx.min(), gx.max())
print(cv2.magnitude(gx, gy).max().round(2))
print(cv2.Scharr(f, cv2.CV_32F, 1, 0).max(), cv2.Laplacian(f, cv2.CV_32F).max())
plt.imshow(cv2.convertScaleAbs(gx), cmap='gray')
plt.show()`,
  },
  {
    id: 'separable', cat: 'filter', label: 'boxFilter / sepFilter2D',
    code: `import numpy as np
import cv2

a = np.zeros((5, 5), np.float32)
a[2, 2] = 9
print(cv2.boxFilter(a, -1, (3, 3)))
k = np.array([1., 2., 1.], np.float32) / 4
print(cv2.sepFilter2D(a, -1, k, k))`,
  },
  {
    id: 'morph_more', cat: 'filter', label: 'morphologyEx / getStructuringElement',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

a = np.zeros((30, 30), np.uint8)
a[8:22, 8:22] = 255
a[3, 3] = 255        # a speck of noise
a[14, 14] = 0        # a hole
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
print(cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
opened = cv2.morphologyEx(a, cv2.MORPH_OPEN, k)
closed = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k)
edge = cv2.morphologyEx(a, cv2.MORPH_GRADIENT, k)
print(opened[3, 3], closed[14, 14], cv2.countNonZero(edge))
fig, ax = plt.subplots(1, 4, figsize=(8, 2))
for x, im in zip(ax, [a, opened, closed, edge]):
    x.imshow(im, cmap='gray')
plt.show()`,
  },
  {
    id: 'lut', cat: 'filter', label: 'cv2.LUT (gamma)',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gamma = 0.5
table = np.array([255 * (i / 255) ** gamma for i in range(256)], np.uint8)
print(table[:5], table[-1])
out = cv2.LUT(gray, table)
print(gray[0, 0], out[0, 0])
plt.imshow(out, cmap='gray')
plt.show()`,
  },
  {
    id: 'perspective', cat: 'geometry', label: 'warpPerspective',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

src = np.float32([[0, 0], [79, 0], [79, 59], [0, 59]])
dst = np.float32([[10, 5], [70, 0], [79, 59], [0, 50]])
M = cv2.getPerspectiveTransform(src, dst)
print(M.shape, np.round(M @ np.array([0, 0, 1]), 3))
out = cv2.warpPerspective(img, M, (80, 60))
A = cv2.getAffineTransform(src[:3], dst[:3])
print(A.shape)
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.show()`,
  },
  {
    id: 'stitch', cat: 'geometry', label: 'copyMakeBorder / hconcat',
    code: `import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((60, 80, 3), np.uint8)   # BGR, as OpenCV stores it
img[:] = (200, 120, 40)
img[15:45, 20:60] = (40, 180, 240)
img[25:35, 30:50] = (30, 30, 30)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

framed = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
print(framed.shape)
a = np.array([[1, 2, 3]], np.uint8)
print(cv2.copyMakeBorder(a, 0, 0, 2, 2, cv2.BORDER_REPLICATE))
print(cv2.copyMakeBorder(a, 0, 0, 2, 2, cv2.BORDER_REFLECT))
print(cv2.hconcat([img, img]).shape, cv2.vconcat([img, img]).shape)
print(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE).shape, cv2.transpose(img).shape)
plt.imshow(cv2.cvtColor(framed, cv2.COLOR_BGR2RGB))
plt.show()`,
  },
]

export const CV2_PLAYGROUND: PlayConfig = {
  id: 'cv2',
  libs: [CV2, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 3000,
  debounce: 400,
  cats: ['colour', 'filter', 'geometry', 'arith'],
  entries: ENTRIES,
  prelude: '',
  callApis: [],
}
