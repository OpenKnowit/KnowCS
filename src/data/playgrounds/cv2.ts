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
]

export const CV2_PLAYGROUND: PlayConfig = {
  id: 'cv2',
  libs: [CV2, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 3000,
  debounce: 400,
  cats: ['colour', 'filter', 'geometry'],
  entries: ENTRIES,
  prelude: '',
  callApis: [],
}
