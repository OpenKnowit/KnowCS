import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { CV2 } from './pyCv2'

const run = (src: string) => runPython(src, { libs: [CV2], maxSize: 65536, timeBudget: 20_000 })

// Expected stdout and exception type from OpenCV 4.13 and NumPy 2.2 running the same source, on this Mac (arm64, carotene HAL)
// and on the x86-64 wheel under Rosetta (IPP, the build Colab uses): only cases where the two agree are kept.
const CASES: Record<string, { src: string; stdout: string; error?: string }> = {
  "gray_exact_rounding": {
    "src": "import cv2\nimport numpy as np\np = np.array([[[255, 255, 255], [1, 1, 1], [0, 0, 255], [255, 0, 0], [0, 255, 0], [127, 128, 129], [2, 3, 4], [100, 150, 200]]], dtype=np.uint8)\nprint(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY))\nprint(cv2.cvtColor(p, cv2.COLOR_RGB2GRAY))\nprint(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)[0, :3].tolist())\nprint(cv2.cvtColor(p[:, :2], cv2.COLOR_BGR2BGRA).tolist())",
    "stdout": "[[255   1  76  29 150 128   3 159]]\n[[255   1  29  76 150 128   3 141]]\n[[255, 255, 255], [1, 1, 1], [255, 0, 0]]\n[[[255, 255, 255, 255], [1, 1, 1, 255]]]\n"
  },
  "gray_back": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)[1, 1].tolist(), cv2.cvtColor(a, cv2.COLOR_GRAY2RGB).shape, cv2.cvtColor(a[:, :, None], cv2.COLOR_GRAY2RGB).shape)",
    "stdout": "[250, 250, 250] (4, 5, 3) (4, 5, 3)\n"
  },
  "hsv_lab_roundtrip": {
    "src": "import cv2\nimport numpy as np\np = np.array([[[10, 20, 200], [200, 10, 10], [10, 200, 10], [0, 0, 0], [255, 255, 255], [128, 128, 128], [30, 60, 90], [90, 60, 30]]], dtype=np.uint8)\nh = cv2.cvtColor(p, cv2.COLOR_RGB2HSV)\nprint(h)\nprint(cv2.cvtColor(h, cv2.COLOR_HSV2RGB))\nprint(cv2.cvtColor(p, cv2.COLOR_BGR2HSV))",
    "stdout": "[[[118 242 200]\n  [  0 242 200]\n  [ 60 242 200]\n  [  0   0   0]\n  [  0   0 255]\n  [  0   0 128]\n  [105 170  90]\n  [ 15 170  90]]]\n[[[ 10  23 200]\n  [200  10  10]\n  [ 10 200  10]\n  [  0   0   0]\n  [255 255 255]\n  [128 128 128]\n  [ 30  60  90]\n  [ 90  60  30]]]\n[[[  2 242 200]\n  [120 242 200]\n  [ 60 242 200]\n  [  0   0   0]\n  [  0   0 255]\n  [  0   0 128]\n  [ 15 170  90]\n  [105 170  90]]]\n"
  },
  "resize_all_interpolations": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.resize(a, (7, 3), interpolation=cv2.INTER_NEAREST))\nprint(cv2.resize(a, (3, 2), interpolation=cv2.INTER_AREA))\nprint(cv2.resize(a, None, fx=2, fy=1, interpolation=cv2.INTER_NEAREST))\nprint(cv2.resize(a, (5, 4)) is not a, cv2.resize(a, (5, 4)).tolist() == a.tolist())",
    "stdout": "[[ 10  10 200  30  30  90   7]\n [ 40  40 250  60  60   5 120]\n [ 70  70  80  90  90 120  33]]\n[[105  82  57]\n [ 88 117 107]]\n[[ 10  10 200 200  30  30  90  90   7   7]\n [ 40  40 250 250  60  60   5   5 120 120]\n [ 70  70  80  80  90  90 120 120  33  33]\n [  1   1 255 255 128 128  64  64 200 200]]\nTrue True\n"
  },
  "resize_integer_shrink": {
    "src": "import cv2\nimport numpy as np\nb = np.arange(0, 96, dtype=np.uint8).reshape(8, 12) * 2\nprint(cv2.resize(b, (6, 4)))\nprint(cv2.resize(b, (6, 4), interpolation=cv2.INTER_AREA))\nprint(cv2.resize(b, (4, 4), interpolation=cv2.INTER_AREA))",
    "stdout": "[[ 13  17  21  25  29  33]\n [ 61  65  69  73  77  81]\n [109 113 117 121 125 129]\n [157 161 165 169 173 177]]\n[[ 13  17  21  25  29  33]\n [ 61  65  69  73  77  81]\n [109 113 117 121 125 129]\n [157 161 165 169 173 177]]\n[[ 14  20  26  32]\n [ 62  68  74  80]\n [110 116 122 128]\n [158 164 170 176]]\n"
  },
  "resize_ties_on_two_by_two": {
    "src": "import cv2\nimport numpy as np\nprint(cv2.resize(np.array([[10, 21], [10, 21]], dtype=np.uint8), (1, 1)))\nprint(cv2.resize(np.array([[10, 21]], dtype=np.uint8), (1, 1), interpolation=cv2.INTER_AREA))",
    "stdout": "[[16]]\n[[16]]\n"
  },
  "resize_linear_gray": {
    "src": "import cv2\nimport numpy as np\np = np.array([[0, 255], [255, 0]], dtype=np.uint8)\nprint(cv2.resize(p, (4, 4), interpolation=cv2.INTER_NEAREST))\nq = np.array([[100, 100, 100], [100, 100, 100]], dtype=np.uint8)\nprint(cv2.resize(q, (5, 3)))\nprint(cv2.resize(q, (2, 5), interpolation=cv2.INTER_CUBIC))",
    "stdout": "[[  0   0 255 255]\n [  0   0 255 255]\n [255 255   0   0]\n [255 255   0   0]]\n[[100 100 100 100 100]\n [100 100 100 100 100]\n [100 100 100 100 100]]\n[[100 100]\n [100 100]\n [100 100]\n [100 100]\n [100 100]]\n"
  },
  "resize_errors_float_size": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\ncv2.resize(a, (1.5, 2))",
    "stdout": "",
    "error": "error"
  },
  "warp_affine": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nM = np.float32([[1, 0, 1.5], [0, 1, -0.25]])\nprint(cv2.warpAffine(a, M, (5, 4)))\nprint(cv2.warpAffine(a, M, (5, 4), flags=cv2.INTER_NEAREST))\nprint(cv2.warpAffine(a, M, (6, 3), borderMode=cv2.BORDER_REPLICATE))\nprint(cv2.warpAffine(a, M, (5, 4), borderMode=cv2.BORDER_CONSTANT, borderValue=9))",
    "stdout": "[[  0   9 115 125  53]\n [  0  24 128 138  51]\n [  0  26  88 112 103]\n [  0   0  96 144  72]]\n[[  0  10 200  30  90]\n [  0  40 250  60   5]\n [  0  70  80  90 120]\n [  0   1 255 128  64]]\n[[ 18  18 115 125  53  52]\n [ 48  48 128 138  51  66]\n [ 53  53  88 112 103  90]]\n[[  9  13 115 125  53]\n [  9  28 128 138  51]\n [  9  31  88 112 103]\n [  9   6  98 146  74]]\n"
  },
  "rotation": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nM = cv2.getRotationMatrix2D((2.5, 2), 30, 0.9)\nprint(np.round(M, 8))\nprint(cv2.warpAffine(a, M, (5, 4)))\nprint(cv2.warpAffine(a, np.round(M, 3), (5, 4), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR))",
    "stdout": "[[ 0.77942286  0.45       -0.34855716]\n [-0.45        0.77942286  1.56615427]]\n[[  0  10  65  42   0]\n [ 88 102  38  68  22]\n [ 48 214  82 104 168]\n [ 30  82 161  75  25]]\n[[ 37 125 193  37  60]\n [ 55  92 108  36  50]\n [126 171  99  93  75]\n [ 24  83  92 102  43]]\n"
  },
  "affine_points": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\np1 = np.float32([[0, 0], [4, 0], [0, 3]])\np2 = np.float32([[1, 0.5], [4, 1], [0.5, 3]])\nM = cv2.getAffineTransform(p1, p2)\nprint(np.round(M, 6))\nprint(cv2.warpAffine(a, M, (5, 4)))",
    "stdout": "[[ 0.75     -0.166667  1.      ]\n [ 0.125     0.833333  0.5     ]]\n[[  0   4  38   2   0]\n [  0  53 147  63   7]\n [  6  85 101  45  80]\n [  0 159 123  97  40]]\n"
  },
  "perspective": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\np1 = np.float32([[0, 0], [5, 0], [5, 4], [0, 4]])\np2 = np.float32([[0.5, 0.25], [4.5, 0], [5, 4], [0, 3.5]])\nM = cv2.getPerspectiveTransform(p1, p2)\nprint(np.round(M, 6))\nprint(cv2.warpPerspective(a, M, (5, 4), flags=cv2.INTER_NEAREST))",
    "stdout": "[[ 0.651899 -0.125     0.5     ]\n [-0.05      0.651899  0.25    ]\n [-0.032911 -0.045886  1.      ]]\n[[  0 200  30  90   7]\n [  0 250  60   5 120]\n [ 70  80  90 120  33]\n [  1 255 128  64 200]]\n"
  },
  "filter2d": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nk = np.array([[0.0, -1, 0], [-1, 5, -1], [0, -1, 0]])\nprint(cv2.filter2D(a, -1, k))\nprint(cv2.filter2D(a, cv2.CV_16S, k))\nprint(cv2.filter2D(a, cv2.CV_64F, k / 2))\nprint(cv2.filter2D(a, -1, np.ones((3, 3)) / 9))\nprint(cv2.filter2D(a, -1, k, borderType=cv2.BORDER_CONSTANT))\nprint(cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REPLICATE))\nprint(cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REFLECT))\nprint(cv2.filter2D(a, -1, np.array([[1, 2], [3, 4]]) / 10))\nprint(cv2.filter2D(a, -1, k, delta=10))",
    "stdout": "[[  0 255   0 255   0]\n [  0 255   0   0 255]\n [149   0  62 255   0]\n [  0 255 141   0 255]]\n[[-430  460 -260  403 -385]\n [-380  870  -75 -365  550]\n [ 149 -265   62  408 -395]\n [-645  986  141 -248  806]]\n[[-215.   230.  -130.   201.5 -192.5]\n [-190.   435.   -37.5 -182.5  275. ]\n [  74.5 -132.5   31.   204.  -197.5]\n [-322.5  493.    70.5 -124.   403. ]]\n[[166 104 106  55  50]\n [131  92 103  62  66]\n [142 108 117  91  81]\n [108  96 114  98  97]]\n[[  0 255   0 255   0]\n [  0 255   0   0 255]\n [229   0  62 255   0]\n [  0 255 231   0 255]]\n[[  0 255   0 255   0]\n [  0 255   0   0 255]\n [159   0  62 255   0]\n [  0 255 103   0 255]]\n[[  0 255   0 255   0]\n [  0 255   0   0 255]\n [159   0  62 255   0]\n [  0 255 103   0 255]]\n[[ 97 137 109  52  54]\n [113 153 125  41  60]\n [ 85 107  97  82  74]\n [ 99 125 154  97 118]]\n[[  0 255   0 255   0]\n [  0 255   0   0 255]\n [159   0  72 255   0]\n [  0 255 151   0 255]]\n"
  },
  "filter2d_wrap_refused": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nk = np.array([[0.0, -1, 0], [-1, 5, -1], [0, -1, 0]])\ncv2.filter2D(a, -1, k, borderType=cv2.BORDER_WRAP)",
    "stdout": "",
    "error": "error"
  },
  "blur_box": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.blur(a, (3, 3)))\nprint(cv2.blur(a, (5, 1)))\nprint(cv2.blur(a, (2, 2)))\nprint(cv2.blur(np.array([[0, 0, 255, 255]], dtype=np.uint8), (2, 1)))\nprint(cv2.blur(a, (3, 3), borderType=cv2.BORDER_REPLICATE))\nprint(cv2.boxFilter(a, cv2.CV_32S, (3, 3), normalize=False))",
    "stdout": "[[166 104 106  55  50]\n [131  92 103  62  66]\n [142 108 117  91  81]\n [108  96 114  98  97]]\n[[ 94 106  67  83  49]\n [132 121  95  88  50]\n [ 82  88  79  89  91]\n [153 141 130 142 117]]\n[[125 125 135  47  56]\n [125 125 135  47  56]\n [110 110 120  69  70]\n [102 102 139 101 105]]\n[[  1   1 128   0]]\n[[ 86  92 106  49  50]\n [ 86  92 103  62  59]\n [ 90 108 117  91  99]\n [ 82 112 132 114 124]]\n[[1490  940  950  497  447]\n [1180  830  925  555  590]\n [1281  974 1052  820  731]\n [ 971  864 1027  878  874]]\n"
  },
  "gaussian": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.GaussianBlur(a, (3, 3), 0))\nprint(cv2.GaussianBlur(a, (5, 5), 0))\nprint(cv2.GaussianBlur(a, (5, 3), 1.5))\nprint(cv2.GaussianBlur(a, (0, 0), 1.2))\nprint(cv2.GaussianBlur(a, (9, 9), 0))\nprint(cv2.GaussianBlur(a, (7, 7), 2, borderType=cv2.BORDER_REPLICATE))\nprint(np.round(cv2.getGaussianKernel(5, 1.0), 8).tolist(), cv2.getGaussianKernel(3, 0).tolist(), cv2.getGaussianKernel(9, 0).ravel().tolist()[:2])",
    "stdout": "[[125 130  91  51  56]\n [118 123  93  60  63]\n [106 117 107  86  87]\n [102 120 119 102 104]]\n[[124 116  91  65  57]\n [120 115  95  73  66]\n [113 113 104  90  84]\n [111 114 110  99  95]]\n[[128 117  90  73  52]\n [111 106  88  77  63]\n [121 116 104  98  86]\n [107 107 105 105 100]]\n[[119 114  93  71  65]\n [116 113  95  76  71]\n [113 112 102  89  86]\n [112 113 107  97  94]]\n[[110 106  95  85  80]\n [110 106  97  87  82]\n [110 107  99  92  87]\n [111 107 101  94  90]]\n[[ 63  75  76  72  62]\n [ 67  81  86  86  83]\n [ 70  88  98 103 106]\n [ 72  94 110 121 130]]\n[[0.05448868], [0.24420134], [0.40261995], [0.24420134], [0.05448868]] [[0.25], [0.5], [0.25]] [0.015625, 0.05078125]\n"
  },
  "gaussian_even_refused": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\ncv2.GaussianBlur(a, (2, 2), 0)",
    "stdout": "",
    "error": "error"
  },
  "median": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.medianBlur(a, 3))\nprint(cv2.medianBlur(a, 5))\nprint(cv2.medianBlur(a, 7))",
    "stdout": "[[ 40  40  90  30   7]\n [ 70  70  90  60  33]\n [ 70  80  90  90 120]\n [ 70  90 120 120 120]]\n[[ 40  60  60  60  30]\n [ 40  64  70  90  60]\n [ 40  64  80 120  90]\n [ 60  64  90 128 120]]\n[[ 30  33  33  33  33]\n [ 40  40  60  64  90]\n [ 40  60  64  90 120]\n [ 40  64  70 120 128]]\n"
  },
  "median_even_refused": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\ncv2.medianBlur(a, 2)",
    "stdout": "",
    "error": "error"
  },
  "sobel_laplacian": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.Sobel(a, cv2.CV_64F, 1, 0))\nprint(cv2.Sobel(a, cv2.CV_16S, 0, 1, ksize=5))\nprint(cv2.Sobel(a, -1, 1, 1))\nprint(cv2.Sobel(a, cv2.CV_64F, 2, 0, ksize=3))\nprint(cv2.Sobel(a, cv2.CV_16S, 1, 0, ksize=1))\nprint(cv2.Scharr(a, cv2.CV_16S, 0, 1))\nprint(cv2.Laplacian(a, cv2.CV_64F))\nprint(cv2.Laplacian(a, -1, ksize=3))\nprint(cv2.Laplacian(a, cv2.CV_16S, ksize=5))",
    "stdout": "[[   0.   80. -710.   74.    0.]\n [   0.   80. -560.   40.    0.]\n [   0.  187. -356.   18.    0.]\n [   0.  294. -302.   30.    0.]]\n[[    0     0     0     0     0]\n [-1018  -450   877  1878  2120]\n [ -596    90  1496  2454  2692]\n [    0     0     0     0     0]]\n[[  0   0   0   0   0]\n [  0   0 150   0   0]\n [  0 107  54  12   0]\n [  0   0   0   0   0]]\n[[ 1600. -1520.   730.    54.  -128.]\n [ 1240. -1160.   520.    80.  -120.]\n [  968.  -781.   238.   136.  -154.]\n [ 1056.  -762.   166.   166.  -196.]]\n[[   0   20 -110  -23    0]\n [   0   20 -245   60    0]\n [   0   20   40  -57    0]\n [   0  127 -191   72    0]]\n[[   0    0    0    0    0]\n [-120 -840  330  558  440]\n [-360  137  872 1034 1154]\n [   0    0    0    0    0]]\n[[ 440. -260.  290. -313.  392.]\n [ 420. -620.  135.  370. -430.]\n [ -79.  345.   28. -288.  428.]\n [ 646. -731.  -13.  312. -606.]]\n[[255   0 255   0   0]\n [255   0 255 255   0]\n [255   0 255  56  12]\n [255   0   0   0   0]]\n[[ -320 -3200    80  3408   576]\n [  796 -1976   450  3196   744]\n [ 1976 -1244  -728  1140  -728]\n [ 2040 -1736 -2276  -704 -2368]]\n"
  },
  "sobel_flow": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\ngx = cv2.Sobel(a, cv2.CV_16S, 1, 0)\ngy = cv2.Sobel(a, cv2.CV_16S, 0, 1)\nprint(cv2.convertScaleAbs(gx))\nprint(cv2.add(gx, gy))\nprint(cv2.absdiff(gx, gy))\nprint(cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0))\nprint(np.round(cv2.magnitude(cv2.Sobel(a, cv2.CV_64F, 1, 0), cv2.Sobel(a, cv2.CV_64F, 0, 1)), 4))",
    "stdout": "[[  0  80 255  74   0]\n [  0  80 255  40   0]\n [  0 187 255  18   0]\n [  0 255 255  30   0]]\n[[   0   80 -710   74    0]\n [-120  -40 -530  186  112]\n [ -68  226 -156  284  278]\n [   0  294 -302   30    0]]\n[[  0  80 710  74   0]\n [120 200 590 106 112]\n [ 68 148 556 248 278]\n [  0 294 302  30   0]]\n[[  0  40 128  37   0]\n [ 60 100 142  93  56]\n [ 34 113 228 136 128]\n [  0 128 128  15   0]]\n[[  0.      80.     710.      74.       0.    ]\n [120.     144.2221 560.803  151.3803 112.    ]\n [ 68.     191.0236 408.3332 266.6083 278.    ]\n [  0.     294.     302.      30.       0.    ]]\n"
  },
  "canny": {
    "src": "import cv2\nimport numpy as np\ne = np.zeros((8, 8), dtype=np.uint8)\ne[2:6, 2:6] = 200\ne[3, 3] = 90\nprint(cv2.Canny(e, 50, 150))\nprint(cv2.Canny(e, 200, 100))\nprint(cv2.Canny(e, 50, 150, L2gradient=True))\nprint(cv2.Canny(e, 100, 200, apertureSize=5))",
    "stdout": "[[  0   0   0   0   0   0   0   0]\n [  0   0   0 255 255   0   0   0]\n [  0   0 255   0   0 255   0   0]\n [  0 255   0   0   0 255   0   0]\n [  0 255   0   0   0 255   0   0]\n [  0   0 255 255 255 255   0   0]\n [  0   0   0   0   0   0   0   0]\n [  0   0   0   0   0   0   0   0]]\n[[  0   0   0   0   0   0   0   0]\n [  0   0   0 255 255   0   0   0]\n [  0   0 255   0   0 255   0   0]\n [  0 255   0   0   0 255   0   0]\n [  0 255   0   0   0 255   0   0]\n [  0   0 255 255 255 255   0   0]\n [  0   0   0   0   0   0   0   0]\n [  0   0   0   0   0   0   0   0]]\n[[  0   0   0   0   0   0   0   0]\n [  0   0   0 255 255   0   0   0]\n [  0   0 255   0   0 255   0   0]\n [  0 255   0   0   0 255   0   0]\n [  0 255   0   0   0 255   0   0]\n [  0   0 255 255 255 255   0   0]\n [  0   0   0   0   0   0   0   0]\n [  0   0   0   0   0   0   0   0]]\n[[  0   0   0   0   0   0   0   0]\n [  0   0   0   0   0   0   0   0]\n [  0   0 255 255 255 255   0   0]\n [  0   0 255   0   0 255   0   0]\n [  0   0 255   0   0 255   0   0]\n [  0   0 255 255 255 255   0   0]\n [  0   0   0   0   0   0   0   0]\n [  0   0   0   0   0   0   0   0]]\n"
  },
  "threshold": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nfor t in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]:\n    r, o = cv2.threshold(a, 90, 255, t)\n    print(r, o.tolist())\nprint(cv2.threshold(a, 90.7, 100.4, cv2.THRESH_BINARY))\nprint(cv2.threshold(a, 255, 255, cv2.THRESH_BINARY)[1].tolist(), cv2.threshold(a, -1, 7, cv2.THRESH_BINARY_INV)[1].tolist())\nr, o = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\nprint(r, o.tolist())",
    "stdout": "90.0 [[0, 255, 0, 0, 0], [0, 255, 0, 0, 255], [0, 0, 0, 255, 0], [0, 255, 255, 0, 255]]\n90.0 [[255, 0, 255, 255, 255], [255, 0, 255, 255, 0], [255, 255, 255, 0, 255], [255, 0, 0, 255, 0]]\n90.0 [[10, 90, 30, 90, 7], [40, 90, 60, 5, 90], [70, 80, 90, 90, 33], [1, 90, 90, 64, 90]]\n90.0 [[0, 200, 0, 0, 0], [0, 250, 0, 0, 120], [0, 0, 0, 120, 0], [0, 255, 128, 0, 200]]\n90.0 [[10, 0, 30, 90, 7], [40, 0, 60, 5, 0], [70, 80, 90, 0, 33], [1, 0, 0, 64, 0]]\n(90.0, array([[  0, 100,   0,   0,   0],\n       [  0, 100,   0,   0, 100],\n       [  0,   0,   0, 100,   0],\n       [  0, 100, 100,   0, 100]], dtype=uint8))\n[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]\n128.0 [[0, 255, 0, 0, 0], [0, 255, 0, 0, 0], [0, 0, 0, 0, 0], [0, 255, 0, 0, 255]]\n"
  },
  "threshold_otsu_on_colour_refused": {
    "src": "import cv2\nimport numpy as np\nc = np.array([[[10, 200, 30], [90, 7, 40]], [[250, 60, 5], [120, 70, 80]], [[90, 120, 33], [1, 255, 128]]], dtype=np.uint8)\ncv2.threshold(c, 0, 255, cv2.THRESH_OTSU)",
    "stdout": "",
    "error": "error"
  },
  "adaptive": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2))\nprint(cv2.adaptiveThreshold(a, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 1.5))\nprint(cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0))",
    "stdout": "[[  0 255   0 255   0]\n [  0 255   0   0 255]\n [  0   0   0 255   0]\n [  0 255   0   0 255]]\n[[200   0 200   0 200]\n [200   0 200 200   0]\n [200 200 200   0 200]\n [200   0   0 200   0]]\n[[  0 255   0 255   0]\n [  0 255   0   0 255]\n [  0   0   0 255   0]\n [  0 255   0   0 255]]\n"
  },
  "saturation": {
    "src": "import cv2\nimport numpy as np\np = np.array([[250, 5], [200, 100]], dtype=np.uint8)\nq = np.array([[10, 10], [100, 200]], dtype=np.uint8)\nprint(cv2.add(p, q))\nprint(cv2.subtract(q, p))\nprint(cv2.absdiff(p, q))\nprint(cv2.add(p, 10))\nprint(cv2.subtract(p, 100))\nprint(cv2.subtract(100, p))\nprint(cv2.multiply(p, q))\nprint(cv2.max(p, q), cv2.min(p, q))\nprint(cv2.convertScaleAbs(p, alpha=2))\nprint(cv2.convertScaleAbs(p, alpha=-1, beta=3))\nprint(cv2.convertScaleAbs(np.array([[1, 3, 5, 7]], dtype=np.uint8), alpha=0.5))\nprint(cv2.addWeighted(p, 1, q, 1, 0))\nprint(cv2.addWeighted(p, 0.5, q, 0.5, 0))\nprint(cv2.add(p, q, dtype=cv2.CV_16S))\nprint(cv2.filter2D(p, -1, np.array([[2.0]])))",
    "stdout": "[[255  15]\n [255 255]]\n[[  0   5]\n [  0 100]]\n[[240   5]\n [100 100]]\n[[255  15]\n [210 110]]\n[[150   0]\n [100   0]]\n[[ 0 95]\n [ 0  0]]\n[[255  50]\n [255 255]]\n[[250  10]\n [200 200]] [[ 10   5]\n [100 100]]\n[[255  10]\n [255 200]]\n[[247   2]\n [197  97]]\n[[0 2 2 4]]\n[[255  15]\n [255 255]]\n[[130   8]\n [150 150]]\n[[260  15]\n [300 300]]\n[[255  10]\n [255 200]]\n"
  },
  "scalars_per_channel": {
    "src": "import cv2\nimport numpy as np\nc = np.array([[[10, 200, 30], [90, 7, 40]], [[250, 60, 5], [120, 70, 80]], [[90, 120, 33], [1, 255, 128]]], dtype=np.uint8)\nprint(cv2.add(c, 100).tolist())\nprint(cv2.add(c, (100, 0, 0, 0)).tolist())\nprint(cv2.subtract(c, np.array([10.0])).tolist())",
    "stdout": "[[[110, 255, 130], [190, 107, 140]], [[255, 160, 105], [220, 170, 180]], [[190, 220, 133], [101, 255, 228]]]\n[[[110, 200, 30], [190, 7, 40]], [[255, 60, 5], [220, 70, 80]], [[190, 120, 33], [101, 255, 128]]]\n[[[0, 200, 30], [80, 7, 40]], [[240, 60, 5], [110, 70, 80]], [[80, 120, 33], [0, 255, 128]]]\n"
  },
  "sizes_must_match": {
    "src": "import cv2\nimport numpy as np\ncv2.add(np.zeros((3, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))",
    "stdout": "",
    "error": "error"
  },
  "bitwise_masks": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nm = np.array([[255, 0, 255, 0, 255], [0, 0, 0, 0, 0], [255, 255, 255, 255, 255], [0, 255, 0, 255, 0]], dtype=np.uint8)\nprint(cv2.bitwise_and(a, m))\nprint(cv2.bitwise_or(a, m))\nprint(cv2.bitwise_xor(a, m))\nprint(cv2.bitwise_not(a))\nprint(cv2.bitwise_and(a, a, mask=m))\nprint(cv2.add(a, a, mask=m))\nprint(cv2.inRange(a, 50, 150))\nprint(cv2.inRange(a, np.array([50]), np.array([150])))",
    "stdout": "[[ 10   0  30   0   7]\n [  0   0   0   0   0]\n [ 70  80  90 120  33]\n [  0 255   0  64   0]]\n[[255 200 255  90 255]\n [ 40 250  60   5 120]\n [255 255 255 255 255]\n [  1 255 128 255 200]]\n[[245 200 225  90 248]\n [ 40 250  60   5 120]\n [185 175 165 135 222]\n [  1   0 128 191 200]]\n[[245  55 225 165 248]\n [215   5 195 250 135]\n [185 175 165 135 222]\n [254   0 127 191  55]]\n[[ 10   0  30   0   7]\n [  0   0   0   0   0]\n [ 70  80  90 120  33]\n [  0 255   0  64   0]]\n[[ 20   0  60   0  14]\n [  0   0   0   0   0]\n [140 160 180 240  66]\n [  0 255   0 128   0]]\n[[  0   0   0 255   0]\n [  0   0 255   0 255]\n [255 255 255 255   0]\n [  0   0 255 255   0]]\n[[  0   0   0 255   0]\n [  0   0 255   0 255]\n [255 255 255 255   0]\n [  0   0 255 255   0]]\n"
  },
  "inrange_colour": {
    "src": "import cv2\nimport numpy as np\nc = np.array([[[10, 200, 30], [90, 7, 40]], [[250, 60, 5], [120, 70, 80]], [[90, 120, 33], [1, 255, 128]]], dtype=np.uint8)\nprint(cv2.inRange(c, (0, 0, 0), (100, 150, 100)))\nprint(cv2.inRange(c, np.array([50, 50, 0]), np.array([255, 255, 90])))",
    "stdout": "[[  0 255]\n [  0   0]\n [255   0]]\n[[  0   0]\n [255 255]\n [255   0]]\n"
  },
  "borders": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nfor b in [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101, cv2.BORDER_DEFAULT]:\n    print(cv2.copyMakeBorder(a[:2, :3], 1, 2, 2, 1, b, value=9))\n",
    "stdout": "[[  9   9   9   9   9   9]\n [  9   9  10 200  30   9]\n [  9   9  40 250  60   9]\n [  9   9   9   9   9   9]\n [  9   9   9   9   9   9]]\n[[ 10  10  10 200  30  30]\n [ 10  10  10 200  30  30]\n [ 40  40  40 250  60  60]\n [ 40  40  40 250  60  60]\n [ 40  40  40 250  60  60]]\n[[200  10  10 200  30  30]\n [200  10  10 200  30  30]\n [250  40  40 250  60  60]\n [250  40  40 250  60  60]\n [200  10  10 200  30  30]]\n[[250  60  40 250  60  40]\n [200  30  10 200  30  10]\n [250  60  40 250  60  40]\n [200  30  10 200  30  10]\n [250  60  40 250  60  40]]\n[[ 60 250  40 250  60 250]\n [ 30 200  10 200  30 200]\n [ 60 250  40 250  60 250]\n [ 30 200  10 200  30 200]\n [ 60 250  40 250  60 250]]\n[[ 60 250  40 250  60 250]\n [ 30 200  10 200  30 200]\n [ 60 250  40 250  60 250]\n [ 30 200  10 200  30 200]\n [ 60 250  40 250  60 250]]\n"
  },
  "border_colour": {
    "src": "import cv2\nimport numpy as np\nc = np.array([[[10, 200, 30], [90, 7, 40]], [[250, 60, 5], [120, 70, 80]], [[90, 120, 33], [1, 255, 128]]], dtype=np.uint8)\nprint(cv2.copyMakeBorder(c, 1, 0, 0, 1, cv2.BORDER_CONSTANT, value=(1, 2, 3)).tolist())\nprint(cv2.copyMakeBorder(c, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[7, 8, 9]).shape)",
    "stdout": "[[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[10, 200, 30], [90, 7, 40], [1, 2, 3]], [[250, 60, 5], [120, 70, 80], [1, 2, 3]], [[90, 120, 33], [1, 255, 128], [1, 2, 3]]]\n(5, 4, 3)\n"
  },
  "flip_rotate": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.flip(a, 0), cv2.flip(a, 1), cv2.flip(a, -1))\nprint(cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE))\nprint(cv2.rotate(a, cv2.ROTATE_180))\nprint(cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE))\nprint(cv2.transpose(a))",
    "stdout": "[[  1 255 128  64 200]\n [ 70  80  90 120  33]\n [ 40 250  60   5 120]\n [ 10 200  30  90   7]] [[  7  90  30 200  10]\n [120   5  60 250  40]\n [ 33 120  90  80  70]\n [200  64 128 255   1]] [[200  64 128 255   1]\n [ 33 120  90  80  70]\n [120   5  60 250  40]\n [  7  90  30 200  10]]\n[[  1  70  40  10]\n [255  80 250 200]\n [128  90  60  30]\n [ 64 120   5  90]\n [200  33 120   7]]\n[[200  64 128 255   1]\n [ 33 120  90  80  70]\n [120   5  60 250  40]\n [  7  90  30 200  10]]\n[[  7 120  33 200]\n [ 90   5 120  64]\n [ 30  60  90 128]\n [200 250  80 255]\n [ 10  40  70   1]]\n[[ 10  40  70   1]\n [200 250  80 255]\n [ 30  60  90 128]\n [ 90   5 120  64]\n [  7 120  33 200]]\n"
  },
  "split_merge": {
    "src": "import cv2\nimport numpy as np\nc = np.array([[[10, 200, 30], [90, 7, 40]], [[250, 60, 5], [120, 70, 80]], [[90, 120, 33], [1, 255, 128]]], dtype=np.uint8)\nb, g, r = cv2.split(c)\nprint(b, g, r)\nprint(cv2.merge([r, g, b]).tolist())\nprint(cv2.merge((b, g, r)).shape, cv2.hconcat([b, g]).shape, cv2.vconcat([b, g]).shape, cv2.hconcat([c, c]).shape)\nprint(cv2.mean(c), cv2.sumElems(c))\nprint(cv2.countNonZero(b), cv2.minMaxLoc(b), cv2.norm(b), cv2.norm(b, cv2.NORM_L1), cv2.norm(b, cv2.NORM_INF))",
    "stdout": "[[ 10  90]\n [250 120]\n [ 90   1]] [[200   7]\n [ 60  70]\n [120 255]] [[ 30  40]\n [  5  80]\n [ 33 128]]\n[[[30, 200, 10], [40, 7, 90]], [[5, 60, 250], [80, 70, 120]], [[33, 120, 90], [128, 255, 1]]]\n(3, 2, 3) (3, 4) (6, 2) (3, 4, 3)\n(93.5, 118.66666666666666, 52.666666666666664, 0.0) (561.0, 712.0, 316.0, 0.0)\n6 (1.0, 250.0, (1, 2), (0, 1)) 305.2883882495369 561.0 250.0\n"
  },
  "histogram": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nh = cv2.calcHist([a], [0], None, [4], [0, 256])\nprint(h, h.shape)\nprint(cv2.calcHist([a], [0], None, [5], [50, 200]).ravel().tolist())\nprint(cv2.calcHist([a], [0], np.uint8(a > 50) * 255, [2], [0, 256]).ravel().tolist())\nprint(cv2.equalizeHist(a))\nprint(cv2.equalizeHist(np.full((2, 2), 7, dtype=np.uint8)))\nprint(cv2.LUT(a, np.arange(256)[::-1].astype(np.uint8)))",
    "stdout": "[[8.]\n [7.]\n [1.]\n [4.]] (4, 1)\n[4.0, 2.0, 3.0, 0.0, 0.0]\n[8.0, 5.0]\n[[ 40 228  54 161  27]\n [ 81 242  94  13 188]\n [121 134 161 188  67]\n [  0 255 201 107 228]]\n[[7 7]\n [7 7]]\n[[245  55 225 165 248]\n [215   5 195 250 135]\n [185 175 165 135 222]\n [254   0 127 191  55]]\n"
  },
  "histogram_2d": {
    "src": "import cv2\nimport numpy as np\nc = np.array([[[10, 200, 30], [90, 7, 40]], [[250, 60, 5], [120, 70, 80]], [[90, 120, 33], [1, 255, 128]]], dtype=np.uint8)\nprint(cv2.calcHist([c], [0, 2], None, [2, 3], [0, 256, 0, 256]).tolist(), cv2.calcHist([c], [1], None, [2], [0, 256]).shape)",
    "stdout": "[[4.0, 1.0, 0.0], [1.0, 0.0, 0.0]] (2, 1)\n"
  },
  "normalize": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nprint(cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX))\nprint(np.round(cv2.normalize(a, None, 1, 0, cv2.NORM_L2, cv2.CV_64F), 6))\nprint(cv2.normalize(a, None, 255, 0, cv2.NORM_INF))\nd = np.zeros((4, 5), dtype=np.uint8)\ncv2.normalize(a, d, 10, 100, cv2.NORM_MINMAX)\nprint(d)",
    "stdout": "[[  9 200  29  89   6]\n [ 39 250  59   4 119]\n [ 69  79  89 119  32]\n [  0 255 128  63 200]]\n[[0.018516 0.370327 0.055549 0.166647 0.012961]\n [0.074065 0.462908 0.111098 0.009258 0.222196]\n [0.129614 0.148131 0.166647 0.222196 0.061104]\n [0.001852 0.472166 0.237009 0.118504 0.370327]]\n[[ 10 200  30  90   7]\n [ 40 250  60   5 120]\n [ 70  80  90 120  33]\n [  1 255 128  64 200]]\n[[ 13  81  20  42  12]\n [ 24  98  31  11  52]\n [ 34  38  42  52  21]\n [ 10 100  55  32  81]]\n"
  },
  "morphology": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nx = np.array([[0, 0, 0, 0, 0, 0], [0, 255, 255, 255, 0, 0], [0, 255, 0, 255, 0, 0], [0, 255, 255, 255, 0, 255], [0, 0, 0, 0, 0, 0]], dtype=np.uint8)\nk = np.ones((3, 3), np.uint8)\nprint(cv2.erode(x, k))\nprint(cv2.dilate(x, k))\nprint(cv2.erode(a, k), cv2.dilate(a, k, iterations=2))\nprint(cv2.erode(a, None, iterations=2))\nfor op in [cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT, cv2.MORPH_ERODE, cv2.MORPH_DILATE]:\n    print(cv2.morphologyEx(a, op, k).tolist())\nfor s in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:\n    print(cv2.getStructuringElement(s, (5, 3)).tolist(), cv2.getStructuringElement(s, (5, 5)).tolist())\nprint(cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), (1, 3)).tolist())\nh = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]])\nprint(cv2.morphologyEx(x, cv2.MORPH_HITMISS, h))",
    "stdout": "[[0 0 0 0 0 0]\n [0 0 0 0 0 0]\n [0 0 0 0 0 0]\n [0 0 0 0 0 0]\n [0 0 0 0 0 0]]\n[[255 255 255 255 255   0]\n [255 255 255 255 255   0]\n [255 255 255 255 255 255]\n [255 255 255 255 255 255]\n [255 255 255 255 255 255]]\n[[10 10  5  5  5]\n [10 10  5  5  5]\n [ 1  1  5  5  5]\n [ 1  1 64 33 33]] [[250 250 250 250 120]\n [255 255 255 255 200]\n [255 255 255 255 200]\n [255 255 255 255 200]]\n[[10  5  5  5  5]\n [ 1  1  1  5  5]\n [ 1  1  1  5  5]\n [ 1  1  1  5  5]]\n[[10, 10, 10, 5, 5], [10, 10, 10, 5, 5], [10, 64, 64, 64, 33], [1, 64, 64, 64, 33]]\n[[250, 250, 120, 120, 120], [250, 250, 120, 120, 120], [250, 250, 120, 120, 120], [255, 255, 200, 200, 200]]\n[[240, 240, 245, 115, 115], [240, 240, 245, 115, 115], [254, 254, 250, 195, 195], [254, 254, 191, 167, 167]]\n[[0, 190, 20, 85, 2], [30, 240, 50, 0, 115], [60, 16, 26, 56, 0], [0, 191, 64, 0, 167]]\n[[240, 50, 90, 30, 113], [210, 0, 60, 115, 0], [180, 170, 30, 0, 87], [254, 0, 72, 136, 0]]\n[[10, 10, 5, 5, 5], [10, 10, 5, 5, 5], [1, 1, 5, 5, 5], [1, 1, 64, 33, 33]]\n[[250, 250, 250, 120, 120], [250, 250, 250, 120, 120], [255, 255, 255, 200, 200], [255, 255, 255, 200, 200]]\n[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]\n[[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]] [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]\n[[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]] [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]\n[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 0, 0]]\n[[  0   0   0   0   0   0]\n [  0   0   0   0   0   0]\n [  0   0 255   0   0   0]\n [  0   0   0   0   0   0]\n [  0   0   0   0   0   0]]\n"
  },
  "float_images": {
    "src": "import cv2\nimport numpy as np\nf = np.array([[0.125, 0.5, 1.5], [2.0, 0.25, 0.75]], dtype=np.float64)\nprint(cv2.add(f, f))\nprint(cv2.subtract(f, 1.0))\nprint(cv2.resize(f, (3, 4), interpolation=cv2.INTER_NEAREST))\nprint(cv2.threshold(f, 0.5, 9.5, cv2.THRESH_BINARY))\nprint(cv2.threshold(f, 0.5, 9.5, cv2.THRESH_TRUNC)[1])\nprint(cv2.filter2D(f, -1, np.array([[0.5, 0.5]])))\nprint(cv2.flip(f, 1))\nprint(cv2.convertScaleAbs(f, alpha=100))\nprint(cv2.Sobel(f, -1, 1, 0))",
    "stdout": "[[0.25 1.   3.  ]\n [4.   0.5  1.5 ]]\n[[-0.875 -0.5    0.5  ]\n [ 1.    -0.75  -0.25 ]]\n[[0.125 0.5   1.5  ]\n [0.125 0.5   1.5  ]\n [2.    0.25  0.75 ]\n [2.    0.25  0.75 ]]\n(0.5, array([[0. , 0. , 9.5],\n       [9.5, 0. , 9.5]]))\n[[0.125 0.5   0.5  ]\n [0.5   0.25  0.5  ]]\n[[0.3125 0.3125 1.    ]\n [1.125  1.125  0.5   ]]\n[[1.5   0.5   0.125]\n [0.75  0.25  2.   ]]\n[[ 12  50 150]\n [200  25  75]]\n[[0.   0.25 0.  ]\n [0.   0.25 0.  ]]\n"
  },
  "dst_argument": {
    "src": "import cv2\nimport numpy as np\np = np.array([[200, 100]], dtype=np.uint8)\nd = np.zeros((1, 2), dtype=np.uint8)\nr = cv2.add(p, p, d)\nprint(d, r)\nprint(cv2.resize(p, None, None, 2, 1, cv2.INTER_NEAREST))",
    "stdout": "[[255 200]] [[255 200]]\n[[200 200 100 100]]\n"
  },
  "constants": {
    "src": "import cv2\nimport numpy as np\nprint(cv2.COLOR_BGR2GRAY, cv2.COLOR_RGB2GRAY, cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2RGB, cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB, cv2.COLOR_BGR2BGRA, cv2.COLOR_BGRA2BGR, cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101, cv2.BORDER_DEFAULT, cv2.BORDER_REFLECT101, cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV, cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.CV_8U, cv2.CV_8S, cv2.CV_16U, cv2.CV_16S, cv2.CV_32S, cv2.CV_32F, cv2.CV_64F, cv2.CV_8UC1, cv2.CV_8UC3, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.NORM_MINMAX, cv2.NORM_L1, cv2.NORM_L2, cv2.NORM_INF, cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE, cv2.MORPH_ERODE, cv2.MORPH_DILATE, cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT, cv2.WARP_INVERSE_MAP, cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED)",
    "stdout": "6 7 8 8 4 4 40 41 54 55 0 1 0 1 2 3 4 0 1 2 3 4 4 4 0 1 2 3 4 8 16 0 1 0 1 2 3 4 5 6 0 16 0 1 2 32 2 4 1 0 1 2 0 1 2 3 4 5 6 16 1 0 -1\n"
  },
  "np_types_for_images": {
    "src": "import cv2\nimport numpy as np\nprint(np.uint8(np.array([300, -1, 3.7])), np.uint8(3.7), np.uint8([1, 2]))\nprint(np.float32([[1.5, 2]]), np.float32(1.5), np.float64([1, 2]), np.int32([1.5]), np.int64(np.array([1, 2])))\nprint(np.zeros((1, 2), dtype=np.uint8), np.ones((1, 2), np.uint8))",
    "stdout": "[ 44 255   3] 3 [1 2]\n[[1.5 2. ]] 1.5 [1. 2.] [1] [1 2]\n[[0 0]] [[1 1]]\n"
  },
  "np_uint8_python_overflow": {
    "src": "import cv2\nimport numpy as np\nnp.uint8(300)",
    "stdout": "",
    "error": "OverflowError"
  },
  "results_are_uint8": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90, 7], [40, 250, 60, 5, 120], [70, 80, 90, 120, 33], [1, 255, 128, 64, 200]], dtype=np.uint8)\nc = np.array([[[10, 200, 30], [90, 7, 40]], [[250, 60, 5], [120, 70, 80]], [[90, 120, 33], [1, 255, 128]]], dtype=np.uint8)\nK = np.ones((3, 3), np.uint8)\nfor r in [cv2.add(a, a), cv2.subtract(a, a), cv2.absdiff(a, a), cv2.addWeighted(a, 0.5, a, 0.5, 0), cv2.multiply(a, a), cv2.divide(a, a), cv2.max(a, a), cv2.bitwise_and(a, a), cv2.bitwise_not(a), cv2.resize(a, (2, 2)), cv2.warpAffine(a, np.float32([[1, 0, 1], [0, 1, 0]]), (5, 4)), cv2.warpPerspective(a, np.eye(3), (5, 4)), cv2.filter2D(a, -1, K), cv2.blur(a, (3, 3)), cv2.GaussianBlur(a, (3, 3), 0), cv2.medianBlur(a, 3), cv2.threshold(a, 1, 255, 0)[1], cv2.adaptiveThreshold(a, 255, 0, 0, 3, 0), cv2.Canny(a, 1, 2), cv2.convertScaleAbs(a), cv2.convertScaleAbs(a * 1.5), cv2.copyMakeBorder(a, 1, 1, 1, 1, 0), cv2.flip(a, 0), cv2.rotate(a, 1), cv2.transpose(a), cv2.equalizeHist(a), cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX), cv2.inRange(a, 0, 5), cv2.erode(a, K), cv2.dilate(a, K), cv2.morphologyEx(a, 2, K), cv2.getStructuringElement(0, (3, 3)), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2RGB), cv2.cvtColor(c, cv2.COLOR_BGR2HSV), cv2.split(c)[0], cv2.merge([a, a]), cv2.hconcat([a, a]), cv2.Sobel(a, -1, 1, 0), cv2.Laplacian(a, -1)]:\n    print(r.dtype, end=' ')\nprint()\nx = cv2.add(a, a)\nprint(repr(x))\nprint(x + x)\nprint(x[0, 0], x.max(), x.sum(), (x * 2).dtype, (x / 2).dtype)\nprint(repr(cv2.flip(np.array([[1, 2]], dtype=np.uint8), 1)))",
    "stdout": "uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 uint8 \narray([[ 20, 255,  60, 180,  14],\n       [ 80, 255, 120,  10, 240],\n       [140, 160, 180, 240,  66],\n       [  2, 255, 255, 128, 255]], dtype=uint8)\n[[ 40 254 120 104  28]\n [160 254 240  20 224]\n [ 24  64 104 224 132]\n [  4 254 254   0 254]]\n20 255 2915 uint8 float64\narray([[2, 1]], dtype=uint8)\n"
  },
  "int64_reads_as_cv_32s": {
    "src": "import cv2\nimport numpy as np\na = np.array([[10, 200, 30, 90], [40, 300, 60, 5], [70, 80, -90, 120]])\nb = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])\nM = np.float64([[1, 0, 1], [0, 1, 0]])\nprint(cv2.add(a, a), cv2.subtract(a, b), cv2.multiply(a, b), cv2.divide(a, b), cv2.absdiff(a, b))\nprint(cv2.bitwise_and(a, b), cv2.bitwise_not(a), cv2.addWeighted(a, 0.5, b, 0.5, 0))\nprint(cv2.resize(a, (2, 2), interpolation=cv2.INTER_NEAREST), cv2.warpAffine(a, M, (4, 3), flags=cv2.INTER_NEAREST))\nprint(cv2.blur(a, (3, 3)), cv2.boxFilter(a, -1, (3, 3)), cv2.convertScaleAbs(a, alpha=2))\nprint(cv2.copyMakeBorder(a, 1, 1, 1, 1, cv2.BORDER_REFLECT), cv2.flip(a, 1), cv2.rotate(a, cv2.ROTATE_180), cv2.transpose(a))\nprint(cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX), cv2.inRange(a, 0, 100), cv2.merge([a, a]).shape, cv2.hconcat([a, a]).shape)\nprint(cv2.countNonZero(a), cv2.minMaxLoc(a), cv2.mean(a), cv2.sumElems(a), cv2.norm(a))\nprint(cv2.LUT(np.uint8(np.abs(a) % 255), np.arange(256)))\nprint(cv2.filter2D(np.uint8(np.abs(a) % 255), -1, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])), cv2.erode(np.uint8(np.abs(a) % 255), np.ones((3, 3), int)))\nK = np.ones((3, 3), np.uint8)\nfor name, f in [('cvtColor', lambda: cv2.cvtColor(np.stack([a, a, a], -1), cv2.COLOR_BGR2GRAY)), ('resize', lambda: cv2.resize(a, (3, 2))), ('resize_area', lambda: cv2.resize(a, (2, 2), interpolation=cv2.INTER_AREA)), ('resize_cubic', lambda: cv2.resize(a, (3, 2), interpolation=cv2.INTER_CUBIC)), ('warpAffine', lambda: cv2.warpAffine(a, M, (4, 3))), ('warpPerspective', lambda: cv2.warpPerspective(a, np.eye(3), (4, 3))), ('filter2D', lambda: cv2.filter2D(a, -1, K)), ('threshold', lambda: cv2.threshold(a, 50, 255, 0)), ('otsu', lambda: cv2.threshold(a, 0, 255, cv2.THRESH_OTSU)), ('adaptive', lambda: cv2.adaptiveThreshold(a, 255, 0, 0, 3, 0)), ('GaussianBlur', lambda: cv2.GaussianBlur(a, (3, 3), 0)), ('medianBlur', lambda: cv2.medianBlur(a, 3)), ('Sobel', lambda: cv2.Sobel(a, cv2.CV_64F, 1, 0)), ('Laplacian', lambda: cv2.Laplacian(a, cv2.CV_64F)), ('Scharr', lambda: cv2.Scharr(a, cv2.CV_64F, 1, 0)), ('Canny', lambda: cv2.Canny(a, 10, 20)), ('calcHist', lambda: cv2.calcHist([a], [0], None, [4], [0, 256])), ('equalizeHist', lambda: cv2.equalizeHist(a)), ('erode', lambda: cv2.erode(a, K)), ('dilate', lambda: cv2.dilate(a, K)), ('morphologyEx', lambda: cv2.morphologyEx(a, cv2.MORPH_OPEN, K)), ('magnitude', lambda: cv2.magnitude(a, a)), ('LUT', lambda: cv2.LUT(a, np.arange(256))), ('types', lambda: cv2.add(a, np.uint8(np.abs(a) % 255))), ('mask', lambda: cv2.bitwise_and(a, a, mask=np.ones((3, 4), int))), ('colour minMax', lambda: cv2.minMaxLoc(np.stack([a, a], -1))), ('bool', lambda: cv2.GaussianBlur(a > 0, (3, 3), 0))]:\n    try:\n        f()\n        print(name, 'ok')\n    except cv2.error as e:\n        print(name, 'refused')",
    "stdout": "[[  20  400   60  180]\n [  80  600  120   10]\n [ 140  160 -180  240]] [[   9  198   27   86]\n [  35  294   53   -3]\n [  61   70 -101  108]] [[  10  400   90  360]\n [ 200 1800  420   40]\n [ 630  800 -990 1440]] [[ 10 100  10  22]\n [  8  50   9   1]\n [  8   8  -8  10]] [[  9 198  27  86]\n [ 35 294  53   3]\n [ 61  70 101 108]]\n[[0 0 2 0]\n [0 4 4 0]\n [0 0 2 8]] [[ -11 -201  -31  -91]\n [ -41 -301  -61   -6]\n [ -71  -81   89 -121]] [[  6 101  16  47]\n [ 22 153  34   6]\n [ 40  45 -40  66]]\n[[10 30]\n [40 60]] [[  0  10 200  30]\n [  0  40 300  60]\n [  0  70  80 -90]]\n[[188 116 117  44]\n [142  78  88  24]\n [168  96  93  21]] [[188 116 117  44]\n [142  78  88  24]\n [168  96  93  21]] [[ 20 255  60 180]\n [ 80 255 120  10]\n [140 160 180 240]]\n[[ 10  10 200  30  90  90]\n [ 10  10 200  30  90  90]\n [ 40  40 300  60   5   5]\n [ 70  70  80 -90 120 120]\n [ 70  70  80 -90 120 120]] [[ 90  30 200  10]\n [  5  60 300  40]\n [120 -90  80  70]] [[120 -90  80  70]\n [  5  60 300  40]\n [ 90  30 200  10]] [[ 10  40  70]\n [200 300  80]\n [ 30  60 -90]\n [ 90   5 120]]\n[[ 65 190  78 118]\n [ 85 255  98  62]\n [105 111   0 137]] [[255   0 255 255]\n [255   0 255 255]\n [255 255   0   0]] (3, 4, 2) (3, 8)\n12 (-90.0, 300.0, (2, 2), (1, 1)) (76.25, 0.0, 0.0, 0.0) (915.0, 0.0, 0.0, 0.0) 422.0485754033533\n[[ 10 200  30  90]\n [ 40  45  60   5]\n [ 70  80  90 120]]\n[[255   0 255   0]\n [ 10 200   0 255]\n [  0   0   0   0]] [[10 10  5  5]\n [10 10  5  5]\n [40 40  5  5]]\ncvtColor refused\nresize refused\nresize_area refused\nresize_cubic refused\nwarpAffine refused\nwarpPerspective refused\nfilter2D refused\nthreshold refused\notsu refused\nadaptive refused\nGaussianBlur refused\nmedianBlur refused\nSobel refused\nLaplacian refused\nScharr refused\nCanny refused\ncalcHist refused\nequalizeHist refused\nerode refused\ndilate refused\nmorphologyEx refused\nmagnitude refused\nLUT refused\ntypes refused\nmask refused\ncolour minMax refused\nbool refused\n"
  },
  "except_cv2_error": {
    "src": "import cv2\nimport numpy as np\ntry:\n    cv2.cvtColor(np.zeros((3, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY)\nexcept cv2.error as e:\n    print(type(e).__name__, str(e)[:13])\ntry:\n    cv2.medianBlur(np.zeros((3, 3), dtype=np.uint8), 2)\nexcept Exception as e:\n    print(type(e).__name__)",
    "stdout": "error OpenCV(4.13.0\nerror\n"
  },
  "gray_of_gray_refused": {
    "src": "import cv2\nimport numpy as np\ncv2.cvtColor(np.zeros((3, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY)",
    "stdout": "",
    "error": "error"
  },
  "list_is_refused": {
    "src": "import cv2\nimport numpy as np\ncv2.resize([[1, 2]], (1, 1))",
    "stdout": "",
    "error": "error"
  },
  "unknown_keyword": {
    "src": "import cv2\nimport numpy as np\ncv2.resize(np.zeros((3, 3), dtype=np.uint8), (1, 1), foo=1)",
    "stdout": "",
    "error": "error"
  },
  "unknown_attribute": {
    "src": "import cv2\nimport numpy as np\ncv2.foo",
    "stdout": "",
    "error": "AttributeError"
  }
}

describe('cv2 matches OpenCV', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(name, () => {
      const r = run(c.src)
      expect(r.error?.type ?? null).toBe(c.error ?? null)
      expect(r.stdout).toBe(c.stdout)
    })
  }
})

const H = 'import cv2\nimport numpy as np\n'

describe('no files and no windows', () => {
  const cases: [string, string, RegExp][] = [
    ["cv2.imread('a.png')", 'FileNotFoundError', /no files.*NumPy/],
    ["cv2.imwrite('a.png', np.zeros((2, 2), dtype=np.uint8))", 'NotImplementedError', /no files.*NumPy/],
    ['cv2.VideoCapture(0)', 'FileNotFoundError', /no files.*NumPy/],
    ["cv2.imshow('w', np.zeros((2, 2), dtype=np.uint8))", 'NotImplementedError', /no windows/],
    ['cv2.waitKey(0)', 'NotImplementedError', /no windows/],
    ['cv2.destroyAllWindows()', 'NotImplementedError', /no windows/],
  ]
  for (const [src, type, message] of cases) {
    it(src, () => {
      const r = run(H + src)
      expect(r.error?.type).toBe(type)
      expect(r.error?.message).toMatch(message)
    })
  }
})

describe('options that are not implemented say so', () => {
  const cases: [string, RegExp][] = [
    ['cv2.resize(np.zeros((2, 2), dtype=np.uint8), (3, 3), interpolation=cv2.INTER_LANCZOS4)', /INTER_LANCZOS4|INTER_NEAREST/],
    ['cv2.warpAffine(np.zeros((2, 2), dtype=np.uint8), np.float32([[1, 0, 0], [0, 1, 0]]), (2, 2), flags=cv2.INTER_CUBIC)', /INTER_NEAREST and INTER_LINEAR/],
    ['cv2.threshold(np.zeros((2, 2), dtype=np.uint8), 0, 255, cv2.THRESH_TRIANGLE)', /THRESH_TRIANGLE/],
    ['cv2.calcHist([np.zeros((2, 2), dtype=np.uint8)], [0], None, [2], [0, 256], accumulate=True)', /accumulate/],
    ['cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), 44)', /conversion code 44/],
    ['cv2.bitwise_and(np.zeros((2, 2)), np.zeros((2, 2)))', /float image/],
  ]
  for (const [src, message] of cases) {
    it(src.slice(0, 50), () => {
      const r = run(H + src)
      expect(r.error?.type).toBe('NotImplementedError')
      expect(r.error?.message).toMatch(message)
    })
  }
})

describe('the step-through panel sees each call', () => {
  it('a filter records the window behind each output cell', () => {
    const r = run(`${H}a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)\nb = cv2.filter2D(a, -1, np.ones((3, 3)) / 9)\nc = cv2.add(a, a)\nd = cv2.flip(a, 1)\ne = cv2.threshold(a, 4, 255, cv2.THRESH_BINARY)\nf = cv2.resize(a, (2, 2))`)
    expect(r.error).toBeNull()
    const apis = r.calls.filter((c) => c.api.startsWith('cv2.')).map((c) => c.api)
    expect(apis).toEqual(['cv2.filter2D', 'cv2.add', 'cv2.flip', 'cv2.threshold', 'cv2.resize'])
    const filter = r.calls.find((c) => c.api === 'cv2.filter2D')!
    expect(filter.kind).toBe('group')
    expect(filter.groups![0]).toHaveLength(9)
    expect(filter.groups![4]).toEqual([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8]])
    expect(r.calls.find((c) => c.api === 'cv2.flip')!.source).toEqual([[0, 2], [0, 1], [0, 0], [0, 5], [0, 4], [0, 3], [0, 8], [0, 7], [0, 6]])
    expect(r.calls.find((c) => c.api === 'cv2.threshold')!.result!.values).toEqual([0, 0, 0, 0, 255, 255, 255, 255, 255])
  })
})
