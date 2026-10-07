/**
 * Formula sheet (/formulas/): every formula the exams use, in lecture order. LaTeX is language-neutral;
 * names and notes live in the locales under formulas.items.<id>.{name,note}.
 */
export interface Formula {
  id: string
  lec: number
  tex: string
  link?: { kind: 'watch' | 'module'; id: string }
}

const W = (id: string) => ({ kind: 'watch' as const, id })
const M = (id: string) => ({ kind: 'module' as const, id })

export const FORMULAS: Formula[] = [
  { id: 'broadcast', lec: 1, tex: '(\\ldots, a) \\oplus (\\ldots, b):\\ a = b \\ \\text{or}\\ a = 1 \\ \\text{or}\\ b = 1', link: W('broadcast') },
  { id: 'pairwise', lec: 1, tex: '\\lVert x - y \\rVert^2 = \\lVert x \\rVert^2 + \\lVert y \\rVert^2 - 2\\,x\\cdot y', link: W('broadcast') },
  { id: 'bayes', lec: 2, tex: 'P(B \\mid E) = \\dfrac{P(E \\mid B)\\,P(B)}{P(E)}', link: W('bayes') },
  { id: 'total', lec: 2, tex: 'P(E) = \\sum_j P(E \\mid B_j)\\,P(B_j)', link: W('bayes') },
  { id: 'naive', lec: 2, tex: 'B_{NB} = \\arg\\max_{B_i}\\ P(B_i)\\prod_j P(e_j \\mid B_i)', link: M('naive-bayes') },
  { id: 'laplace', lec: 2, tex: 'P(e_j = v \\mid B) = \\dfrac{\\text{count}(e_j = v, B) + \\alpha}{\\text{count}(B) + m\\,\\alpha}', link: W('bayes') },
  { id: 'gauss', lec: 2, tex: 'f(x) = \\dfrac{1}{\\sqrt{2\\pi}\\,\\sigma}\\exp\\!\\left(-\\dfrac{(x - \\mu)^2}{2\\sigma^2}\\right)', link: W('gaussian') },
  { id: 'std', lec: 2, tex: '\\sigma = \\sqrt{\\dfrac{1}{n - 1}\\sum_{i=1}^{n}(x_i - \\mu)^2}', link: M('gaussian-nb') },
  { id: 'euclid', lec: 3, tex: 'd(x, y) = \\sqrt{\\textstyle\\sum_i (x_i - y_i)^2}', link: W('knn') },
  { id: 'zscore', lec: 3, tex: 'x_{\\text{new}} = \\dfrac{x - \\text{mean}}{\\text{standard deviation}}', link: W('knn') },
  { id: 'accuracy', lec: 3, tex: '\\text{accuracy} = \\dfrac{TP + TN}{TP + TN + FP + FN}', link: W('evaluate') },
  { id: 'precision', lec: 3, tex: '\\text{precision} = \\dfrac{TP}{TP + FP} \\qquad \\text{recall} = \\dfrac{TP}{TP + FN}', link: W('evaluate') },
  { id: 'f1', lec: 3, tex: 'F_1 = \\dfrac{2\\,P\\,R}{P + R}', link: W('evaluate') },
  { id: 'sse', lec: 4, tex: '\\text{SSE} = \\sum_k \\sum_{x \\in S_k} \\lVert x - c_k \\rVert^2', link: W('kmeans') },
  { id: 'centroid', lec: 4, tex: 'c_k \\leftarrow \\dfrac{1}{|S_k|}\\sum_{x \\in S_k} x', link: M('kmeans-table') },
  { id: 'neuron', lec: 5, tex: 'O = f\\Big(\\sum_i w_i x_i + \\theta\\Big)', link: W('perceptron') },
  { id: 'perceptron', lec: 5, tex: '\\Delta w_i = \\eta\\,(T - O)\\,x_i \\qquad \\Delta\\theta = \\eta\\,(T - O)', link: M('perceptron') },
  { id: 'sigmoid', lec: 6, tex: '\\sigma(z) = \\dfrac{1}{1 + e^{-z}} \\qquad \\sigma\'(z) = \\sigma(z)\\,(1 - \\sigma(z))', link: W('backprop') },
  { id: 'deltak', lec: 6, tex: '\\delta_k = (O_k - T_k)\\,O_k(1 - O_k)', link: W('backprop') },
  { id: 'deltaj', lec: 6, tex: '\\delta_j = O_j(1 - O_j)\\sum_k \\delta_k\\,w_{jk}', link: W('backprop') },
  { id: 'update', lec: 6, tex: 'w \\leftarrow w - \\eta\\,\\delta\\,(\\text{input of } w)', link: M('xor-mlp') },
  { id: 'dense', lec: 6, tex: '\\#\\text{params} = (n_{\\text{in}} + 1)\\times n_{\\text{out}}', link: M('xor-mlp') },
  { id: 'stretch', lec: 7, tex: 'I_{\\text{new}} = \\dfrac{I - I_{\\min}}{I_{\\max} - I_{\\min}} \\times 255', link: W('otsu') },
  { id: 'otsu', lec: 7, tex: 'T_0 = \\bar I, \\quad T \\leftarrow \\dfrac{\\mu_1 + \\mu_2}{2} \\ \\text{until stable}', link: W('otsu') },
  { id: 'range', lec: 7, tex: '\\max = 255\\sum_{w > 0} w, \\qquad \\min = 255\\sum_{w < 0} w', link: M('convolution') },
  { id: 'outsize', lec: 8, tex: '\\left\\lfloor \\dfrac{N - K + 2P}{S} \\right\\rfloor + 1 \\qquad \\text{same: } \\left\\lceil \\dfrac{N}{S} \\right\\rceil', link: W('cnn') },
  { id: 'convparams', lec: 8, tex: '\\#\\text{params} = (K \\cdot K \\cdot C_{\\text{in}} + 1) \\times F', link: M('cnn-shapes') },
  { id: 'dilated', lec: 8, tex: 'K_{\\text{eff}} = d\\,(K - 1) + 1', link: W('dilated') },
  { id: 'alphabeta', lec: 10, tex: '\\begin{gathered}\\text{MAX: } \\alpha \\leftarrow \\max(\\alpha, v) \\qquad \\text{MIN: } \\beta \\leftarrow \\min(\\beta, v) \\\\ \\alpha \\ge \\beta \\Rightarrow \\text{cut the remaining children}\\end{gathered}', link: W('alphabeta') },
]
