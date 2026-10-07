/** KnowCS Lab content map: lectures, past papers, lab pages, live site modules and recurring exam topics. */

export interface Paper {
  id: string
  label: string
  kind: 'M' | 'F'
}

// The nine papers analysed (with solutions). Order = column order of the exam heat map.
export const PAPERS: Paper[] = [
  { id: '22F', label: '2022 Fall', kind: 'M' },
  { id: '22S', label: '2022 Spring', kind: 'M' },
  { id: '23F', label: '2023 Fall', kind: 'M' },
  { id: '23S', label: '2023 Spring', kind: 'M' },
  { id: '24S', label: '2024 Spring', kind: 'M' },
  { id: '25S', label: '2025 Spring', kind: 'M' },
  { id: 'F22', label: 'Final 2022', kind: 'F' },
  { id: 'F23', label: 'Final 2023', kind: 'F' },
  { id: 'F24', label: 'Final 2024', kind: 'F' },
]

export const paperLabel = (id: string): string => {
  const p = PAPERS.find((x) => x.id === id)
  return p ? (p.kind === 'M' ? `${p.label} mid` : p.label) : id
}

export interface Lecture {
  n: number
  title: string
  /** midterm covers lectures 1–6, the final covers everything */
  scope: 'mid' | 'final'
}

export const LECTURES: Lecture[] = [
  { n: 1, title: 'Intro & NumPy', scope: 'mid' },
  { n: 2, title: 'Naive Bayes', scope: 'mid' },
  { n: 3, title: 'K-Nearest Neighbors', scope: 'mid' },
  { n: 4, title: 'K-Means Clustering', scope: 'mid' },
  { n: 5, title: 'Neurons & Perceptron', scope: 'mid' },
  { n: 6, title: 'Multilayer Perceptron', scope: 'mid' },
  { n: 7, title: 'Image Processing', scope: 'final' },
  { n: 8, title: 'Convolutional Networks', scope: 'final' },
  { n: 9, title: 'PyTorch', scope: 'final' },
  { n: 10, title: 'Minimax & Alpha-Beta', scope: 'final' },
  { n: 11, title: 'Ethics of AI', scope: 'final' },
]

export const lecture = (n: number): Lecture => LECTURES.find((l) => l.n === n) ?? LECTURES[0]

export interface Module {
  id: string
  lec: number
  title: string
  blurb: string
  /** papers that asked this pattern */
  exams: string[]
  /** absolute link for modules on the main site; lab pages use `${id}.html` */
  href?: string
  status: 'lab' | 'live'
}

// New lab pages. Order = prev / next order.
export const LAB: Module[] = [
  { id: 'bayes-virus', lec: 2, title: 'Bayes & the base rate', blurb: 'Why a 99% accurate test is usually wrong: 10,000 people in one picture.', exams: ['22F', '22S', '25S'], status: 'lab' },
  { id: 'gaussian-nb', lec: 2, title: 'Gaussian Naive Bayes', blurb: 'Numerical features: one bell curve per class, likelihood = curve height.', exams: ['23F', '24S', 'F24'], status: 'lab' },
  { id: 'evaluation', lec: 3, title: 'Confusion matrix & F1', blurb: 'TP / FP / FN per class, precision, recall, macro-F1 and MCC.', exams: ['23F', '24S', 'F22'], status: 'lab' },
  { id: 'cross-validation', lec: 3, title: 'D-fold cross-validation trap', blurb: 'Sorted data and no shuffle can score 0%. See which classes each fold loses.', exams: ['22F', '23F', '24S', 'F24'], status: 'lab' },
  { id: 'kmeans-table', lec: 4, title: 'K-Means by hand', blurb: 'The exam table: distances, assignment, new centroids, until nothing moves.', exams: ['22F', '22S', '23F', '23S', '25S', 'F23'], status: 'lab' },
  { id: 'perceptron', lec: 5, title: 'Perceptron learning table', blurb: 'Row-by-row updates of w and θ, and the decision line they draw.', exams: ['22F', '22S', '23F', '23S'], status: 'lab' },
  { id: 'xor-mlp', lec: 6, title: 'XOR & backprop by numbers', blurb: 'Why one line fails, how a hidden layer bends space, δ by δ.', exams: ['22S', '23S', '25S', 'F23', 'F24'], status: 'lab' },
  { id: 'convolution', lec: 7, title: 'Convolution, padding & flips', blurb: 'Four padding modes, the kernel flip, stride and output size.', exams: ['F22', 'F23', 'F24'], status: 'lab' },
  { id: 'otsu', lec: 7, title: 'Histogram, contrast & Otsu', blurb: 'Point operations reshape the histogram; Otsu finds the threshold.', exams: ['F22', 'F24'], status: 'lab' },
  { id: 'cnn-shapes', lec: 8, title: 'CNN shapes & parameters', blurb: 'Stack layers and get every output shape and parameter count.', exams: ['F22', 'F23', 'F24'], status: 'lab' },
]

// Modules already live on the React site.
export const SITE: Module[] = [
  { id: 'numpy', lec: 1, title: 'NumPy playground & broadcasting', blurb: 'Run NumPy in the browser and see where every element comes from.', exams: ['22F', '22S', '23F', '23S', '24S', '25S'], href: '/#/course/numpy', status: 'live' },
  { id: 'bayesBasics', lec: 2, title: 'Bayes basics', blurb: 'Fire and smoke: prior, likelihood, posterior.', exams: [], href: '/#/course/bayesBasics', status: 'live' },
  { id: 'naiveBayes', lec: 2, title: 'Naive Bayes (categorical)', blurb: 'Disease Z table, Laplace smoothing, log mode.', exams: ['22F', '22S', '23F', '24S', 'F22', 'F23'], href: '/#/course/naiveBayes', status: 'live' },
  { id: 'knn', lec: 3, title: 'K-Nearest Neighbors', blurb: 'Click to classify, standardize, pick K from the error curve.', exams: ['22F', '22S', '23F', '23S', '24S', '25S', 'F22'], href: '/#/course/knn', status: 'live' },
  { id: 'kmeans', lec: 4, title: 'K-Means animation', blurb: 'EM iterations, elbow curve, Z-score.', exams: [], href: '/#/course/kmeans', status: 'live' },
  { id: 'backprop', lec: 6, title: 'Backpropagation', blurb: 'One sigmoid neuron, chain rule with live numbers.', exams: [], href: '/#/course/backprop', status: 'live' },
  { id: 'kernel', lec: 7, title: 'Edge-detection kernels', blurb: 'Sobel, Laplacian and custom kernels on an image.', exams: [], href: '/#/course/kernel', status: 'live' },
  { id: 'alphabeta', lec: 10, title: 'Minimax & alpha-beta', blurb: 'Step through DFS with α / β and pruning.', exams: ['F22', 'F23', 'F24'], href: '/#/course/alphabeta', status: 'live' },
]

export const ALL_MODULES: Module[] = [...SITE, ...LAB].sort((a, b) => a.lec - b.lec)
export const moduleById = (id: string): Module | undefined => ALL_MODULES.find((m) => m.id === id)
export const hrefOf = (m: Module): string => m.href ?? `${m.id}.html`

export interface Topic {
  title: string
  lec: number
  papers: string[]
  links: string[]
}

// Recurring exam topics, from the nine papers with solutions.
export const TOPICS: Topic[] = [
  { title: 'NumPy: predict the output', lec: 1, papers: ['22F', '22S', '23F', '23S', '24S', 'F24'], links: ['numpy'] },
  { title: 'Vectorised pairwise distance', lec: 1, papers: ['22F', '22S', '23F', '23S', '24S', '25S'], links: ['numpy'] },
  { title: 'Bayes rule / total probability', lec: 2, papers: ['22F', '22S', '25S'], links: ['bayes-virus', 'bayesBasics'] },
  { title: 'Naive Bayes by hand + Laplace', lec: 2, papers: ['22F', '22S', '23F', '24S', 'F22', 'F23'], links: ['naiveBayes'] },
  { title: 'Gaussian likelihood', lec: 2, papers: ['23F', '24S', 'F24'], links: ['gaussian-nb'] },
  { title: 'KNN distance table', lec: 3, papers: ['22F', '22S', '23F', '23S', '24S', '25S', 'F22'], links: ['knn'] },
  { title: 'Cross-validation', lec: 3, papers: ['22F', '22S', '23F', '24S', 'F24'], links: ['cross-validation'] },
  { title: 'Confusion matrix / F1', lec: 3, papers: ['23F', '24S', 'F22'], links: ['evaluation'] },
  { title: 'K-Means iterations', lec: 4, papers: ['22F', '22S', '23F', '23S', '24S', '25S', 'F23'], links: ['kmeans-table', 'kmeans'] },
  { title: 'Perceptron update table', lec: 5, papers: ['22F', '22S', '23F', '23S'], links: ['perceptron'] },
  { title: 'Linear separability / XOR', lec: 5, papers: ['22F', '22S', '23F', '23S', '25S', 'F24'], links: ['perceptron', 'xor-mlp'] },
  { title: 'MLP numbers, params, activations', lec: 6, papers: ['22F', '22S', 'F22', 'F23', 'F24'], links: ['xor-mlp', 'backprop'] },
  { title: 'Padding, Otsu, kernel effects', lec: 7, papers: ['F22', 'F23', 'F24'], links: ['convolution', 'otsu', 'kernel'] },
  { title: 'CNN output shape / parameters', lec: 8, papers: ['F22', 'F23', 'F24'], links: ['cnn-shapes'] },
  { title: 'Minimax / alpha-beta', lec: 10, papers: ['F22', 'F23', 'F24'], links: ['alphabeta'] },
  { title: 'Ethics (multiple choice)', lec: 11, papers: ['F22', 'F23', 'F24'], links: [] },
]
