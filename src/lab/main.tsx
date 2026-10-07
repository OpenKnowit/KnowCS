import { StrictMode, type ComponentType } from 'react'
import { createRoot } from 'react-dom/client'
import './lab.css'

// One entry for every lab/*.html; each page only downloads its own chunk.
const PAGES: Record<string, () => Promise<{ default: ComponentType }>> = {
  index: () => import('./pages/LabIndex'),
  'home-a': () => import('./homes/HomeCourseMap'),
  'home-b': () => import('./homes/HomeExamRadar'),
  'home-c': () => import('./homes/HomeStudyPath'),
  'bayes-virus': () => import('./pages/BayesVirus'),
  'gaussian-nb': () => import('./pages/GaussianNb'),
  evaluation: () => import('./pages/Evaluation'),
  'cross-validation': () => import('./pages/CrossValidation'),
  'kmeans-table': () => import('./pages/KMeansTable'),
  perceptron: () => import('./pages/Perceptron'),
  'xor-mlp': () => import('./pages/XorMlp'),
  convolution: () => import('./pages/Convolution'),
  otsu: () => import('./pages/Otsu'),
  'cnn-shapes': () => import('./pages/CnnShapes'),
}

const id = document.body.dataset.page ?? 'index'
const load = PAGES[id] ?? PAGES.index
load().then(({ default: Page }) => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <Page />
    </StrictMode>,
  )
})
