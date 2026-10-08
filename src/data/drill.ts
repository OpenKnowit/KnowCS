/**
 * Concept check (/drill/): original true/false statements in the style of the finals' Problem 1.
 * Text lives in the locales under drill.items.<id>.{q,why}; `link` points to the explainer or practice page.
 */
export interface DrillItem {
  id: string
  lec: number
  answer: boolean
  link: { kind: 'watch' | 'module'; id: string }
}

const W = (id: string) => ({ kind: 'watch' as const, id })
const M = (id: string) => ({ kind: 'module' as const, id })

export const DRILL: DrillItem[] = [
  { id: 'np1', lec: 1, answer: true, link: M('numpy') },
  { id: 'np2', lec: 1, answer: true, link: W('broadcast') },
  { id: 'np3', lec: 1, answer: false, link: W('broadcast') },
  { id: 'np4', lec: 1, answer: false, link: M('numpy') },
  { id: 'by1', lec: 2, answer: false, link: W('bayes') },
  { id: 'by2', lec: 2, answer: true, link: W('bayes') },
  { id: 'by3', lec: 2, answer: true, link: W('bayes') },
  { id: 'by4', lec: 2, answer: false, link: M('naive-bayes') },
  { id: 'by5', lec: 2, answer: true, link: W('gaussian') },
  { id: 'by6', lec: 2, answer: false, link: M('naive-bayes') },
  { id: 'kn1', lec: 3, answer: true, link: W('knn') },
  { id: 'kn2', lec: 3, answer: false, link: W('knn') },
  { id: 'kn3', lec: 3, answer: true, link: W('knn') },
  { id: 'kn4', lec: 3, answer: false, link: W('knn') },
  { id: 'kn5', lec: 3, answer: false, link: W('knn') },
  { id: 'ev1', lec: 3, answer: true, link: W('evaluate') },
  { id: 'ev2', lec: 3, answer: true, link: W('evaluate') },
  { id: 'ev3', lec: 3, answer: true, link: W('evaluate') },
  { id: 'ev4', lec: 3, answer: false, link: W('evaluate') },
  { id: 'km1', lec: 4, answer: false, link: W('kmeans') },
  { id: 'km2', lec: 4, answer: true, link: W('kmeans') },
  { id: 'km3', lec: 4, answer: true, link: M('kmeans-table') },
  { id: 'km4', lec: 4, answer: false, link: W('kmeans') },
  { id: 'pc1', lec: 5, answer: false, link: W('perceptron') },
  { id: 'pc2', lec: 5, answer: true, link: W('perceptron') },
  { id: 'pc3', lec: 5, answer: true, link: M('perceptron') },
  { id: 'ml1', lec: 6, answer: false, link: W('xor') },
  { id: 'ml2', lec: 6, answer: true, link: M('xor-mlp') },
  { id: 'ml3', lec: 6, answer: true, link: M('xor-mlp') },
  { id: 'ml4', lec: 6, answer: false, link: W('backprop') },
  { id: 'ml5', lec: 6, answer: true, link: M('xor-mlp') },
  { id: 'ml6', lec: 6, answer: false, link: W('xor') },
  { id: 'im1', lec: 7, answer: false, link: W('otsu') },
  { id: 'im2', lec: 7, answer: true, link: W('otsu') },
  { id: 'im3', lec: 7, answer: true, link: W('otsu') },
  { id: 'im4', lec: 7, answer: false, link: M('affine') },
  { id: 'im5', lec: 7, answer: false, link: M('affine') },
  { id: 'im6', lec: 7, answer: true, link: M('affine') },
  { id: 'im7', lec: 7, answer: false, link: M('affine') },
  { id: 'im8', lec: 7, answer: true, link: M('affine') },
  { id: 'cn1', lec: 8, answer: false, link: W('cnn') },
  { id: 'cn2', lec: 8, answer: true, link: W('cnn') },
  { id: 'cn3', lec: 8, answer: false, link: M('cnn-shapes') },
  { id: 'cn4', lec: 8, answer: true, link: W('dilated') },
  { id: 'cn5', lec: 8, answer: false, link: W('dilated') },
  { id: 'cn6', lec: 8, answer: false, link: M('cnn-shapes') },
  { id: 'cn7', lec: 8, answer: false, link: W('dilated') },
  { id: 'pt1', lec: 9, answer: false, link: M('pytorch') },
  { id: 'pt2', lec: 9, answer: true, link: M('pytorch') },
  { id: 'pt3', lec: 9, answer: true, link: M('pytorch') },
  { id: 'pt4', lec: 9, answer: false, link: M('pytorch') },
  { id: 'ab1', lec: 10, answer: false, link: W('alphabeta') },
  { id: 'ab2', lec: 10, answer: false, link: W('alphabeta') },
  { id: 'ab3', lec: 10, answer: true, link: W('alphabeta') },
  { id: 'ab4', lec: 10, answer: false, link: W('alphabeta') },
  { id: 'et1', lec: 11, answer: false, link: W('ethics') },
  { id: 'et2', lec: 11, answer: true, link: W('ethics') },
  { id: 'et3', lec: 11, answer: true, link: W('ethics') },
]
