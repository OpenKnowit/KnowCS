import type { Episode } from '../Player'
import { alphabeta } from './alphabeta'
import { backpropEp } from './backprop'
import { bayes } from './bayes'
import { cnn } from './cnn'
import { kmeans } from './kmeans'
import { knn } from './knn'
import { perceptron } from './perceptron'
import { xor } from './xor'

/** Animation code for every episode listed in sitemap EPISODES. */
export const EPISODE_CODE: Record<string, Episode> = { xor, bayes, backprop: backpropEp, cnn, kmeans, alphabeta, knn, perceptron }
