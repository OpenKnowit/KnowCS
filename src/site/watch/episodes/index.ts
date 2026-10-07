import type { Episode } from '../Player'
import { alphabeta } from './alphabeta'
import { backpropEp } from './backprop'
import { bayes } from './bayes'
import { broadcast } from './broadcast'
import { cnn } from './cnn'
import { evaluate } from './evaluate'
import { kmeans } from './kmeans'
import { knn } from './knn'
import { perceptron } from './perceptron'
import { xor } from './xor'

/** Animation code for every episode listed in sitemap EPISODES. */
export const EPISODE_CODE: Record<string, Episode> = { xor, bayes, backprop: backpropEp, cnn, kmeans, alphabeta, knn, perceptron, evaluate, broadcast }
