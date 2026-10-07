import type { Episode } from '../Player'
import { bayes } from './bayes'
import { xor } from './xor'

/** Animation code for every episode listed in sitemap EPISODES. */
export const EPISODE_CODE: Record<string, Episode> = { xor, bayes }
