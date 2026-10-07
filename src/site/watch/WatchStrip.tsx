import { EPISODES } from '../../lib/sitemap'
import { EpisodeCard } from './EpisodeCard'

/** The three most-examined explainers, for the home page. */
export default function WatchStrip() {
  return (
    <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {[...EPISODES]
        .sort((x, y) => y.exams.length - x.exams.length)
        .slice(0, 3)
        .map((e) => (
          <EpisodeCard key={e.id} info={e} />
        ))}
    </div>
  )
}
