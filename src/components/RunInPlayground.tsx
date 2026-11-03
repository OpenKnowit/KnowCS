import { Play } from 'lucide-react'
import { useTranslation } from 'react-i18next'

// "Run in Playground": opens the playground with this code in the given library; the playground shows a Back button
// to the page it came from (PlaygroundPage reads ?code= and ?from=).
const playgroundHref = (code: string, lib: 'numpy' | 'pytorch' | 'keras'): string =>
  `/playground/?code=${encodeURIComponent(code)}&from=${encodeURIComponent(window.location.pathname)}#${lib}`

export function RunInPlayground({ code, lib, className = '' }: { code: string; lib: 'numpy' | 'pytorch' | 'keras'; className?: string }) {
  const { t } = useTranslation()
  return (
    <a href={playgroundHref(code, lib)} className={`inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-blue-600 px-2.5 py-1 text-xs font-bold text-white hover:bg-blue-500 ${className}`}>
      <Play size={12} aria-hidden />
      {t('site.ui.run_playground')}
    </a>
  )
}
