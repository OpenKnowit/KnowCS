import type { LucideIcon } from 'lucide-react'

interface SectionTitleProps {
  icon: LucideIcon
  title: string
  subtitle: string
  /** heading level: h1 when it is the page's main title (notes, extend), h2 inside a module */
  as?: 'h1' | 'h2'
}

export const SectionTitle = ({ icon: Icon, title, subtitle, as: Heading = 'h2' }: SectionTitleProps) => (
  <div className="mb-6">
    <div className="flex items-center gap-2 mb-1">
      <div className="p-2 bg-blue-100 rounded-lg text-blue-600">
        <Icon size={20} />
      </div>
      <Heading className="text-xl font-bold text-gray-800 uppercase tracking-tight">{title}</Heading>
    </div>
    <p className="text-sm text-gray-500 ml-10 italic">{subtitle}</p>
  </div>
)
