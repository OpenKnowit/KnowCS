/**
 * What the sandbox libraries (pyplot, torch, keras, pandas …) report besides stdout: events recorded at the line that
 * caused them (a figure after each plotting call, a model's shape flow, an autograd graph …) and displays shown under
 * the output (open figures, a DataFrame as a table). The playground panels render these; minipy only stores them.
 */

// ---------------------------------------------------------------- matplotlib

/** a matplotlib marker symbol ('o', 's', '^', 'x', …) */
export type Marker = string | null

/** which coordinate system a position is in */
export type Coord = 'data' | 'axes' | 'figure'

/** what every drawn artist has: draw order, visibility, clipping to the Axes, coordinate system per direction */
export interface ArtistBase { z: number; visible: boolean; clip: boolean; tx: Coord; ty: Coord }

/** dash pattern in units of the line width (matplotlib scales dashes by it), null = solid */
export type Dash = number[] | null

export interface LineArtist extends ArtistBase {
  kind: 'line'; x: number[]; y: number[]; color: string; width: number; /** false for linestyle 'None' */ line: boolean; dash: Dash
  marker: Marker; ms: number; mfc: string; mec: string; mew: number; steps: 'pre' | 'post' | 'mid' | null; markevery: number | null; label: string | null; alpha: number
}
export interface ScatterArtist extends ArtistBase {
  kind: 'scatter'; x: number[]; y: number[]; colors: string[]; sizes: number[]; marker: Marker; /** null = same as the face */ edges: string[] | null; lw: number[]; label: string | null; alpha: number
}
/** bars: x = left edge (bottom edge if horizontal), heights along the value axis from bottoms */
export interface BarArtist extends ArtistBase {
  kind: 'bar'; x: number[]; heights: number[]; bottoms: number[]; widths: number[]; colors: string[]; edges: string[]; lw: number; dash: Dash; label: string | null; horizontal: boolean; alpha: number
}
export interface ImageArtist extends ArtistBase {
  kind: 'image'; rows: number; cols: number; /** CSS colours, row-major */ pixels: string[]; values: number[]; cmap: string | null; vmin: number; vmax: number; auto: boolean; rgb: boolean; extent: [number, number, number, number]; alpha: number; smooth: boolean
}
export interface TextBox { fc: string; ec: string; lw: number; pad: number; round: boolean; alpha: number }
export interface TextArrow { x: number; y: number; tx: Coord; ty: Coord; /** 'simple' = the shaft-and-head arrow, otherwise an arrowstyle like '->' */ style: string; color: string; lw: number; head: number; shrink: number; edge: string }
export interface TextArtist extends ArtistBase {
  kind: 'text'; x: number; y: number; text: string; color: string; size: number; ha: 'left' | 'center' | 'right'; va: 'top' | 'center' | 'bottom' | 'baseline'; rotation: number; weight: string; italic: boolean; alpha: number
  /** shift in points, for offset coordinates */ dx: number; dy: number; bbox: TextBox | null; arrow: TextArrow | null
}
/** filled and / or stroked polygons; fcs and ecs repeat over the paths ('none' = nothing) */
export interface PolyArtist extends ArtistBase {
  kind: 'poly'; /** flat x, y lists */ paths: number[][]; closed: boolean; fcs: string[]; ecs: string[]; lw: number; dash: Dash; alpha: number; label: string | null; /** stroke with the fill colour so neighbours leave no seams */ seamless: boolean
}
export interface SegsArtist extends ArtistBase {
  kind: 'segs'; paths: number[][]; colors: string[]; widths: number[]; dash: Dash; alpha: number; label: string | null
}
/** pcolormesh: (ny + 1) × (nx + 1) corners, ny × nx colours */
export interface MeshArtist extends ArtistBase {
  kind: 'mesh'; nx: number; ny: number; X: number[]; Y: number[]; colors: string[]; alpha: number; /** evenly spaced rows and columns: drawn as one picture */ uniform: boolean; edge: string | null; lw: number
}

export type Artist = LineArtist | ScatterArtist | BarArtist | ImageArtist | TextArtist | PolyArtist | SegsArtist | MeshArtist

/** one tick direction, fully resolved: where the ticks are and how they look */
export interface AxisSpec {
  at: number[]; labels: string[]; minor: number[]; minorLabels: string[]; /** '1e6' or '+1.5e3' at the end of the axis */ offset: string
  visible: boolean
  /** ticks and labels on the low side (bottom / left) and the high side (top / right) */
  tick1: boolean; tick2: boolean; label1: boolean; label2: boolean
  length: number; width: number; minorLength: number; minorWidth: number; color: string; dir: 'out' | 'in' | 'inout'; pad: number
  labelSize: number; labelColor: string; rotation: number; ha: string | null; va: string | null
  /** per-tick overrides set through a tick label's handle */
  styles: Record<number, { color?: string; size?: number; rotation?: number; visible?: boolean; weight?: string }>
  labelSide: 'low' | 'high'; labelPad: number
  label: LabelSpec | null
}

export interface LabelSpec { text: string; size: number; color: string; weight: string; italic: boolean; loc: 'left' | 'center' | 'right'; rotation: number }

export interface LegendEntry {
  label: string
  kind: 'line' | 'patch' | 'scatter'
  color: string; width: number; dash: Dash; line: boolean; marker: Marker; ms: number; mfc: string; mec: string; mew: number
  fc: string; ec: string; lw: number; size: number; alpha: number
}

export interface LegendSpec {
  /** a location name, or the lower-left corner in Axes fractions */
  loc: string | [number, number]
  anchor: number[] | null
  anchorCoord: Coord
  entries: LegendEntry[]; ncol: number; size: number; title: string | null; titleSize: number
  frame: boolean; alpha: number; fc: string; ec: string; markerscale: number
  /** paddings in font sizes */
  borderpad: number; labelspacing: number; handlelength: number; handletextpad: number; borderaxespad: number; columnspacing: number
  labelcolor: string | null; z: number
}

export interface SpineSpec { visible: boolean; color: string; width: number; at: ['data' | 'axes', number] | null; dash: Dash }

export interface AxesSpec {
  /** position in the figure as fractions: left, bottom, width, height */
  rect: [number, number, number, number]
  title: string | null
  titles: (LabelSpec & { pad: number; y: number | null })[]
  xlim: [number, number]
  ylim: [number, number]
  xscale: string
  yscale: string
  xticks: AxisSpec
  yticks: AxisSpec
  axisOff: boolean
  frame: boolean
  facecolor: string
  patchVisible: boolean
  visible: boolean
  grid: { x: boolean; y: boolean; xminor: boolean; yminor: boolean; color: string; width: number; dash: Dash; alpha: number; below: boolean }
  spines: Record<'left' | 'right' | 'top' | 'bottom', SpineSpec>
  legends: LegendSpec[]
  artists: Artist[]
}

export interface FigureSpec {
  num: number
  /** inches, as in figsize=(w, h) */
  size: [number, number]
  facecolor: string
  suptitle: string | null
  /** suptitle, supxlabel, supylabel, figtext, in figure fractions */
  texts: TextArtist[]
  /** artists added to the figure itself */
  legends: LegendSpec[]
  axes: AxesSpec[]
}

// ---------------------------------------------------------------- tables (pandas)

export interface TableSpec {
  /** column names; [] for a Series' single unnamed column */
  columns: string[]
  index: string[]
  /** cell text, row-major; null = missing (NaN / None) */
  cells: (string | null)[][]
  /** dtype per column */
  dtypes: string[]
  /** a Series is drawn as one column */
  series: boolean
  name: string | null
  indexName: string | null
}

// ---------------------------------------------------------------- neural networks

export interface FlowRow {
  /** attribute name (conv1) or layer index */
  name: string
  /** Conv2d(1, 32, kernel_size=(3, 3), …) */
  layer: string
  /** null = a dimension Keras leaves open (the batch) */
  input: (number | null)[]
  output: (number | null)[]
  params: number
  /** extra explanation: the output-size formula with numbers filled in */
  note: string | null
}

export interface GraphNode {
  id: number
  /** variable name or the operation that made it (mul, add, pow …) */
  label: string
  op: string | null
  shape: number[]
  value: string
  grad: string | null
  leaf: boolean
  requiresGrad: boolean
  inputs: number[]
}

// ---------------------------------------------------------------- events and displays

export type PyEventData =
  | { type: 'figure'; fig: FigureSpec; /** the Axes and the artists the call added (artist … last) */ focus: { ax: number; artist: number | null; last?: number } | null }
  | { type: 'frame'; api: string; inputs: { label: string; table: TableSpec }[]; result: TableSpec | null; resultText: string; /** source rows / columns (positions) that made the result */ picks: { rows: number[] | null; cols: number[] | null } | null; /** group of each input row (groupby) */ groups: { keys: string[]; ofRow: number[] } | null }
  | { type: 'flow'; framework: 'torch' | 'keras'; model: string; rows: FlowRow[]; total: number; trainable: number; title: string }
  | { type: 'graph'; nodes: GraphNode[]; root: number; title: string }
  | { type: 'history'; metrics: Record<string, number[]>; title: string }
  | { type: 'note'; tone: 'info' | 'warn'; key: string; params: Record<string, string | number> }

export type PyEventInput = PyEventData & { code?: string; line?: number }
export type PyEvent = PyEventData & { id: number; seq: number; line: number; code: string }

export type Display =
  | { type: 'figure'; fig: FigureSpec }
  | { type: 'table'; table: TableSpec }
