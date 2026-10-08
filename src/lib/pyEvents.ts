/**
 * What the sandbox libraries (pyplot, torch, keras, pandas …) report besides stdout: events recorded at the line that
 * caused them (a figure after each plotting call, a model's shape flow, an autograd graph …) and displays shown under
 * the output (open figures, a DataFrame as a table). The playground panels render these; minipy only stores them.
 */

// ---------------------------------------------------------------- matplotlib

export type Marker = 'o' | 's' | '^' | 'x' | '+' | '.' | '*' | 'D' | 'v' | null

export type Artist =
  | { kind: 'line'; x: number[]; y: number[]; color: string; width: number; dash: string | null; marker: Marker; label: string | null; alpha: number }
  | { kind: 'scatter'; x: number[]; y: number[]; colors: string[]; sizes: number[]; marker: Marker; label: string | null; alpha: number; edge: string | null }
  | { kind: 'bar'; x: number[]; heights: number[]; bottoms: number[]; width: number; colors: string[]; label: string | null; horizontal: boolean; alpha: number; tickLabels: string[] | null }
  | { kind: 'hist'; edges: number[]; counts: number[]; color: string; label: string | null; alpha: number }
  | { kind: 'image'; rows: number; cols: number; /** CSS colours, row-major */ pixels: string[]; values: number[]; cmap: string | null; vmin: number; vmax: number; auto: boolean; rgb: boolean; extent: [number, number, number, number] }
  | { kind: 'hline' | 'vline'; at: number; color: string; dash: string | null; width: number; label: string | null }
  | { kind: 'text'; x: number; y: number; text: string; color: string; size: number; ha: 'left' | 'center' | 'right'; va: 'top' | 'center' | 'bottom' }
  | { kind: 'fill'; x: number[]; y1: number[]; y2: number[]; color: string; alpha: number; label: string | null }
  /** contourf: one colour per grid point (row 0 = the first y), stretched over the grid's extent */
  | { kind: 'mesh'; nx: number; ny: number; extent: [number, number, number, number]; colors: string[]; values: number[]; alpha: number }
  /** pie: wedges counter-clockwise from startangle (degrees), labels at 1.1 r, autopct text at 0.6 r */
  | { kind: 'pie'; fracs: number[]; colors: string[]; labels: string[] | null; pct: string[] | null; start: number; explode: number[] }
  /** errorbar: [below, above] error per point */
  | { kind: 'errbar'; x: number[]; y: number[]; xerr: [number[], number[]] | null; yerr: [number[], number[]] | null; color: string; cap: number }
  /** boxplot: quartiles, whiskers at the furthest points within 1.5 IQR, fliers beyond */
  | { kind: 'box'; boxes: { pos: number; q1: number; med: number; q3: number; lo: number; hi: number; fliers: number[] }[]; width: number; vert: boolean }
  | { kind: 'arrow'; x1: number; y1: number; x2: number; y2: number; color: string }

export interface AxesSpec {
  /** grid position: row, column, and how many rows / columns the grid has */
  pos: { rows: number; cols: number; index: number }
  title: string | null
  xlabel: string | null
  ylabel: string | null
  xlim: [number, number] | null
  ylim: [number, number] | null
  /** xticks / yticks set by the user; [] hides them (plt.axis('off') hides both and the frame) */
  xticks: { at: number[]; labels: string[] | null } | null
  yticks: { at: number[]; labels: string[] | null } | null
  axisOff: boolean
  grid: boolean
  legend: { loc: string } | null
  equal: boolean
  /** image axes put y = 0 at the top */
  yInverted: boolean
  artists: Artist[]
  colorbar: { cmap: string; vmin: number; vmax: number } | null
}

export interface FigureSpec {
  num: number
  /** inches, as in figsize=(w, h) */
  size: [number, number]
  suptitle: string | null
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
  | { type: 'figure'; fig: FigureSpec; focus: { ax: number; artist: number | null } | null }
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
