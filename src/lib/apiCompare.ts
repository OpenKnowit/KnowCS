// How a sandbox output is compared with the real library's: the same text after dropping what only the machine decides
// (memory addresses, NumPy's scalar wrappers), with numbers equal to within rounding.
const NUM = /[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?/g
export const norm = (s: string) => s.replace(/\bnp\.(?:int|float|uint|bool)\w*\(([^)]*)\)/g, '$1').replace(/\bnp\.str_\(('[^']*')\)/g, '$1').replace(/ at 0x[0-9a-f]+/g, '').replace(/^No artists with labels found to put in legend\..*\n/m, '').replace(/^\(sandbox\) (?:figure|image) would be saved.*\n?/gm, '').replace(/\b(?:builtin_function_or_method|method|function)\b(?=\s*$)/gm, 'callable').replace(/[ \t]+$/gm, '').trimEnd()
const shape = (s: string) => s.replace(NUM, '#').replace(/\s+/g, ' ').replace(/([[(]) /g, '$1').replace(/ ([\]),])/g, '$1')
export const same = (a: string, b: string) => {
  if (a === b) return true
  if (shape(a) !== shape(b)) return false
  const x = a.match(NUM) ?? []
  const y = b.match(NUM) ?? []
  return x.every((v, i) => Math.abs(Number(v) - Number(y[i])) <= Math.max(5e-4 * Math.abs(Number(y[i])), 2e-4) && Math.abs(v.length - y[i].length) <= 8)
}
