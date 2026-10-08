// --- Real Python in the browser (Pyodide): where it is downloaded from and what it can import ---
// minipy stays the playground's engine (it traces every step); Pyodide is an opt-in check with the real libraries.

export const PYODIDE_VERSION = '314.0.7'

/**
 * Where pyodide.mjs, the runtime and the package wheels are fetched from. jsDelivr is unreliable from mainland China,
 * so a build can point elsewhere (a self-hosted copy, COS + CDN) with VITE_PYODIDE_BASE=https://…/full/
 */
export const pyodideBase = (env: string | undefined = import.meta.env.VITE_PYODIDE_BASE): string => {
  const base = env?.trim() || `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`
  return base.endsWith('/') ? base : `${base}/`
}

/** import names that are not in Pyodide's lock file but are pure Python, so micropip installs them from PyPI */
export const MICROPIP_IMPORTS: Record<string, string> = { seaborn: 'seaborn' }

/** import name → Pyodide package name, where they differ */
const PACKAGE_OF: Record<string, string> = { sklearn: 'scikit-learn', cv2: 'opencv-python', PIL: 'pillow', skimage: 'scikit-image' }
export const packageOf = (mod: string): string => PACKAGE_OF[mod] ?? mod

/** the top-level modules a piece of code imports (a quick scan for the status line; Pyodide does the real one) */
export const importsOf = (code: string): string[] => {
  const out = new Set<string>()
  for (const m of code.matchAll(/^[ \t]*(?:from[ \t]+([A-Za-z_]\w*)[\w.]*[ \t]+import|import[ \t]+([^\n#;]+))/gm)) {
    if (m[1]) out.add(m[1])
    else for (const part of m[2].split(',')) {
      const name = part.trim().split(/[\s.]/)[0]
      if (/^[A-Za-z_]\w*$/.test(name)) out.add(name)
    }
  }
  return [...out]
}

/** libraries whose playground offers the real-Python check (PyTorch, Keras and TensorFlow have no Pyodide build) */
export const REAL_PYTHON_LIBS = new Set(['numpy', 'matplotlib', 'pandas'])
