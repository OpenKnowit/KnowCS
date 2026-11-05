import { describe, expect, it } from 'vitest'
import { importsOf, loadLibsFor, loadedLibs, waitingFor } from './pyLazy'
import { runPython } from './minipy'

describe('libraries fetched when a program imports them', () => {
  it('reads the imports out of a program', () => {
    const code = 'import numpy as np, pandas as pd\nfrom torch import nn\n  import keras.layers\nx = 1  # import os\nprint("import y")'
    expect(importsOf(code).sort()).toEqual(['keras', 'numpy', 'pandas', 'torch'])
  })
  it('waits only for libraries it knows', () => {
    expect(waitingFor('import numpy\nimport os\nimport nosuchthing')).toEqual([])
    expect(loadLibsFor('import numpy')).toBeNull()
  })
  it('a program runs once its library has arrived', async () => {
    const code = "import pandas as pd\nprint(pd.DataFrame({'a': [1]}).shape)"
    expect(runPython(code).error?.type).toBe('ModuleNotFoundError')
    await loadLibsFor(code)
    expect(waitingFor(code)).toEqual([])
    const r = runPython(code, { libs: loadedLibs() })
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('(1, 1)\n')
  })
})
