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
  it('sklearn and cv2 arrive the same way', async () => {
    const code = "import numpy as np, cv2\nfrom sklearn.neighbors import KNeighborsClassifier as K\nprint(K(1).fit([[0], [1]], [0, 1]).predict([[0.9]]), cv2.cvtColor(np.zeros((1, 1, 3), np.uint8), cv2.COLOR_BGR2GRAY).dtype)"
    expect(waitingFor(code).sort()).toEqual(['cv2', 'sklearn'])
    await loadLibsFor(code)
    const r = runPython(code, { libs: loadedLibs() })
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('[1] uint8\n')
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
