import { describe, expect, it } from 'vitest'
import { ON_DEMAND, importsOf, loadLibsFor, loadedLibs, waitingFor } from './pyLazy'
import { runPython } from './minipy'

describe('libraries fetched when a program imports them', () => {
  it('reads the imports out of a program', () => {
    const code = 'import numpy as np, pandas as pd\nfrom torch import nn\n  import keras.layers\nx = 1  # import os\nprint("import y")'
    expect(importsOf(code).sort()).toEqual(['keras', 'numpy', 'pandas', 'torch'])
  })
  it('reads imports a line does not start with', () => {
    const names = (code: string) => importsOf(code).sort()
    expect(names('import os; import argparse')).toEqual(['argparse', 'os'])
    expect(names('try: import enum\nexcept ImportError: pass')).toEqual(['enum'])
    expect(names('if True: from pathlib import Path')).toEqual(['pathlib'])
    expect(names('import os, \\\n  inspect')).toEqual(['inspect', 'os'])
    expect(names('exec("import textwrap")\nexec(\'from fractions import Fraction\')')).toEqual(['fractions', 'textwrap'])
    // not imports, so nothing is fetched for them
    expect(names('# note: import torch\nx = 1  # todo; from pandas import x\nprint("import y")\nd = {"k": 1}')).toEqual([])
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
  it('the rarely used standard library arrives the same way, as one chunk', async () => {
    const code = "import argparse, enum\nfrom pathlib import Path\nprint(Path('a/b.txt').suffix, argparse.ArgumentParser(prog='p').prog)"
    expect(runPython(code).error?.message).toMatch(/^No module named 'argparse': this sandbox fetches it/)
    expect(waitingFor(code).sort()).toEqual(['argparse', 'enum', 'pathlib'])
    await loadLibsFor(code)
    expect(waitingFor(code)).toEqual([])
    expect(new Set(loadedLibs().filter((l) => l.modules.includes('pathlib') || l.modules.includes('enum'))).size).toBe(1)
    const r = runPython(code, { libs: loadedLibs() })
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('.txt p\n')
  })
  it('every module of that chunk can be fetched by name, and the error for an unknown module still lists them', async () => {
    await loadLibsFor('import uuid')
    const more = loadedLibs().find((l) => l.modules.includes('uuid'))!
    expect(more.modules.every((name) => ON_DEMAND.includes(name))).toBe(true)
    for (const name of more.modules) expect(runPython(`import ${name}`, { libs: [more] }).error).toBeNull()
    const listed = runPython('import nosuchthing').error?.message
    for (const name of ['argparse', 'pathlib', 'pandas', 'torch', 'json', 'pickle']) expect(listed).toContain(name)
  })
  it('what other modules lean on stays with the interpreter, not in that chunk', () => {
    // each runs with no chunk fetched: open, pickle, copy, os and sys are always there
    const r = runPython(`
import os, sys, json, pickle, copy, re, collections, itertools, functools, dataclasses, time, io, typing, abc
import numpy as np
np.save('a.npy', np.arange(3))
print(np.load('a.npy'), os.path.exists('a.npy'), os.getcwd())
with open('b.pkl', 'wb') as f:
    pickle.dump({'x': [1, 2]}, f)
with open('b.pkl', 'rb') as f:
    print(pickle.load(f), copy.deepcopy([1, [2]]), json.dumps({'a': 1}), time.strftime('%%') )
`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe("[0 1 2] True /sandbox\n{'x': [1, 2]} [1, [2]] {\"a\": 1} %\n")
  })
  it('a Path and a Fraction survive pickle and copy as before they moved', async () => {
    await loadLibsFor('import fractions')
    const r = runPython(`
import pickle, copy
from pathlib import Path
from fractions import Fraction
p = Path('a/b.txt')
f = Fraction(1, 3)
q = pickle.loads(pickle.dumps({'p': p, 'f': f}))
print(q['p'], q['f'], q['p'] == p, q['f'] == f, type(q['p']).__name__, copy.deepcopy(p), copy.copy(f))
`, { libs: loadedLibs() })
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('a/b.txt 1/3 True True PosixPath a/b.txt 1/3\n')
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
