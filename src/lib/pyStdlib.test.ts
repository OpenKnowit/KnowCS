import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { PANDAS } from './pyPandas'
import { SKLEARN } from './pySklearn'
import { TORCH } from './pyTorch'
import { KERAS } from './pyKeras'

const out = (code: string) => {
  const r = runPython(code, { libs: [PANDAS, SKLEARN] })
  expect(r.error).toBeNull()
  return r.stdout
}
const failure = (code: string) => runPython(code).error

// What CPython cannot be asked about: the sandbox's own limits, and programs python -c cannot run.
describe('a parameter or feature the sandbox lacks says so', () => {
  const refused: [string, string][] = [
    ["import argparse; argparse.ArgumentParser(parents=[argparse.ArgumentParser(add_help=False)])", 'parents'],
    ["import argparse; argparse.ArgumentParser(fromfile_prefix_chars='@')", 'fromfile_prefix_chars'],
    ["import argparse; argparse.ArgumentParser(conflict_handler='resolve')", 'conflict_handler'],
    ['import argparse; argparse.ArgumentParser().add_subparsers()', 'add_subparsers'],
    ["import argparse\nclass A: pass\nargparse.ArgumentParser().add_argument('--x', action=A)", 'custom Action'],
    ["import argparse; argparse.ArgumentParser().add_argument_group('g', prefix_chars='+')", 'prefix_chars'],
    ["import enum\nclass E(enum.Enum):\n    def __new__(cls, v): return object.__new__(cls)\n    A = 1", '__new__'],
    ["import enum\nclass E(tuple, enum.Enum):\n    A = (1, 2)", 'tuple'],
    ["import pickle\nclass R:\n    def __reduce__(self): return (int, ())\npickle.dumps(R())", '__reduce__'],
    ['import pickle; len(pickle.dumps(1))', 'size of a pickle'],
    ["open('f', 'wb').write(b'abc')", 'no bytes type'],
    ["open('f', 'w').write('x'); open('f', 'rb')", 'no bytes type'],
    ["import io; io.BytesIO(b'abc')", 'no bytes type'],
    ["import pathlib; pathlib.Path('f').write_bytes(b'abc')", 'no bytes type'],
    ["import pickle, os; pickle.dump(1, open('f', 'wb')); os.path.getsize('f')", 'size of a pickle'],
    ["import pathlib; pathlib.PureWindowsPath('a')", 'PureWindowsPath'],
    ["import os; list(os.walk('.', topdown=False))", 'topdown'],
    ["import inspect; inspect.getsource(print)", 'getsource'],
    ["import uuid; uuid.UUID(int=5)", 'int'],
    ["import shutil; shutil.rmtree('.', onerror=print)", 'onerror'],
    ["import glob; glob.glob('*', dir_fd=1)", 'dir_fd'],
    ["open('f', 'w', opener=print)", 'opener'],
    ["open('f', 'w', newline='\\r')", 'newline'],
  ]
  for (const [code, word] of refused) {
    it(word, () => {
      const e = failure(code)
      expect(e?.type).toBe('NotImplementedError')
      expect(e?.message).toContain(word)
    })
  }
})

describe('files keep their helpful message for a run that stops', () => {
  it('names the sandbox when the error is not caught', () => {
    expect(failure("open('nope.txt')")?.message).toMatch(/^\[Errno 2\] No such file or directory: 'nope.txt' \(this sandbox starts with no files/)
  })
  it('is plain Python when it is caught', () => {
    expect(out("try:\n    open('nope.txt')\nexcept OSError as e:\n    print(e)")).toBe("[Errno 2] No such file or directory: 'nope.txt'\n")
  })
})

describe('a script run in the sandbox', () => {
  it('shows why argparse gave up, since there is no terminal to show it', () => {
    const e = failure("import argparse\np = argparse.ArgumentParser()\np.add_argument('--n', type=int)\np.parse_args(['--n', 'x'])\nprint('never')")
    expect(e?.type).toBe('SystemExit')
    expect(e?.message).toBe("usage: main.py [-h] [--n N]\nmain.py: error: argument --n: invalid int value: 'x'")
  })
  it('stays quiet when argparse exits happily, or sys.exit is called', () => {
    const r = runPython("import argparse\nargparse.ArgumentParser(prog='p').parse_args(['-h'])\nprint('never')")
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('usage: p [-h]\n\noptions:\n  -h, --help  show this help message and exit\n')
  })
  it('has an empty argv and a name', () => {
    expect(out("import argparse, sys\np = argparse.ArgumentParser()\nprint(p.prog, p.parse_args(), sys.argv)")).toBe("main.py Namespace() ['main.py']\n")
  })
  it('knows __file__ and its docstring', () => {
    expect(out(`"""the doc"""\nfrom pathlib import Path\nprint(__doc__, Path(__file__).name, Path(__file__).parent, Path(__file__).is_absolute())`)).toBe('the doc main.py /sandbox True\n')
    expect(out('print(__doc__)')).toBe('None\n')
  })
  it('works from /sandbox, on nothing real', () => {
    expect(out("import os, pathlib, tempfile\nprint(os.getcwd(), pathlib.Path.cwd(), pathlib.Path('a').resolve(), tempfile.gettempdir(), pathlib.Path.home())")).toBe('/sandbox /sandbox /sandbox/a /tmp /sandbox\n')
    expect(out("import pathlib\npathlib.Path('/tmp/x').mkdir()\nprint(pathlib.Path('/tmp/x').is_dir())")).toBe('True\n')
  })
  it('lets a notebook write to a Drive it mounted', () => {
    expect(out("from google.colab import drive\ndrive.mount('/content/drive')\nopen('/content/drive/MyDrive/a.txt', 'w').write('x')\nopen('/content/b.txt', 'w').write('y')\nprint(open('/content/drive/MyDrive/a.txt').read())")).toContain('x\n')
  })
  it('gives a platform that agrees with sys', () => {
    expect(out("import platform, sys\nprint(platform.python_version() == '.'.join(map(str, sys.version_info[:3])))")).toBe('True\n')
  })
  it('keeps uuid and secrets away from the seeded random', () => {
    expect(out("import random, uuid, secrets, tempfile\nrandom.seed(1); a = random.random()\nrandom.seed(1); uuid.uuid4(); secrets.token_hex(); tempfile.mkdtemp(); print(random.random() == a)")).toBe('True\n')
  })
})

describe('a Path reaches the libraries that read files', () => {
  it('pandas reads and writes through one', () => {
    expect(out("import pandas as pd\nfrom pathlib import Path\nd = Path('data'); d.mkdir()\npd.DataFrame({'a': [1, 2]}).to_csv(d / 't.csv', index=False)\nprint((d / 't.csv').read_text(), pd.read_csv(d / 't.csv').a.sum())")).toBe('a\n1\n2\n 3\n')
  })
  it('pandas says what is missing', () => {
    const r = runPython("import pandas as pd\nfrom pathlib import Path\npd.read_csv(Path('nope.csv'))", { libs: [PANDAS] })
    expect(r.error?.type).toBe('FileNotFoundError')
  })
})

describe('pickle copies library objects too', () => {
  it('a fitted estimator survives, and does not share state', () => {
    const code = `
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
X = np.array([[0.0], [1.0], [10.0], [11.0]]); y = np.array([0, 0, 1, 1])
m = KNeighborsClassifier(n_neighbors=1).fit(X, y)
with open('m.pkl', 'wb') as f:
    pickle.dump(m, f)
n = pickle.load(open('m.pkl', 'rb'))
print(n.predict(np.array([[0.5], [10.5]])), n is m, n)
m.fit(X, 1 - y)
print(m.predict(np.array([[0.5]])), n.predict(np.array([[0.5]])))
`
    expect(out(code)).toBe('[0 1] False KNeighborsClassifier(n_neighbors=1)\n[1] [0]\n')
  })
  it('a pipeline, a torch model and a Keras model too', () => {
    const r = runPython(`
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X = np.array([[0.0], [1.0], [10.0], [11.0]]); y = np.array([0, 0, 1, 1])
p = Pipeline([('s', StandardScaler()), ('c', LogisticRegression())]).fit(X, y)
q = pickle.loads(pickle.dumps(p))
before = q.predict(X).tolist()
p.fit(X, 1 - y)
print(before, q.predict(X).tolist(), p.predict(X).tolist())

import torch, torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(1, 1)
    def forward(self, x):
        return self.f(x)
torch.manual_seed(0)
a = Net()
b = pickle.loads(pickle.dumps(a))
with torch.no_grad():
    a.f.weight.fill_(5.0)
x = torch.ones(1, 1)
print(a.f.weight.item(), b.f.weight.item() != 5.0, type(b).__name__, isinstance(b, nn.Module), b(x).shape)
s = pickle.loads(pickle.dumps(a.state_dict()))
print(sorted(s), s['f.weight'].item())
t = torch.tensor([1.0, 2.0])
u = pickle.loads(pickle.dumps(t))
u[0] = 9.0
print(t.tolist(), u.tolist())

from tensorflow import keras
m = keras.Sequential([keras.layers.Dense(1, input_shape=(1,))])
n = pickle.loads(pickle.dumps(m))
m.layers[0].kernel.assign([[7.0]])
print(float(m.layers[0].kernel.numpy()[0, 0]), float(n.layers[0].kernel.numpy()[0, 0]) != 7.0, n.predict(np.array([[1.0]]), verbose=0).shape)
`, { libs: [SKLEARN, TORCH, KERAS] })
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('[0, 0, 1, 1] [0, 0, 1, 1] [1, 1, 0, 0]\n5.0 True Net True torch.Size([1, 1])\n[\'f.bias\', \'f.weight\'] 5.0\n[1.0, 2.0] [9.0, 2.0]\n7.0 True (1, 1)\n')
  })
  it('a DataFrame too', () => {
    expect(out("import pickle, pandas as pd\ndf = pd.DataFrame({'a': [1]})\ne = pickle.loads(pickle.dumps(df))\ne.loc[0, 'a'] = 9\nprint(df.a[0], e.a[0], type(e).__name__)")).toBe('1 9 DataFrame\n')
  })
  it('the stand-in for bytes prints as bytes, and only pickle reads it', () => {
    expect(out("import pickle\nb = pickle.dumps([1])\nprint(b, type(b).__name__)")).toBe("b'\\x80\\x04<pickled list: only pickle can read this>' bytes\n")
    expect(out("import pickle\nb = pickle.dumps(1)\nprint(pickle.loads(b), b == pickle.dumps(1))")).toBe('1 True\n')
    expect(failure("import pickle\nb = pickle.dumps(1)\nb + b")?.type).toBe('TypeError')
    expect(failure("import pickle\nb = pickle.dumps(1)\nb[0]")?.type).toBe('TypeError')
  })
})
