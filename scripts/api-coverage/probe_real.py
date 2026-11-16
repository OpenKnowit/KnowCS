#!/usr/bin/env python3
"""Tries each public name on the real library with a few ordinary arguments and keeps the calls that work.

A kept call is a small program and what it printed; gen.ts then runs it in the sandbox. Modules that touch the machine
(os, shutil, sys, ...) are never called, and names that change global state or could block are skipped.

    python3 scripts/api-coverage/probe_real.py out.json numpy matplotlib pandas keras tensorflow python sklearn cv2
    ~/repo/minimind/.venv/bin/python scripts/api-coverage/probe_real.py out-torch.json pytorch
"""
import contextlib
import io
import json
import os
import pathlib
import re
import shutil
import signal
import sys
import tempfile
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')
out_path, libs = sys.argv[1], sys.argv[2:]
real = json.loads((pathlib.Path(__file__).parent.parent / 'api-real-names.json').read_text())
if 'matplotlib' in libs:
    import matplotlib
    matplotlib.use('Agg')

UNSAFE = re.compile(r'^(set_|enable|disable|use_|seed$|manual_seed|shutdown|exit|abort|kill|fork|exec|system|popen|spawn|remove|unlink|rmdir|rmtree|save|load|show$|pause$|ginput|waitfor|input$|breakpoint|help$|get_file|download|fetch_|load_|from_file|fromfile|memmap|tofile|dump|to_(csv|pickle|parquet|excel|hdf|sql|feather|stata|json|html|latex|xml|clipboard|markdown|orc)|read_|DataFrame$|Series$|imread|imwrite|VideoCapture|namedWindow|waitKey|imshow$|select|connect|xkcd|install|switch_backend|ion$|ioff$|sleep|share_memory|print$|open$|compile$|globals|locals|vars|memoryview|quit$|copyright|credits|license)', re.I)
# modules that may be called with made-up arguments: pure functions of their arguments
PURE = {'math', 'random', 're', 'json', 'collections', 'itertools', 'functools', 'operator', 'dataclasses', 'copy', 'heapq', 'bisect', 'string', 'statistics', 'datetime', 'enum', 'textwrap', 'types', 'fnmatch', 'inspect', 'fractions', 'abc', 'typing', 'csv'}

X = 'x = np.array([[1., 2.], [3., 4.]])\nv = np.array([1., 2., 3.])\n'
ARGS_NP = ['x', 'v', 'x, x', 'v, v', 'x, 1', 'v, 2', '(2, 2)', '3', '0.5', '[x, x]', 'x > 1', '', "x, axis=0"]
SETUP = {
    'numpy': 'import numpy as np\n' + X,
    'matplotlib': 'import numpy as np\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n',
    'pandas': "import pandas as pd\nimport numpy as np\nD = pd.DataFrame({'a': [1, 2, 3], 'b': [4., 5., 6.]})\ns = pd.Series([3, 1, 2])\nt = pd.Series(['Ab c', 'de'])\nG = D.groupby('a')\n",
    'pytorch': 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch.utils.data\nimport torch.distributions\nt = torch.tensor([[1., 2.], [3., 4.]])\nv = torch.tensor([1., 2., 3.])\n',
    'keras': 'import numpy as np\nimport keras\nfrom keras import layers\nx = np.array([[1., 2.], [3., 4.]], dtype="float32")\n',
    'tensorflow': 'import numpy as np\nimport tensorflow as tf\nx = tf.constant([[1., 2.], [3., 4.]])\nv = tf.constant([1., 2., 3.])\nd = tf.data.Dataset.range(4)\n',
    'sklearn': 'import numpy as np\nX = np.array([[0., 0.], [1., 1.], [8., 8.], [9., 9.]])\ny = np.array([0, 0, 1, 1])\n',
    'cv2': 'import numpy as np\nimport cv2\nimg = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]], np.uint8)\nf = img.astype(np.float32)\nc = np.stack([img, img, img], axis=2)\n',
    'python': '',
}


def calls(lib, module, g, name):
    """(setup line, expression) forms to try"""
    expr = g['expr']
    if lib == 'numpy':
        target = {'np': 'np', 'ndarray': 'x', 'np.random': 'np.random', 'np.linalg': 'np.linalg'}[module]
        if module == 'ndarray':
            return [f'x.{name}'] + [f'x.{name}({a})' for a in ['', '0', 'x', '1, 0', '(4,)', 'float']]
        return [f'{target}.{name}({a})' for a in (['x + 2 * np.eye(2)', 'x', 'x, v[:2]', 'x, 2', ''] if module == 'np.linalg' else ['size=3', '1., size=3', '1., 2., size=3', '3, size=3', '5', 'v'] if module == 'np.random' else ARGS_NP)] + [f'{target}.{name}']
    if lib == 'matplotlib':
        t = {'plt': 'plt', 'Axes': 'ax', 'Figure': 'fig'}[module]
        return [f'{t}.{name}({a})' for a in ['', '[1, 2], [3, 4]', '[1, 2, 3]', '1', '"a"', 'np.eye(2)', '0.5', '1, 2']] + [f'{t}.{name}']
    if lib == 'pandas':
        t = {'pd': 'pd', 'DataFrame': 'D', 'Series': 's', 'Series.str': 't.str', 'GroupBy': 'G'}[module]
        args = ['D', 's', 'D, D', '[D, D]', '[1, 2]'] if module == 'pd' else ['', '1', '"a"', '"b"', 'D', 's', "'sum'", "lambda v: v", '" "', '2']
        return [f'{t}.{name}({a})' for a in args] + [f'{t}.{name}']
    if lib == 'pytorch':
        t = {'torch': 'torch', 'Tensor': 't', 'torch.nn': 'nn', 'nn.Module': 'nn.Linear(2, 1)', 'nn.functional': 'F', 'nn.init': 'nn.init', 'nn.utils': 'nn.utils', 'torch.linalg': 'torch.linalg', 'torch.special': 'torch.special', 'torch.optim': 'torch.optim', 'optim.lr_scheduler': 'torch.optim.lr_scheduler', 'torch.autograd': 'torch.autograd', 'utils.data': 'torch.utils.data', 'distributions': 'torch.distributions'}[module]
        args = {'torch.nn': ['', '2', '2, 3', '1, 2, 3', '0.5'], 'torch.optim': ['[nn.Parameter(t)], lr=0.1'], 'optim.lr_scheduler': ['torch.optim.SGD([nn.Parameter(t)], lr=0.1), 1', 'torch.optim.SGD([nn.Parameter(t)], lr=0.1)', 'torch.optim.SGD([nn.Parameter(t)], lr=0.1), [1]', 'torch.optim.SGD([nn.Parameter(t)], lr=0.1), lambda e: 1'], 'distributions': ['torch.tensor(0.), torch.tensor(1.)', 'torch.tensor(0.5)', 'torch.tensor([0.5, 0.5])']}.get(module, ['t', 'v', 't, t', 'v, v', 't, 1', 'v, 1', '', '2', '2, 2', '(2, 2)', '1.5', '[t, t]', 't + 2 * torch.eye(2)'] if module != 'Tensor' else ['', '1', 't', '0', '2.0', '0, 1'])
        return [f'{t}.{name}({a})' for a in args] + [f'{t}.{name}']
    if lib == 'keras':
        t = expr if module != 'keras.Model' else 'm'
        pre = 'm = keras.Sequential([layers.Input((2,)), layers.Dense(1)])\n' if module == 'keras.Model' else ''
        args = {'keras.layers': ['', '2', '2, 3', '0.5', '(1, 1)'], 'keras.ops': ['x', 'x, x', 'x, 1', '', '(2, 2)', '2'], 'keras.activations': ['x'], 'keras.losses': ['', 'x, x'], 'keras.metrics': ['', 'x, x', '"m"'], 'keras.callbacks': [''], 'keras.initializers': ['', '1.'], 'keras.regularizers': ['', '0.1'], 'keras.optimizers': ['', '0.1']}.get(module, ['', 'x', '2', '[1, 2]'])
        return [pre + f'{t}.{name}({a})' for a in args] + [pre + f'{t}.{name}']
    if lib == 'tensorflow':
        t = {'tf.Tensor': 'x', 'tf.Variable': 'w', 'tf.data.Dataset': 'd'}.get(module, module)
        pre = 'w = tf.Variable([1., 2.])\n' if module == 'tf.Variable' else ''
        args = ['', '1', 'x', '2'] if module in ('tf.Tensor', 'tf.Variable') else ['', '2', '1', 'lambda a: a'] if module == 'tf.data.Dataset' else ['x', 'v', 'x, x', 'v, v', 'x, 1', '(2, 2)', '2', '', '[2, 2]']
        return [pre + f'{t}.{name}({a})' for a in args] + [pre + f'{t}.{name}']
    if lib == 'sklearn':
        return [f'{g["setup"]}{e}' for e in [f'M.{name}()', f'M.{name}(y, y)', f'M.{name}(X)', f'M.{name}(2)', f'M.{name}().fit(X, y)']]
    if lib == 'cv2':
        if name[0].isupper():
            return [f'cv2.{name}']
        return [f'cv2.{name}({a})' for a in ['img', 'f', 'c', 'img, img', 'f, f', 'img, 100, 255, cv2.THRESH_BINARY', 'img, (2, 2)', 'img, (3, 3)', 'img, 3', 'img, -1, np.ones((3, 3), np.float32)', 'c, cv2.COLOR_BGR2GRAY', 'img, 1', '3, 1.0', '(1, 1), 30, 1.0', 'cv2.MORPH_RECT, (3, 3)', 'img, np.ones((3, 3), np.uint8)']]
    # python
    if module == 'builtins':
        return [f'{name}({a})' for a in ['[1, 2, 3]', '-1.5', '"a"', '65', '1, 2', '(1, 2)', '""']] + [name]
    if module in ('str', 'list', 'dict', 'set', 'tuple', 'int', 'float'):
        return [f'{expr}.{name}{a}' for a in ['()', '("a")', '(1)', "('a', 'b')", '([1])', "(['a'])", '']]
    if module not in PURE:
        return []
    pre = g['setup']
    return [f'{pre}M.{name}({a})' for a in ['2.5', '4, 2', '[3, 1, 2]', '"a b"', '"a", "a b"', '""', '', '[1, 2], 2']] + [f'{pre}M.{name}']


class Timeout(BaseException):
    pass


def on_alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, on_alarm)
top = os.getcwd()
tmp = tempfile.mkdtemp()
os.chdir(tmp)


def clean(s):
    return re.sub(r' at 0x[0-9a-f]+', '', s).rstrip('\n')


def run(src):
    buf = io.StringIO()
    scope = {'__name__': '__main__'}
    signal.setitimer(signal.ITIMER_REAL, 3)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(src, '<p>', 'exec'), scope)
        return clean(buf.getvalue()), scope
    except (Exception, Timeout, SystemExit):
        return None, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


# Some calls kill the process (torch.FileCheck) or close stdout (a number taken for a file descriptor), so each module
# runs in its own process: results go line by line to a file, and after a crash the next process starts one name later.
def quiet_crashes():
    """A native crash (OpenCV's seamless-cloning functions on a 4x4 image segfault) ends this process with _exit
    instead: the parent goes on one name later either way, and macOS shows no "Python quit unexpectedly" dialog."""
    import ctypes
    libc = ctypes.CDLL(None)
    libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    for sig in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT, signal.SIGILL, signal.SIGFPE):
        libc.signal(sig, ctypes.cast(libc._exit, ctypes.c_void_p))


if os.environ.get('PROBE_MODULE'):
    module = os.environ['PROBE_MODULE']
    start = int(os.environ['PROBE_START'])
    g = real['groups'][module]
    lib = g['lib']
    head = SETUP[lib]
    state = open(os.environ['PROBE_STATE'], 'w')
    sink = open(os.environ['PROBE_SINK'], 'a')
    for i, name in enumerate(g['names']):
        if i < start or UNSAFE.search(name):
            continue
        state.seek(0)
        state.write(str(i) + '   ')
        state.flush()
        quiet_crashes()   # again for each name: an import (TensorFlow's) may install its own handlers
        kept = 0
        for form in calls(lib, module, g, name):
            lines = form.split('\n')
            pre, expr = '\n'.join(lines[:-1]), lines[-1]
            pre = pre + '\n' if pre else ''
            for show in (f'print({expr})', f'r = {expr}\nprint(getattr(r, "shape", type(r).__name__))'):
                src = f'{head}{pre}{show}\n'
                a, _ = run(src)
                if a is None or a == '':
                    break
                if show.startswith('print(') and ('object' in a or (a[:1] == '<' and '>' in a)) and 'tf.Tensor' not in a:
                    continue   # an address or a class's repr says nothing: show the type instead
                b, _ = run(src)
                if a != b:
                    continue   # depends on random numbers: show the shape instead
                sink.write(json.dumps({'module': module, 'name': name, 'src': src, 'out': a}, ensure_ascii=False) + '\n')
                sink.flush()
                kept += 1
                break
            if kept >= 2:
                break
    state.seek(0)
    state.write('done')
    state.flush()
    os._exit(0)

os.chdir(top)
shutil.rmtree(tmp, ignore_errors=True)
import subprocess  # noqa: E402
sink = out_path + '.jsonl'
open(sink, 'w').close()
state = out_path + '.state'
script = str(pathlib.Path(__file__).resolve())
for module, g in real['groups'].items():
    if g['lib'] not in libs:
        continue
    start = 0
    for attempt in range(40):
        open(state, 'w').close()
        env = {**os.environ, 'PROBE_MODULE': module, 'PROBE_START': str(start), 'PROBE_STATE': os.path.abspath(state), 'PROBE_SINK': os.path.abspath(sink)}
        try:
            subprocess.run([sys.executable, script, out_path, *libs], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        except subprocess.TimeoutExpired:
            pass
        at = open(state).read().strip()
        if at == 'done' or not at:
            break
        print(module, 'stopped at', g['names'][int(at)], flush=True)
        start = int(at) + 1
    print(module, 'done', flush=True)
rows = [json.loads(l) for l in open(sink)]
pathlib.Path(out_path).write_text(json.dumps(rows, ensure_ascii=False))
print('kept', len(rows))
