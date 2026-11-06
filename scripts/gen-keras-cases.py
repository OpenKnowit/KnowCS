#!/usr/bin/env python3
"""Runs scripts/keras-cases.txt on real TensorFlow and Keras and freezes what each program printed.

src/lib/pyKerasCases.test.ts then checks that the sandbox prints the same. Each program runs under two seeds: when the
two runs differ the program depends on random weights, and only the form of its output is compared. Run it with a
Python that has tensorflow:

    python3 scripts/gen-keras-cases.py
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
import types
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.simplefilter('ignore')

import keras  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

matplotlib.use('Agg')
here = pathlib.Path(__file__).parent


def programs():
    title, block = None, []
    for line in (here / 'keras-cases.txt').read_text().split('\n'):
        if line.startswith('### '):
            if title is not None:
                yield title, '\n'.join(block).rstrip('\n') + '\n'
            title, block = line[4:], []
        else:
            block.append(line)
    if title is not None:
        yield title, '\n'.join(block).rstrip('\n') + '\n'


def clean(s):
    """the same text however fast the machine is: no colours, addresses, timings or the bar's earlier frames"""
    s = re.sub(r'\x1b\[[0-9;]*m', '', s).replace('\x08', '')
    s = re.sub(r'^[^\n]*\r(?=[^\n])', '', s, flags=re.M).replace('\r', '')
    s = re.sub(r' at 0x[0-9a-f]+', '', s)
    s = re.sub(r' \d+s \d+[mu]s/step', '', s)
    s = re.sub(r' - \d+s - \d+[mu]?s/step', '', s)
    s = re.sub(r'^/[^\n]*\n', '', s, flags=re.M)
    return re.sub(r'[ \t\r]+$', '', s, flags=re.M)


class Timeout(BaseException):
    pass


def on_alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, on_alarm)
top = os.getcwd()


def run(src, seed):
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    keras.utils.clear_session()
    keras.utils.set_random_seed(seed)
    np.set_printoptions(precision=8, threshold=1000, edgeitems=3, linewidth=75, suppress=False)
    buf, err = io.StringIO(), None
    module, real = types.ModuleType('__main__'), sys.modules['__main__']
    sys.modules['__main__'] = module
    sys.argv = ['main.py']
    signal.alarm(40)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(src, '<p>', 'exec'), module.__dict__)
    except BaseException as e:
        err = {'type': type(e).__name__, 'message': clean(str(e))}
    finally:
        signal.alarm(0)
        sys.modules['__main__'] = real
        os.chdir(top)
        shutil.rmtree(tmp, ignore_errors=True)
    return clean(buf.getvalue()), err


cases = []
for title, src in programs():
    a, e1 = run(src, 1)
    b, e2 = run(src, 2)
    cases.append({'title': title, 'src': src, 'out': a, 'error': e1, 'random': a != b or e1 != e2})
(here.parent / 'src/lib/pyKerasCases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=0) + '\n')
print(len(cases), 'programs on TensorFlow', tf.__version__, 'Keras', keras.__version__, sum(c['random'] for c in cases), 'depend on random weights')
