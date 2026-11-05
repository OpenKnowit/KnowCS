#!/usr/bin/env python3
"""Runs scripts/torch-cases.txt on real PyTorch and freezes what each program printed.

src/lib/pyTorchCases.test.ts then checks that the sandbox prints the same. Run it with a Python that has torch:

    ~/repo/minimind/.venv/bin/python scripts/gen-torch-cases.py
"""
import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import types
import warnings

import torch

here = pathlib.Path(__file__).parent
HEAD = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
DEFAULTS = dict(precision=4, threshold=1000, edgeitems=3, linewidth=80, sci_mode=None, profile='default')


def programs():
    block = None
    for line in (here / 'torch-cases.txt').read_text().split('\n'):
        if block is not None:
            if line == '>>>':
                yield '\n'.join(block)
                block = None
            else:
                block.append(line)
        elif line == '<<<':
            block = []
        elif line and not line.startswith('#'):
            yield line


def with_repr(src):
    """the last statement's repr is printed, when it is an expression"""
    lines = src.split('\n')
    if len(lines) == 1:
        lines = [p.strip() for p in src.split('; ')]
    try:
        compile(lines[-1], '<last>', 'eval')
        lines[-1] = f'print(repr({lines[-1]}))'
    except SyntaxError:
        pass
    return HEAD + '\n'.join(lines) + '\n'


warnings.simplefilter('ignore')
torch.set_num_threads(1)
cases = []
seen = set()
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    for src in programs():
        code = with_repr(src)
        if code in seen:
            continue
        seen.add(code)
        buf = io.StringIO()
        error = None
        module = types.ModuleType('__main__')
        real_main = sys.modules['__main__']
        sys.modules['__main__'] = module
        torch.set_printoptions(profile='default', sci_mode=None)
        torch.manual_seed(0)
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, '<case>', 'exec'), module.__dict__)
        except BaseException as e:
            error = f'{type(e).__name__}: {e}'
        finally:
            sys.modules['__main__'] = real_main
            torch.set_grad_enabled(True)
        cases.append({'src': code, 'out': buf.getvalue(), 'error': error})
out = here.parent / 'src/lib/pyTorchCases.json'
with open(out, 'w') as f:
    json.dump({'torch': torch.__version__, 'cases': cases}, f, ensure_ascii=False, indent=0)
    f.write('\n')
print(f'{len(cases)} programs on torch {torch.__version__}')
