#!/usr/bin/env python3
"""Runs each program in python-cases.txt with real CPython and freezes what it printed.

src/lib/minipyCases.test.ts then checks that the sandbox prints exactly the same.
Run with no arguments after editing python-cases.txt.
"""
import json
import pathlib
import subprocess
import sys

here = pathlib.Path(__file__).parent
cases = []
for block in (here / 'python-cases.txt').read_text().split('### ')[1:]:
    name, code = block.split('\n', 1)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    # sys.exit() fails with no message, a warning prints a message and succeeds
    last = r.stderr.strip().splitlines()[-1] if r.returncode and r.stderr.strip() else ''
    cases.append({'name': name, 'code': code, 'out': r.stdout, 'error': last.split(':')[0] if last else None})
(here.parent / 'src/lib/minipyCases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1) + '\n')
print(f'{len(cases)} programs on Python {sys.version.split()[0]}')
