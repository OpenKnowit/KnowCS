#!/usr/bin/env python3
"""Runs each program of a curated file (scripts/api-coverage/curated/<lib>.txt) on the real library, each in its own
process and folder, and writes what it printed: python3 scripts/api-coverage/curated_real.py matplotlib out.json"""
import json, os, pathlib, subprocess, sys, tempfile

lib, out = sys.argv[1], sys.argv[2]
text = (pathlib.Path(__file__).parent / 'curated' / f'{lib}.txt').read_text()
sections = []
for block in text.split('\n### ')[1:]:
    head, _, src = block.partition('\n')
    sections.append({'keys': head.split(), 'src': src.strip() + '\n'})
env = {**os.environ, 'MPLBACKEND': 'Agg', 'TF_CPP_MIN_LOG_LEVEL': '3', 'PYTHONWARNINGS': 'ignore'}
for s in sections:
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, '-c', s['src']], cwd=d, env=env, capture_output=True, text=True, timeout=120)
    s['real'] = r.stdout
    s['realError'] = r.stderr.strip().splitlines()[-1] if r.returncode else None
json.dump(sections, open(out, 'w'), indent=1, ensure_ascii=False)
print(len(sections), 'programs,', sum(1 for s in sections if s['realError']), 'failed on the real library')
