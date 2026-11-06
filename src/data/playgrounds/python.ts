// Python 实验台：标准库的文件、数据结构、文本和脚本工具；文件夹一开始是空的
import { STDLIB_MORE } from '../../lib/pyStdlibMore'
import type { PlayConfig, PlayEntry } from './types'

const ENTRIES: PlayEntry[] = [
  {
    id: 'pathlib', cat: 'files', label: 'pathlib.Path',
    code: `from pathlib import Path

run = Path('runs') / 'exp1'
run.mkdir(parents=True, exist_ok=True)
(run / 'notes.txt').write_text('hello\\nworld\\n')
print(run, run.parent, run.name, run.exists())
f = run / 'notes.txt'
print(f.suffix, f.stem, f.read_text().split())
print(sorted(p.name for p in Path('runs').rglob('*')))`,
  },
  {
    id: 'open_csv', cat: 'files', label: 'open / csv',
    code: `import csv

with open('marks.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['name', 'mark'])
    w.writerows([['Ann', 90], ['Bo', 75]])
print(open('marks.csv').read())
with open('marks.csv') as f:
    rows = list(csv.DictReader(f))
print(rows, sum(int(r['mark']) for r in rows))`,
  },
  {
    id: 'json', cat: 'files', label: 'json',
    code: `import json

cfg = {'lr': 0.1, 'layers': [32, 16], 'name': 'cnn', 'bn': True, 'seed': None}
text = json.dumps(cfg, indent=2, sort_keys=True)
print(text)
print(json.loads(text) == cfg, type(json.loads('[1, 2.5]')[1]).__name__)`,
  },
  {
    id: 'pickle', cat: 'files', label: 'pickle',
    code: `import pickle
from dataclasses import dataclass

@dataclass
class Cfg:
    lr: float
    tags: list

with open('cfg.pkl', 'wb') as f:
    pickle.dump(Cfg(0.1, ['a']), f)
with open('cfg.pkl', 'rb') as f:
    back = pickle.load(f)
print(back, back.tags)
print(pickle.loads(pickle.dumps({'x': (1, 2)})))`,
  },
  {
    id: 'tempfile', cat: 'files', label: 'tempfile / shutil / glob',
    code: `import glob, os, shutil, tempfile

with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, 'a.txt'), 'w').write('A')
    shutil.copy(os.path.join(d, 'a.txt'), os.path.join(d, 'b.txt'))
    os.makedirs(os.path.join(d, 'sub'))
    shutil.move(os.path.join(d, 'b.txt'), os.path.join(d, 'sub', 'b.txt'))
    print(sorted(os.listdir(d)))
    print([os.path.relpath(p, d) for p in glob.glob(os.path.join(d, '**', '*.txt'), recursive=True)])
    kept = d
print(os.path.exists(kept))`,
  },
  {
    id: 'collections', cat: 'data', label: 'collections',
    code: `from collections import Counter, defaultdict, deque, namedtuple

print(Counter('mississippi').most_common(2))
groups = defaultdict(list)
for word in ['apple', 'bob', 'avocado']:
    groups[word[0]].append(word)
print(dict(groups))
q = deque([1, 2, 3], maxlen=3)
q.append(4)
print(q, q[0])
Point = namedtuple('Point', 'x y')
print(Point(1, 2), Point(1, 2).y)`,
  },
  {
    id: 'itertools', cat: 'data', label: 'itertools',
    code: `import itertools as it

print(list(it.product('ab', [1, 2])))
print(list(it.combinations('abc', 2)))
print(list(it.permutations([1, 2, 3], 2))[:3])
print(list(it.chain([1], [2, 3])), list(it.accumulate([1, 2, 3])))
print(list(it.islice(it.count(10, 5), 3)))
print({k: list(g) for k, g in it.groupby('aabbc')})`,
  },
  {
    id: 'functools', cat: 'data', label: 'functools',
    code: `from functools import lru_cache, partial, reduce

@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print(fib(30), fib.cache_info().hits)
square = partial(pow, exp=2)
print(square(5), reduce(lambda a, b: a * b, [1, 2, 3, 4]))`,
  },
  {
    id: 'dataclass', cat: 'data', label: 'dataclasses / enum',
    code: `from dataclasses import dataclass, field, asdict, replace
from enum import Enum, auto

class Mode(Enum):
    TRAIN = auto()
    EVAL = auto()

@dataclass
class Config:
    lr: float = 0.1
    mode: Mode = Mode.TRAIN
    tags: list = field(default_factory=list)

c = Config(tags=['a'])
print(c, asdict(c)['lr'], replace(c, lr=0.5).lr, c == Config(tags=['a']))
print(Mode.EVAL, Mode.EVAL.name, Mode.EVAL.value, list(Mode), Mode['TRAIN'])`,
  },
  {
    id: 'heapq', cat: 'data', label: 'heapq / bisect',
    code: `import bisect, heapq

h = [5, 1, 4]
heapq.heapify(h)
heapq.heappush(h, 0)
print(heapq.heappop(h), heapq.nsmallest(2, h), heapq.nlargest(1, h))
s = [1, 3, 5]
print(bisect.bisect_left(s, 3), bisect.bisect_right(s, 3))
bisect.insort(s, 4)
print(s)`,
  },
  {
    id: 'regex', cat: 'text', label: 're',
    code: `import re

s = 'lr=0.01, epochs=20; lr=0.1'
print(re.findall(r'lr=([\\d.]+)', s))
print(re.sub(r'\\d+', '#', 'a1b22'), re.split(r'[,;]\\s*', s))
m = re.search(r'(?P<key>\\w+)=(?P<val>\\d+)', 'epochs=20')
print(m.group('key'), m.group(2), m.span(), m.groupdict())`,
  },
  {
    id: 'format', cat: 'text', label: 'format / textwrap',
    code: `import string, textwrap

x = 3.14159
print(f'{x:.2f} {x:8.3f} {x:e} {1234567:,} {0.25:.0%} {42:05d} {255:x}')
print(f'{"left":<8}|{"right":>8}|{"mid":^8}|')
print(textwrap.fill('The quick brown fox jumps over the lazy dog', 20))
print(textwrap.shorten('The quick brown fox', 12), string.digits)`,
  },
  {
    id: 'argparse', cat: 'script', label: 'argparse',
    code: `import argparse

p = argparse.ArgumentParser(prog='train', description='Train a model')
p.add_argument('--lr', type=float, default=0.01, help='learning rate')
p.add_argument('--opt', choices=['sgd', 'adam'], default='adam')
p.add_argument('--verbose', action='store_true')
print(p.parse_args([]))
print(p.parse_args(['--lr', '0.5', '--verbose']))
print(p.format_help())`,
  },
  {
    id: 'contextlib', cat: 'script', label: 'with / contextlib',
    code: `import contextlib, io
from contextlib import contextmanager

@contextmanager
def tag(name):
    print('<' + name + '>')
    yield name.upper()
    print('</' + name + '>')

with tag('b') as t:
    print(t)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    print('captured')
print(repr(out.getvalue()))
with contextlib.suppress(ZeroDivisionError):
    1 / 0`,
  },
  {
    id: 'datetime', cat: 'script', label: 'datetime / time',
    code: `from datetime import date, datetime, timedelta

t = datetime(2024, 2, 28, 13, 5, 9)
print(t + timedelta(days=2), t.strftime('%Y-%m-%d %a'), t.isoformat())
print(date(2024, 3, 1) - date(2024, 2, 1), date(2024, 2, 28) < date(2024, 3, 1))
print(repr(timedelta(days=2, seconds=5)), timedelta(hours=36).days)`,
  },
  {
    id: 'numbers', cat: 'script', label: 'statistics / fractions',
    code: `import math, statistics
from fractions import Fraction as F

print(statistics.mean([1, 2, 4]), statistics.median([1, 3, 2, 4]), statistics.stdev([1, 2, 3, 4]))
print(F(1, 3) + F(1, 6), F('0.25'), float(F(1, 3)))
print(math.gcd(12, 18), math.comb(5, 2), math.isclose(0.1 + 0.2, 0.3), 0.1 + 0.2 == 0.3)`,
  },
  {
    id: 'inspect', cat: 'script', label: 'inspect / types',
    code: `import inspect, types

def f(a, b=2, *rest, c=3, **kw) -> int:
    """Add things."""
    return a

s = inspect.signature(f)
print(s, list(s.parameters), s.parameters['b'].default)
print(inspect.getdoc(f), inspect.isfunction(f), inspect.isclass(int))
ns = types.SimpleNamespace(lr=0.1, epochs=2)
print(ns, ns.lr, vars(ns))`,
  },
  {
    id: 'eval_exec', cat: 'script', label: 'eval / exec',
    code: `x = 3
print(eval('x * 2 + 1'))
scope = {}
exec('y = 10\\nz = y + 1', scope)
print(scope['z'])
try:
    eval('1 +')
except SyntaxError as e:
    print(type(e).__name__)`,
  },
]

export const PYTHON_PLAYGROUND: PlayConfig = {
  id: 'python',
  libs: [STDLIB_MORE],
  maxSize: 2_000_000,
  timeBudget: 2000,
  debounce: 400,
  cats: ['files', 'data', 'text', 'script'],
  entries: ENTRIES,
  prelude: '',
  callApis: [],
}
