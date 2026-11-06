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
  // ---- the everyday APIs; each one was run on the real library
  {
    id: 'str_methods', cat: 'text', label: 'str.split / strip / join',
    code: `s = '  Hello, World  '
print(s.strip(), s.lstrip(), s.rstrip(), sep='|')
print(s.lower(), s.upper(), s.title(), s.swapcase(), sep='|')
print('a,b,,c'.split(','), 'a b  c'.split(), 'a.b.c'.rsplit('.', 1), 'x=1'.partition('='))
print('-'.join(['2026', '11', '06']), 'hello'.replace('l', 'L', 1))
print('hello'.startswith('he'), 'hello'.endswith(('x', 'lo')), 'hello'.find('l'), 'hello'.find('z'), 'hello'.count('l'))
print('42'.isdigit(), 'ab'.isalpha(), 'a1'.isalnum(), ' '.isspace(), '7'.zfill(3), 'x'.center(5, '*'), 'x'.ljust(3, '.'), 'x'.rjust(3, '.'))
print('report.csv'.removesuffix('.csv'), 'a\\nb\\n'.splitlines())`,
  },
  {
    id: 'list_methods', cat: 'data', label: 'list.append / sort / pop',
    code: `a = [3, 1, 2]
a.append(5)
a.extend([4, 0])
a.insert(0, 9)
print(a, a.pop(), a.pop(0), a)
a.remove(1)
print(a, a.index(2), a.count(2))
a.sort()
print(a)
a.sort(key=lambda x: -x)
print(a)
a.reverse()
print(a, sorted(a, reverse=True), a[::-1], a[1:3])
b = a.copy()
b.clear()
print(a, b, [x * x for x in a if x % 2])`,
  },
  {
    id: 'dict_methods', cat: 'data', label: 'dict.get / items / setdefault',
    code: `d = {'a': 1, 'b': 2}
print(d['a'], d.get('z'), d.get('z', 0), 'a' in d, len(d))
print(list(d.keys()), list(d.values()), list(d.items()))
d.update(c=3)
d.setdefault('e', []).append(1)
print(d, d.pop('a'), d)
print({k: v * 2 for k, v in d.items() if k != 'e'})
print(dict(zip('xy', [1, 2])), dict.fromkeys('ab', 0), {**d, 'b': 9}['b'])
counts = {}
for w in 'a b a'.split():
    counts[w] = counts.get(w, 0) + 1
print(counts)
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(k, v)`,
  },
  {
    id: 'set_methods', cat: 'data', label: 'set union / intersection',
    code: `a = {1, 2, 3}
b = {2, 3, 4}
print(a | b, a & b, a - b, a ^ b)
print(a.union(b), a.intersection(b), a.difference(b), a.symmetric_difference(b))
print({1, 2} <= a, a.issubset({1, 2, 3, 4}), a.issuperset({1}), a.isdisjoint({9}))
a.add(5)
a.discard(99)
a.remove(1)
print(a, 3 in a, len(a), sorted(set('banana')), frozenset([1]))
print(list(dict.fromkeys([3, 1, 3, 2])))`,
  },
  {
    id: 'builtins_iter', cat: 'data', label: 'enumerate / zip / map / filter',
    code: `names = ['ann', 'bob', 'cat']
marks = [90, 72, 85]
for i, n in enumerate(names, start=1):
    print(i, n)
for n, m in zip(names, marks):
    print(n, m)
print(list(map(str.upper, names)), list(filter(lambda m: m > 80, marks)))
print(sum(marks), min(marks), max(marks), max(names, key=len), sorted(zip(marks, names))[0])
print(any(m > 89 for m in marks), all(m > 89 for m in marks), list(reversed(names)), list(range(0, 10, 3)))
it = iter(names)
print(next(it), next(it), next(it), next(it, 'done'))
print(abs(-2), round(2.567, 1), divmod(7, 2), pow(2, 5), bin(5), hex(255), chr(65), ord('A'), int('12'), float('1.5'), len(names))`,
  },
  {
    id: 'exceptions', cat: 'script', label: 'try / except / raise',
    code: `def parse(text):
    try:
        return int(text)
    except ValueError as e:
        print('bad number:', e)
        return None
    finally:
        print('checked', repr(text))

print(parse('7'), parse('x'))
for f in [lambda: 1 / 0, lambda: [][1], lambda: {}['k'], lambda: None.x, lambda: open('nope.txt')]:
    try:
        f()
    except (ZeroDivisionError, IndexError, KeyError) as e:
        print(type(e).__name__, e)
    except Exception as e:
        print('other:', type(e).__name__)

class TooBig(ValueError):
    pass
try:
    raise TooBig('limit is 10')
except ValueError as e:
    print(repr(e), isinstance(e, ValueError))`,
  },
  {
    id: 'classes', cat: 'script', label: 'class / property / staticmethod',
    code: `class Account:
    count = 0

    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance = balance
        Account.count += 1

    @property
    def balance(self):
        return self._balance

    @classmethod
    def empty(cls, owner):
        return cls(owner)

    @staticmethod
    def fee(n):
        return n * 0.01

    def __repr__(self):
        return f'Account({self.owner!r}, {self._balance})'

class Savings(Account):
    def __init__(self, owner, balance=0, rate=0.02):
        super().__init__(owner, balance)
        self.rate = rate

a = Account('ann', 10)
b = Savings.empty('bob')
print(a, b, a.balance, Account.fee(100), Account.count)
print(isinstance(b, Account), issubclass(Savings, Account), hasattr(a, 'owner'), getattr(a, 'nope', None), callable(a))
setattr(a, 'tag', 1)
print(vars(a), type(b).__name__)`,
  },
  {
    id: 'math_random', cat: 'script', label: 'math / random',
    code: `import math
import random

print(math.pi, math.e, math.inf, math.nan)
print(math.sqrt(2), math.floor(2.7), math.ceil(2.1), math.trunc(-2.7), math.fabs(-1))
print(math.log(math.e), math.log2(8), math.log10(100), math.exp(0), math.pow(2, 3))
print(math.sin(0), math.cos(0), math.atan2(1, 1), math.degrees(math.pi), math.radians(180), math.hypot(3, 4))
print(math.factorial(5), math.comb(5, 2), math.perm(5, 2), math.gcd(12, 18), math.lcm(4, 6), math.prod([2, 3]), math.fsum([.1] * 10))
print(math.isnan(math.nan), math.isinf(math.inf), math.isclose(0.1 + 0.2, 0.3), math.dist([0, 0], [3, 4]))
random.seed(0)
print(0 <= random.random() < 1, 1 <= random.randint(1, 6) <= 6, random.choice('abc') in 'abc')
x = [1, 2, 3, 4]
random.shuffle(x)
print(sorted(x), len(random.sample(x, 2)), 0 <= random.uniform(0, 1) <= 1)`,
  },
  {
    id: 'iter_more', cat: 'data', label: 'zip_longest / takewhile / pairwise',
    code: `import itertools as it

print(list(it.zip_longest('ab', [1, 2, 3], fillvalue='-')))
print(list(it.takewhile(lambda x: x < 3, [1, 2, 5, 1])), list(it.dropwhile(lambda x: x < 3, [1, 2, 5, 1])))
print(list(it.filterfalse(lambda x: x % 2, range(5))), list(it.compress('abc', [1, 0, 1])))
print(list(it.pairwise([1, 3, 6])), list(it.starmap(pow, [(2, 3), (3, 2)])))
print(list(it.combinations_with_replacement('ab', 2)))
print(list(it.islice(it.cycle('ab'), 5)), list(it.repeat(0, 3)))
a, b = it.tee([1, 2])
print(list(a), list(b))`,
  },
  {
    id: 'functools_more', cat: 'data', label: 'cache / wraps / cached_property',
    code: `from functools import cache, cached_property, reduce, total_ordering, wraps

@cache
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
print(fib(50), fib.cache_info().hits > 0)

def shout(f):
    @wraps(f)
    def inner(*a, **k):
        return f(*a, **k).upper()
    return inner

@shout
def hi(name):
    return f'hi {name}'
print(hi('ann'), hi.__name__)

class Data:
    @cached_property
    def size(self):
        print('computing')
        return 3
d = Data()
print(d.size, d.size)

@total_ordering
class V:
    def __init__(self, n):
        self.n = n
    def __eq__(self, o):
        return self.n == o.n
    def __lt__(self, o):
        return self.n < o.n
print(V(1) < V(2), V(1) >= V(2), reduce(lambda a, b: a * b, [1, 2, 3, 4]))`,
  },
  {
    id: 'regex_more', cat: 'text', label: 're.compile / match / fullmatch',
    code: `import re

p = re.compile(r'(\\d{4})-(\\d{2})')
m = p.search('due 2026-11 ok')
print(m.group(), m.group(1), m.groups(), m.start(), m.end(), m.span())
print(re.match(r'\\d+', 'a1'), re.match(r'\\d+', '1a').group(), re.fullmatch(r'\\d+', '1a'))
for f in re.finditer(r'\\w+', 'to be'):
    print(f.group(), f.span())
print(re.findall(r'(\\w)=(\\d)', 'a=1 b=2'), re.split(r'[,;]\\s*', 'a, b;c'))
print(re.sub(r'\\d+', '#', 'a1 b22'), re.subn(r'\\d', '#', 'a1 b2'), re.sub(r'(\\w+)@', r'<\\1>', 'ann@x'))
print(re.findall('hello', 'Hello hello', re.IGNORECASE), re.findall(r'^b', 'a\\nb', re.M), re.escape('a.b'))
m = re.search(r'(?P<y>\\d+)/(?P<m>\\d+)', '2026/11')
print(m.group('y'), m.groupdict())`,
  },
  {
    id: 'os_paths', cat: 'files', label: 'os / os.path',
    code: `import os
import os.path as op

print(op.join('runs', 'a.txt'), op.basename('runs/a.txt'), op.dirname('runs/a.txt'), op.splitext('a.tar.gz'), op.split('runs/a.txt'))
os.makedirs('runs/x', exist_ok=True)
with open('runs/a.txt', 'w') as f:
    f.write('hi')
print(os.listdir('runs'), op.exists('runs'), op.isdir('runs'), op.isfile('runs/a.txt'), op.getsize('runs/a.txt'))
for root, dirs, files in os.walk('runs'):
    print(root, dirs, files)
os.rename('runs/a.txt', 'runs/b.txt')
os.remove('runs/b.txt')
os.rmdir('runs/x')
print(os.listdir('runs'), op.normpath('a/./b/../c'), op.isabs('/x'))
os.environ['MODE'] = 'test'
print(os.environ['MODE'], os.getenv('MODE'), os.getenv('NOPE', 'default'), os.sep)`,
  },
  {
    id: 'sys_time', cat: 'script', label: 'sys / time',
    code: `import sys
import time

print(sys.version_info[:2] >= (3, 8), isinstance(sys.argv, list), type(sys.modules).__name__)
print('to stderr', file=sys.stderr)
sys.stdout.write('no newline ')
print(sys.getrecursionlimit() > 100, sys.getsizeof(1) > 0)
start = time.perf_counter()
time.sleep(0.01)
print(time.perf_counter() - start >= 0, time.time() > 0, time.monotonic() >= 0, time.process_time() >= 0)
try:
    sys.exit(3)
except SystemExit as e:
    print('exit code', e.code)`,
  },
  {
    id: 'copy_io', cat: 'data', label: 'copy.deepcopy / io.StringIO',
    code: `import copy
import io

a = [[1, 2], [3]]
b = a[:]
c = copy.copy(a)
d = copy.deepcopy(a)
a[0].append(9)
print(b, c, d)
s = io.StringIO()
s.write('line 1\\n')
print('line 2', file=s)
print(repr(s.getvalue()))
s.seek(0)
print(s.readline(), s.read())`,
  },
  {
    id: 'typing_abc', cat: 'script', label: 'typing / abc',
    code: `from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

def mean(xs: List[float], scale: Optional[float] = None) -> float:
    return sum(xs) / len(xs) * (scale or 1)

def apply(f: Callable[[int], int], xs: Iterable[int]) -> Tuple[int, ...]:
    return tuple(f(x) for x in xs)

table: Dict[str, Union[int, str]] = {'a': 1}
print(mean([1., 2.]), mean([1.], 2.), apply(abs, [-1]), table)

class Shape(ABC):
    @abstractmethod
    def area(self): ...

class Square(Shape):
    def __init__(self, s):
        self.s = s
    def area(self):
        return self.s ** 2

print(Square(2).area())
try:
    Shape()
except TypeError as e:
    print('TypeError', e)`,
  },
  {
    id: 'strings_const', cat: 'text', label: 'string / f-string',
    code: `import string

print(string.ascii_lowercase[:3], string.ascii_uppercase[:3], string.digits[:3], string.punctuation[:3])
x, pi, n = 'ab', 3.14159, 1234567
print(f'{x:>5}|{x:<5}|{x:^5}|{pi:.2f}|{pi:8.3f}|{n:,}|{0.25:.0%}|{255:x}|{5:03d}|{pi:e}')
print(f'{x!r} {x=} {pi=:.1f}')
print('{} {}'.format(1, 2), '{1} {0}'.format(1, 2), '{a}'.format(a=1), '%d %s %.2f' % (1, 'a', pi))`,
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
