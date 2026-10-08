import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'

// Expected stdout produced by CPython 3.10 (+ NumPy 2.2) on the same source.
const CASES: Record<string, { src: string; stdout: string }> = {
  "loops": {
    "src": "total = 0\nfor i in range(1, 6):\n    if i % 2 == 0:\n        continue\n    total += i\nprint(total)\nn = 0\nwhile True:\n    n += 1\n    if n >= 4:\n        break\nprint(n)\nfor i, ch in enumerate('abc', start=1):\n    print(i, ch, end=' ')\nprint()\nfor a, b in zip([1, 2, 3], [10, 20]):\n    print(a * b)\n",
    "stdout": "9\n4\n1 a 2 b 3 c \n10\n40\n"
  },
  "funcs": {
    "src": "def f(x, y=2):\n    \"\"\"multiply\"\"\"\n    return x * y\nprint(f(3), f(3, y=5), f(x=4))\nsq = lambda t: t ** 2\nprint(list(map(sq, [1, 2, 3])))\ndef make(k):\n    def add(x):\n        return x + k\n    return add\nprint(make(10)(5))\ndef fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\nprint(fact(6))\n",
    "stdout": "6 15 8\n[1, 4, 9]\n15\n720\n"
  },
  "comps": {
    "src": "xs = [x * x for x in range(6) if x % 2 == 0]\nprint(xs)\nd = {k: v for k, v in zip('abc', [1, 2, 3])}\nprint(d, d['b'], len(d))\nprint(sum(x for x in range(5)))\nprint([[r * 3 + c for c in range(3)] for r in range(2)])\n",
    "stdout": "[0, 4, 16]\n{'a': 1, 'b': 2, 'c': 3} 2 3\n10\n[[0, 1, 2], [3, 4, 5]]\n"
  },
  "fstrings": {
    "src": "loss = 0.123456\nacc = 0.9\nprint(f\"epoch {3:>3} loss={loss:.4f} acc={acc:.1%}\")\nname = 'kNN'\nprint(f'{name!r} {name:>6}|{7:03d}|{1234567:,}|{2.5e-5:.2e}|{3.0}')\nprint('{} + {} = {:.2f}'.format(1, 2, 3))\n",
    "stdout": "epoch   3 loss=0.1235 acc=90.0%\n'kNN'    kNN|007|1,234,567|2.50e-05|3.0\n1 + 2 = 3.00\n"
  },
  "classes": {
    "src": "class Animal:\n    sound = '...'\n    def __init__(self, name):\n        self.name = name\n    def speak(self):\n        return f'{self.name} says {self.sound}'\nclass Dog(Animal):\n    sound = 'woof'\n    def __init__(self, name, age):\n        super().__init__(name)\n        self.age = age\n    def __repr__(self):\n        return f'Dog({self.name!r}, {self.age})'\nd = Dog('Rex', 3)\nprint(d.speak())\nprint(d)\nprint(isinstance(d, Animal), isinstance(d, Dog), type(d).__name__)\nclass Counter:\n    def __init__(self):\n        self.n = 0\n    def __call__(self):\n        self.n += 1\n        return self.n\nc = Counter()\nc(); c()\nprint(c())\n",
    "stdout": "Rex says woof\nDog('Rex', 3)\nTrue True Dog\n3\n"
  },
  "dicts": {
    "src": "d = {'a': 1}\nd['b'] = 2\nd.update(c=3)\nfor k, v in d.items():\n    print(k, v)\nprint(d.get('z', 0), 'a' in d, list(d.keys()))\ndel d['a']\nprint(d)\ncounts = {}\nfor w in 'the cat the hat'.split():\n    counts[w] = counts.get(w, 0) + 1\nprint(counts)\nprint(sorted(counts, key=counts.get, reverse=True)[0])\nprint(max([3, 1, 4]), min(5, 2, 8), round(2.5), round(3.14159, 2))\n",
    "stdout": "a 1\nb 2\nc 3\n0 True ['a', 'b', 'c']\n{'b': 2, 'c': 3}\n{'the': 2, 'cat': 1, 'hat': 1}\nthe\n4 2 2 3.14\n"
  },
  "numpy_loop": {
    "src": "import numpy as np\nX = np.array([[1, 2], [3, 4], [5, 6]])\nfor row in X:\n    print(row.sum())\ndists = [np.sqrt(((X[i] - X[0]) ** 2).sum()) for i in range(len(X))]\nprint([round(float(d), 3) for d in dists])\n",
    "stdout": "3\n7\n11\n[0.0, 2.828, 5.657]\n"
  }
}

describe('Python language subset matches CPython', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(name, () => {
      const r = runPython(c.src)
      expect(r.error).toBeNull()
      expect(r.stdout).toBe(c.stdout)
    })
  }
})

describe('sandbox limits and errors', () => {
  it('an endless loop stops with a TimeoutError instead of freezing the page', () => {
    const r = runPython('n = 0\nwhile True:\n    n += 1', { timeBudget: 50 })
    expect(r.error?.type).toBe('TimeoutError')
    expect([2, 3]).toContain(r.error?.line)
  })

  it('errors inside a function report the line inside it', () => {
    const r = runPython('def f(x):\n    y = x + 1\n    return z\nf(1)')
    expect(r.error).toMatchObject({ type: 'NameError', line: 3, message: "name 'z' is not defined" })
    expect(runPython('def f(a, b):\n    return a\nf(1)').error?.message).toBe("f() missing 1 required positional argument: 'b'")
    expect(runPython('break').error?.message).toBe("'break' outside loop")
  })

  it('unknown modules name what the sandbox has', () => {
    expect(runPython('import sklearn').error).toMatchObject({ type: 'ModuleNotFoundError', message: "No module named 'sklearn' (this sandbox has numpy, math, random)" })
  })

  it('a loop body records traces for its first iterations only', () => {
    const r = runPython('import numpy as np\na = np.arange(6)\nfor i in range(50):\n    b = a[1:3]\n    s = np.sum(a)')
    expect(r.error).toBeNull()
    expect(r.traces.length).toBe(2)
    expect(r.calls.filter((c) => c.api === 'np.sum').length).toBe(2)
  })
})
