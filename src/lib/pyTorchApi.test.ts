import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { TORCH } from './pyTorch'
import VALUES from './pyTorchApi.values.json'
import LAYERS from './pyTorchApi.layers.json'

/*
 * What in this file has been checked on real PyTorch (2.14.0), and how:
 * - pyTorchApi.values.json: each row holds the torch expression, the NumPy / SciPy / OpenCV expression it was first written
 *   from, the value and the dtype. The values and dtypes are now what real torch returns for the expression.
 * - pyTorchApi.layers.json: BatchNorm, the recurrent layers, attention, the optimisers and schedulers, with fixed weights;
 *   first computed in NumPy from the documented equations, then replaced by what real torch returns.
 * - the 'errors students meet' table and the printed layers come from real torch, and so do the full printed outputs.
 * - gradients: central differences through the sandbox's own forward pass, which does not need torch.
 * - 'never silently ignored': options the sandbox refuses where real torch accepts them, so not from real torch.
 * pyTorchCases.test.ts holds the larger set of programs frozen from real torch.
 */

const HEAD = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
const run = (src: string) => runPython(HEAD + src, { libs: [TORCH], maxSize: 2_000_000, timeBudget: 30_000 })
const out = (src: string) => {
  const r = run(src)
  expect(r.error).toBeNull()
  return r.stdout
}
const msg = (src: string) => {
  const e = run(src).error
  return e ? `${e.type}: ${e.message}` : null
}

type Nested = number | boolean | null | Nested[]
const flat = (x: Nested): (number | boolean | null)[] => (Array.isArray(x) ? x.flatMap(flat) : [x])
const shapeOf = (x: Nested): number[] => (Array.isArray(x) ? [x.length, ...(x.length ? shapeOf(x[0]) : [])] : [])
const parseList = (s: string): Nested => JSON.parse(s.replace(/True/g, 'true').replace(/False/g, 'false').replace(/\bnan\b/g, 'null').replace(/-inf/g, '-1e999').replace(/\binf\b/g, '1e999'))

/** r.tolist() and r.dtype against the expected values */
const expectValue = (stdout: string, want: Nested, dtype: string, tol: number) => {
  const [list, dt] = stdout.trim().split('\n')
  expect(dt).toBe(`torch.${dtype}`)
  const got = parseList(list)
  expect(shapeOf(got)).toEqual(shapeOf(want))
  const [fg, fw] = [flat(got), flat(want)]
  fg.forEach((v, i) => {
    const w = fw[i]
    if (typeof w === 'boolean' || dtype === 'int64') expect(v).toBe(w)
    else if (w === null) expect(v).toBeNull()
    else if (Math.abs((v as number) - w) > tol * (1 + Math.abs(w))) expect(v).toBeCloseTo(w, 5)
  })
}

describe('values against real torch', () => {
  const rows = VALUES.rows as [string, string, Nested, string][]
  it.each(rows.map((r) => [r[0], r] as const))('%s', (_name, [expr, , want, dtype]) => {
    if (dtype === 'pybool') {
      expect(out(`${VALUES.pre}print(${expr})`).trim()).toBe(want ? 'True' : 'False')
      return
    }
    expectValue(out(`${VALUES.pre}r = ${expr}\nprint(r.tolist())\nprint(r.dtype)\n`), want, dtype, 2e-6)
  })
})

describe('layers and optimisers against real torch', () => {
  const rows = LAYERS.rows as [string, string, Nested, string, string][]
  it.each(rows.map((r, i) => [`${i} ${r[4]} ${r[1]}`, r] as const))('%s', (_name, [setup, expr, want, dtype]) => {
    expectValue(out(`${setup}r = ${expr}\nprint(r.tolist())\nprint(r.dtype)\n`), want, dtype, dtype === 'float64' ? 1e-9 : 5e-6)
  })
  const IMPORT = 'from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LambdaLR, ReduceLROnPlateau\np = nn.Parameter(torch.zeros(1))\nopt = torch.optim.SGD([p], lr=0.1)\n'
  it.each(LAYERS.schedulers as [string, number[]][])('%s: exact, floating-point rounding included', (py, want) => {
    const got = out(`${IMPORT}sch = ${py}\nlrs = sch.get_last_lr()\nfor e in range(${want.length - 1}):\n    opt.step()\n    sch.step()\n    lrs += sch.get_last_lr()\nprint(lrs)`)
    expect(JSON.parse(got)).toEqual(want)
  })
  it.each(LAYERS.plateau as [string, number[], number[]][])('%s', (py, metrics, want) => {
    const got = out(`${IMPORT}sch = ${py}\nlrs = []\nfor m in ${JSON.stringify(metrics)}:\n    sch.step(m)\n    lrs += sch.get_last_lr()\nprint(lrs)`)
    expect(JSON.parse(got)).toEqual(want)
  })
})

/** central differences through the sandbox's forward pass: how many elements of the gradient disagree */
const CHECK = `
def S(t):
    return (t * torch.cos(torch.arange(t.numel(), dtype=t.dtype).view(t.shape))).sum() + (t * t).sum()
def check(f, x0, tol=1e-4, eps=1e-6):
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    f(x).backward()
    g = x.grad.view(-1)
    bad = 0
    for i in range(x.numel()):
        d = torch.zeros(x.numel(), dtype=torch.float64)
        d[i] = eps
        d = d.view(x.shape)
        with torch.no_grad():
            n = (f(x + d) - f(x - d)).item() / (2 * eps)
        if abs(n - g[i].item()) > tol * (1 + abs(n)):
            bad += 1
    return bad
def check_params(m, f, tol=1e-4, eps=1e-6):
    m.double()
    m.zero_grad()
    f().backward()
    bad = 0
    moved = 0
    for name, p in m.named_parameters():
        g = p.grad.view(-1)
        flat = p.data.view(-1)
        for i in range(p.numel()):
            old = flat[i].item()
            with torch.no_grad():
                flat[i] = old + eps
                a = f().item()
                flat[i] = old - eps
                b = f().item()
                flat[i] = old
            n = (a - b) / (2 * eps)
            if abs(n - g[i].item()) > tol * (1 + abs(n)):
                bad += 1
            if g[i].item() != 0:
                moved += 1
    return bad, moved > 0
M = [[.5, -1., 2.], [1.5, .25, -.75]]
N = [[.45, -1.2, 2.3], [1.55, .25, -.75]]
Q = [[2., 1., .5], [.25, 3., 1.], [.5, -1., 4.]]
V = [.3, -1.2, 2.1]
IMG = [[[[.1, .9, -.3, .4], [1.2, -.7, .6, .2], [.8, .5, -1.1, .3], [-.2, 1.4, .7, -.6]]]]
idx = torch.tensor([[0, 2], [1, 1]])
mask = torch.tensor([[True, False, True], [False, True, True]])
`

describe('gradients against central differences', () => {
  const OPS: [string, string][] = [
    // elementwise
    ...['log2', 'log10', 'log1p', 'rsqrt', 'reciprocal'].map((f): [string, string] => [f, `check(lambda x: S(torch.${f}(x + 3)), M)`]),
    ...['expm1', 'exp2', 'tan', 'atan', 'sinh', 'cosh', 'erf', 'erfc'].map((f): [string, string] => [f, `check(lambda x: S(torch.${f}(x)), M)`]),
    ['asin', 'check(lambda x: S(torch.asin(x / 3)), M)'], ['acos', 'check(lambda x: S(torch.acos(x / 3)), M)'],
    ['frac', 'check(lambda x: S(x.frac()), N)'], ['floor has gradient 0', 'check(lambda x: x.floor().sum() + (x * 0).sum(), N)'],
    ['atan2', 'check(lambda x: S(torch.atan2(x, x ** 2 + 1)), M)'], ['fmod', 'check(lambda x: S(torch.fmod(x, x * 0 + .7)), N)'], ['remainder', 'check(lambda x: S(x % .7), N)'], ['remainder by a tensor', 'check(lambda x: S(torch.remainder(x, x ** 2 + 1)), N)'],
    ['clamp', 'check(lambda x: S(x.clamp(-.5, 1.)), M)'], ['clamp by tensors', 'check(lambda x: S(torch.clamp(x, min=x * 0 - .5, max=x * 0 + 1.)), M)'], ['clamp_min', 'check(lambda x: S(x.clamp_min(.3)), M)'],
    ['lerp by a number', 'check(lambda x: S(torch.lerp(x, x ** 2, .3)), M)'], ['lerp by a tensor', 'check(lambda x: S(x.lerp(x * 3, torch.full_like(x, .7))), M)'],
    ['where', 'check(lambda x: S(torch.where(x > 0, x ** 2, -x)), M)'], ['masked_fill', 'check(lambda x: S(x.masked_fill(mask, 2.)), M)'], ['index_fill', 'check(lambda x: S(x.index_fill(1, torch.tensor([1]), 0.)), M)'],
    // reductions
    ['prod', 'check(lambda x: x.prod(), M)'], ['prod along a dim', 'check(lambda x: S(x.prod(dim=1)), M)'], ['prod through a zero', 'check(lambda x: S(x.prod(dim=1)), [[0., 1., 2.], [0., 0., 3.]])'],
    ['median', 'check(lambda x: x.median() ** 2, M)'], ['median along a dim', 'check(lambda x: S(x.median(dim=1).values), M)'], ['kthvalue', 'check(lambda x: S(x.kthvalue(2, dim=1).values), M)'],
    ['amax', 'check(lambda x: S(x.amax(dim=1)), M)'], ['amin of all dims', 'check(lambda x: x.amin(dim=(0, 1)) ** 2, M)'],
    ['norm 2', 'check(lambda x: x.norm(), M)'], ['norm 1', 'check(lambda x: torch.norm(x, p=1), M)'], ['norm 3 along a dim', 'check(lambda x: S(torch.norm(x, p=3, dim=1)), M)'], ['norm ∞', "check(lambda x: torch.norm(x, p=float('inf')) ** 2, M)"], ['norm -∞', "check(lambda x: torch.norm(x, p=-float('inf')) ** 2, M)"],
    ['linalg.norm ord=1', 'check(lambda x: torch.linalg.norm(x, ord=1), M)'], ['linalg.norm ∞', "check(lambda x: torch.linalg.norm(x, ord=float('inf')), M)"], ['linalg.vector_norm', 'check(lambda x: torch.linalg.vector_norm(x, ord=2, dim=0).sum(), M)'],
    ['trace', 'check(lambda x: x.trace() ** 2, Q)'], ['cumsum', 'check(lambda x: S(x.cumsum(1)), M)'], ['cumprod', 'check(lambda x: S(x.cumprod(1)), M)'], ['cumprod through a zero', 'check(lambda x: S(x.cumprod(0)), [0., 2., 3.])'],
    // sorted and selected
    ['sort', 'check(lambda x: S(x.sort(dim=1).values), M)'], ['sort descending', 'check(lambda x: S(torch.sort(x, dim=0, descending=True).values), M)'], ['topk', 'check(lambda x: S(x.topk(2, dim=1).values), M)'],
    ['gather', 'check(lambda x: S(x.gather(1, idx)), M)'], ['take', 'check(lambda x: S(x.take(torch.tensor([0, 5, 2]))), M)'], ['take_along_dim', 'check(lambda x: S(torch.take_along_dim(x, idx, dim=1)), M)'],
    ['scatter into', 'check(lambda x: S(x.scatter(1, torch.tensor([[0], [2]]), 5.)), M)'], ['scatter from', 'check(lambda x: S(torch.zeros(2, 3, dtype=torch.float64).scatter(1, torch.tensor([[0, 2], [1, 2]]), x[:, :2])), M)'], ['scatter_add', 'check(lambda x: S(torch.ones(2, 3, dtype=torch.float64).scatter_add(1, idx, x[:, :2])), M)'],
    ['index_select', 'check(lambda x: S(x.index_select(1, torch.tensor([2, 0, 2]))), M)'], ['masked_select', 'check(lambda x: S(x.masked_select(mask)), M)'], ['narrow', 'check(lambda x: S(x.narrow(1, 1, 2)), M)'],
    ['split', 'check(lambda x: S(x.split(2, dim=1)[1]) + S(x.split(2, dim=1)[0]), M)'], ['split by sizes', 'check(lambda x: S(x.split([1, 2], dim=1)[1]), M)'], ['chunk', 'check(lambda x: S(torch.chunk(x, 2)[1]), M)'], ['unbind', 'check(lambda x: S(x.unbind(1)[2]), M)'],
    // shape
    ['view_as', 'check(lambda x: S(x.view_as(x.t())), M)'], ['view of a transpose', 'check(lambda x: S(x.t().view(3, 1, 2)), M)'], ['expand', 'check(lambda x: S(x.expand(2, 2, 3)), M)'], ['expand_as', 'check(lambda x: S(x[0].expand_as(x)), M)'], ['broadcast_to', 'check(lambda x: S(torch.broadcast_to(x, (2, 2, 3))), M)'],
    ['repeat', 'check(lambda x: S(x.repeat(2, 1, 2)), M)'], ['tile', 'check(lambda x: S(x.tile(2)), M)'], ['repeat_interleave', 'check(lambda x: S(x.repeat_interleave(2, dim=1)), M)'], ['repeat_interleave flat', 'check(lambda x: S(torch.repeat_interleave(x, 2)), M)'],
    ['flip', 'check(lambda x: S(x.flip(0, 1)), M)'], ['roll', 'check(lambda x: S(x.roll(1, 1)), M)'], ['roll flat', 'check(lambda x: S(x.roll(2)), M)'], ['movedim', 'check(lambda x: S(x.movedim(0, 1)), M)'], ['unflatten', 'check(lambda x: S(x.unflatten(1, (3, 1))), M)'], ['ravel', 'check(lambda x: S(x.ravel()), M)'],
    ['tril', 'check(lambda x: S(x.tril()), Q)'], ['triu', 'check(lambda x: S(x.triu(1)), Q)'], ['diag of a vector', 'check(lambda x: S(torch.diag(x, 1)), V)'], ['diag of a matrix', 'check(lambda x: S(x.diag()), Q)'], ['diagonal', 'check(lambda x: S(x.diagonal(1)), M)'],
    ['hstack', 'check(lambda x: S(torch.hstack([x, x ** 2])), M)'], ['vstack', 'check(lambda x: S(torch.vstack([x, x ** 2])), V)'], ['column_stack', 'check(lambda x: S(torch.column_stack([x, x ** 2])), V)'], ['meshgrid', 'check(lambda x: S(torch.meshgrid(x, x ** 2, indexing="ij")[1]), V)'],
    ['outer', 'check(lambda x: S(torch.outer(x, x ** 2)), V)'], ['inner', 'check(lambda x: S(torch.inner(x, x ** 2)), M)'], ['inner of vectors', 'check(lambda x: torch.inner(x, x ** 2), V)'], ['mv', 'check(lambda x: S(x.mv(x[0])), Q)'],
    // linear algebra
    ['inv', 'check(lambda x: S(torch.linalg.inv(x)), Q)'], ['det', 'check(lambda x: torch.linalg.det(x), Q)'], ['det at a singular matrix', 'check(lambda x: torch.linalg.det(x), [[1., 2., 3.], [2., 4., 6.], [1., 0., 1.]])'], ['solve for A and b', 'check(lambda x: S(torch.linalg.solve(x, x[0])), Q)'], ['solve for a matrix', 'check(lambda x: S(torch.linalg.solve(x, x ** 2)), Q)'], ['matrix_power', 'check(lambda x: S(torch.linalg.matrix_power(x, 3)), Q)'], ['matrix_power -1', 'check(lambda x: S(torch.linalg.matrix_power(x, -1)), Q)'],
    // activations
    ...['elu', 'selu', 'gelu', 'softplus', 'silu', 'mish', 'hardtanh', 'relu6', 'logsigmoid', 'softsign', 'leaky_relu'].map((f): [string, string] => [`F.${f}`, `check(lambda x: S(F.${f}(x)), N)`]),
    ['F.gelu tanh', 'check(lambda x: S(F.gelu(x, approximate="tanh")), M)'], ['F.softplus beta and threshold', 'check(lambda x: S(F.softplus(x, beta=2.0, threshold=1.0)), N)'], ['F.softmin', 'check(lambda x: S(F.softmin(x, dim=1)), M)'],
    // functional
    ['F.normalize', 'check(lambda x: S(F.normalize(x)), M)'], ['F.cosine_similarity', 'check(lambda x: S(F.cosine_similarity(x, x ** 2)), M)'], ['F.pairwise_distance', 'check(lambda x: S(F.pairwise_distance(x, x ** 2)), M)'],
    ['F.pad constant', 'check(lambda x: S(F.pad(x, (1, 2, 0, 1), value=3.)), M)'], ['F.pad reflect', 'check(lambda x: S(F.pad(x, (1, 1, 1, 1), mode="reflect")), IMG)'], ['F.pad replicate', 'check(lambda x: S(F.pad(x, (2, 1, 1, 2), mode="replicate")), IMG)'], ['F.pad circular', 'check(lambda x: S(F.pad(x, (2, 1, 1, 2), mode="circular")), IMG)'],
    ['F.interpolate nearest', 'check(lambda x: S(F.interpolate(x, scale_factor=2)), IMG)'], ['F.interpolate bilinear', 'check(lambda x: S(F.interpolate(x, size=(3, 7), mode="bilinear")), IMG)'], ['F.interpolate align_corners', 'check(lambda x: S(F.interpolate(x, size=(6, 7), mode="bilinear", align_corners=True)), IMG)'],
    ['F.adaptive_avg_pool2d', 'check(lambda x: S(F.adaptive_avg_pool2d(x, (3, 2))), IMG)'], ['F.adaptive_max_pool2d', 'check(lambda x: S(F.adaptive_max_pool2d(x, 3)), IMG)'], ['F.max_pool1d', 'check(lambda x: S(F.max_pool1d(x, 2)), [[[.5, -1., 2., 1.5, .25]]])'], ['F.avg_pool1d', 'check(lambda x: S(F.avg_pool1d(x, 2, stride=1)), [[[.5, -1., 2., 1.5, .25]]])'], ['F.dropout off', 'check(lambda x: S(F.dropout(x, p=0.5, training=False)), M)'],
    ['F.conv1d', 'check(lambda x: S(F.conv1d(x, torch.tensor([[[1., -1., .5]]], dtype=torch.float64), stride=2, padding=1)), [[[.5, -1., 2., 1.5, .25, 1.]]])'], ['F.conv2d dilation', 'check(lambda x: S(F.conv2d(x, torch.tensor([[[[1., -1.], [.5, 2.]]]], dtype=torch.float64), dilation=2)), IMG)'], ['F.conv2d same padding, even kernel', 'check(lambda x: S(F.conv2d(x, torch.tensor([[[[1., -1.], [.5, 2.]]]], dtype=torch.float64), padding="same")), IMG)'], ['F.conv_transpose2d', 'check(lambda x: S(F.conv_transpose2d(x, torch.tensor([[[[1., -1.], [.5, 2.]]]], dtype=torch.float64), stride=2, padding=1, output_padding=1)), IMG)'],
    ['l1_loss', 'check(lambda x: F.l1_loss(x, x ** 2), M)'], ['smooth_l1_loss', 'check(lambda x: F.smooth_l1_loss(x, x * 0 + .4, beta=.5), M)'], ['huber_loss', 'check(lambda x: F.huber_loss(x, x * 0 + .4, delta=.5), M)'],
    ['cross_entropy with weights and smoothing', 'check(lambda x: F.cross_entropy(x, torch.tensor([2, 0]), weight=torch.tensor([1., 2., .5], dtype=torch.float64), label_smoothing=.1), M)'], ['cross_entropy with ignore_index', 'check(lambda x: F.cross_entropy(x, torch.tensor([2, 0]), ignore_index=0), M)'], ['cross_entropy over positions', 'check(lambda x: F.cross_entropy(x, torch.tensor([[0, 1, 1, 2]])), [[[.1, .9, -.3, .4], [1.2, -.7, .6, .2], [.8, .5, -1.1, .3]]])'], ['nll_loss with weights', 'check(lambda x: F.nll_loss(F.log_softmax(x, 1), torch.tensor([2, 0]), weight=torch.tensor([1., 2., .5], dtype=torch.float64)), M)'],
    ['BCE with logits, pos_weight', 'check(lambda x: F.binary_cross_entropy_with_logits(x, torch.ones_like(x) * .5, pos_weight=torch.full((3,), 2., dtype=torch.float64)), M)'], ['BCE with logits, weight', 'check(lambda x: F.binary_cross_entropy_with_logits(x, torch.ones_like(x), weight=torch.full((3,), 2., dtype=torch.float64)), M)'],
    ['embedding', 'check(lambda x: S(F.embedding(torch.tensor([[1, 0], [0, 0]]), x)), M)'],
    // in place: the history continues
    ['y *= 3', 'check(lambda x: S(x.clone().mul_(3)), M)'], ['y += x', 'check(lambda x: S(x.sin().add_(x ** 2)), M)'], ['y.sqrt_()', 'check(lambda x: S((x + 3).sqrt_()), M)'], ['y.clamp_()', 'check(lambda x: S(x.clone().clamp_(-.5, 1.)), M)'], ['y.t_()', 'check(lambda x: S(x.clone().t_()), M)'], ['y.unsqueeze_()', 'check(lambda x: S(x.clone().unsqueeze_(0)), M)'], ['y.copy_()', 'check(lambda x: S(torch.zeros(2, 3, dtype=torch.float64).copy_(x ** 2)), M)'],
  ]
  const setItem = 'def assign(x):\n    y = torch.zeros(2, 3, dtype=torch.float64)\n    y[0] = x[1] ** 2\n    y[1, 2] = x[0, 0]\n    return S(y)\nprint(check(assign, M))\ndef over(x):\n    y = x * 2\n    y[0, 1] = 5.\n    return S(y)\nprint(check(over, M))\n'
  it.each(OPS)('%s', (_name, code) => {
    expect(out(`${CHECK}print(${code})`).trim()).toBe('0')
  })
  it('x[i] = v joins the graph', () => expect(out(`${CHECK}${setItem}`).trim().split('\n')).toEqual(['0', '0']))

  const MODELS: [string, string, string][] = [
    ['Linear', 'nn.Linear(3, 2)', 'S(m(torch.tensor(M, dtype=torch.float64)))'],
    ['Conv1d', 'nn.Conv1d(2, 3, 2, stride=2, padding=1)', 'S(m(torch.tensor([[[.5, -1., 2., 1.5, .25], [.1, .9, -.3, .4, 1.]]], dtype=torch.float64)))'],
    ['Conv1d, groups and dilation', 'nn.Conv1d(2, 4, 2, dilation=2, groups=2)', 'S(m(torch.tensor([[[.5, -1., 2., 1.5, .25], [.1, .9, -.3, .4, 1.]]], dtype=torch.float64)))'],
    ['Conv2d, groups', 'nn.Conv2d(2, 2, 2, groups=2)', 'S(m(torch.tensor(IMG, dtype=torch.float64).repeat(1, 2, 1, 1)))'],
    ['Conv2d, reflect padding', 'nn.Conv2d(1, 2, 3, padding=1, padding_mode="reflect")', 'S(m(torch.tensor(IMG, dtype=torch.float64)))'],
    ['ConvTranspose2d', 'nn.ConvTranspose2d(1, 2, 2, stride=2, padding=1, output_padding=1)', 'S(m(torch.tensor(IMG, dtype=torch.float64)))'],
    ['BatchNorm1d training', 'nn.BatchNorm1d(3)', 'S(m(torch.tensor(M, dtype=torch.float64)))'],
    ['BatchNorm2d training', 'nn.BatchNorm2d(1)', 'S(m(torch.tensor(IMG, dtype=torch.float64)))'],
    ['BatchNorm1d eval', 'nn.BatchNorm1d(3).eval()', 'S(m(torch.tensor(M, dtype=torch.float64)))'],
    ['LayerNorm', 'nn.LayerNorm(3)', 'S(m(torch.tensor(M, dtype=torch.float64)))'],
    ['GroupNorm', 'nn.GroupNorm(1, 3)', 'S(m(torch.tensor(M, dtype=torch.float64)))'],
    ['PReLU', 'nn.PReLU()', 'S(m(torch.tensor(M, dtype=torch.float64)))'],
    ['Embedding', 'nn.Embedding(4, 2)', 'S(m(torch.tensor([[0, 1], [3, 1]])))'],
    ['RNN', 'nn.RNN(2, 2, 2)', 'S(m(torch.tensor([[[.5, -1.]], [[2., 1.5]], [[.25, .1]]], dtype=torch.float64))[0])'],
    ['LSTM', 'nn.LSTM(2, 2)', 'S(m(torch.tensor([[[.5, -1.]], [[2., 1.5]], [[.25, .1]]], dtype=torch.float64))[0])'],
    ['LSTM bidirectional, final state', 'nn.LSTM(2, 2, bidirectional=True)', 'S(m(torch.tensor([[[.5, -1.]], [[2., 1.5]]], dtype=torch.float64))[1][1])'],
    ['GRU', 'nn.GRU(2, 2, batch_first=True)', 'S(m(torch.tensor([[[.5, -1.], [2., 1.5], [.25, .1]]], dtype=torch.float64))[0])'],
    ['LSTMCell', 'nn.LSTMCell(2, 2)', 'S(m(torch.tensor([[.5, -1.]], dtype=torch.float64))[1])'],
    ['GRUCell', 'nn.GRUCell(2, 2)', 'S(m(torch.tensor([[.5, -1.]], dtype=torch.float64)))'],
    ['MultiheadAttention', 'nn.MultiheadAttention(2, 2)', 'S(m(*QKV)[0])'],
  ]
  it.each(MODELS)('%s trains: autograd agrees for every parameter', (_name, make, loss) => {
    const r = out(`${CHECK}torch.manual_seed(3)\nq = torch.tensor([[[.5, -1.]], [[2., 1.5]]], dtype=torch.float64)\nm = ${make}\ndef f():\n    QKV = (q, q, q)\n    return ${loss.replace('*QKV', 'q, q, q')}\nprint(check_params(m, f))`)
    expect(r.trim()).toBe('(0, True)')
  })
  it('inputs too: gradients reach the input through BatchNorm in training', () => {
    expect(out(`${CHECK}m = nn.BatchNorm1d(3).double()\nprint(check(lambda x: S(m(x)), M))`).trim()).toBe('0')
    expect(out(`${CHECK}m = nn.BatchNorm2d(1).double()\nm.train()\nprint(check(lambda x: S(m(x)), IMG))`).trim()).toBe('0')
    expect(out(`${CHECK}m = nn.LSTM(3, 2).double()\nprint(check(lambda x: S(m(x.unsqueeze(1))[0]), M))`).trim()).toBe('0')
  })
  it('mode picks one element: the gradient goes to that element alone', () => {
    expect(out('x = torch.tensor([1., 2., 2., 3.], requires_grad=True)\nx.mode().values.backward()\nprint(x.grad)')).toBe('tensor([0., 0., 1., 0.])\n')
  })
  it('max() of everything, amax and median() share the gradient among ties; max along a dim follows its index', () => {
    const grad = (f: string) => out(`x = torch.tensor([3., 1., 3., 1.], requires_grad=True)\n${f}.backward()\nprint(x.grad.tolist())`).trim()
    expect(grad('x.max()')).toBe('[0.5, 0.0, 0.5, 0.0]')
    expect(grad('x.min()')).toBe('[0.0, 0.5, 0.0, 0.5]')
    expect(grad('x.amax()')).toBe('[0.5, 0.0, 0.5, 0.0]')
    expect(grad('x.amin(dim=0)')).toBe('[0.0, 0.5, 0.0, 0.5]')
    expect(grad('x.median()')).toBe('[0.0, 0.5, 0.0, 0.5]')
    expect(grad('x.max(dim=0).values')).toBe('[1.0, 0.0, 0.0, 0.0]')
  })
  it('a padding_idx row of an Embedding starts at zero and gets no gradient', () => {
    expect(out('m = nn.Embedding(3, 2, padding_idx=1)\nprint(m.weight[1].tolist())\nm(torch.tensor([0, 1, 1])).sum().backward()\nprint(m.weight.grad[1].tolist(), m.weight.grad[0].tolist())')).toBe('[0.0, 0.0]\n[0.0, 0.0] [1.0, 1.0]\n')
  })
  it('torch.autograd.grad returns gradients and leaves .grad alone', () => {
    expect(out(`x = torch.tensor([1., 2.], requires_grad=True)
y = x ** 2
z = y.sum()
print(torch.autograd.grad(z, x, retain_graph=True))
print(x.grad)
print(torch.autograd.grad(z, [y, x]))
w = torch.tensor([1., 2.], requires_grad=True)
print(torch.autograd.grad((w ** 3).sum(), w, grad_outputs=torch.tensor(2.)))
`)).toBe('(tensor([2., 4.]),)\nNone\n(tensor([1., 1.]), tensor([2., 4.]))\n(tensor([ 6., 24.]),)\n')
  })
})

describe('in-place operations keep autograd honest', () => {
  it('a residual connection written with += trains the path it adds', () => {
    expect(out('x = torch.tensor([2.], requires_grad=True)\ny = x * 3\ny += x ** 2\ny.backward()\nprint(x.grad, y.grad_fn.name())')).toBe('tensor([7.]) AddBackward0\n')
  })
  it('y *= k multiplies the gradient', () => {
    expect(out('x = torch.tensor([2.], requires_grad=True)\ny = x * 2\ny.mul_(3)\ny.backward()\nprint(x.grad)')).toBe('tensor([6.])\n')
  })
  it('a tensor without history joins the graph when assigned into', () => {
    expect(out('w = torch.tensor(2., requires_grad=True)\nz = torch.zeros(3)\nz[1] = w * 5\nprint(z.requires_grad, z.grad_fn is not None)\nz.sum().backward()\nprint(w.grad)')).toBe('True True\ntensor(5.)\n')
  })
  it('no_grad: values change, no history', () => {
    expect(out('w = torch.ones(2, requires_grad=True)\nwith torch.no_grad():\n    w *= 3\n    w.add_(1)\nprint(w, w.is_leaf)')).toBe('tensor([4., 4.], requires_grad=True) True\n')
  })
  it('views share memory with what they view', () => {
    expect(out('x = torch.arange(6.).view(2, 3)\nx.narrow(1, 0, 1)[0, 0] = 9.\nx.split(2, dim=1)[1].fill_(7.)\nx.unbind(0)[1][1] = -1.\nx.diagonal()[1] = 5.\nprint(x)\nprint(x.chunk(2)[0])')).toBe('tensor([[9., 1., 7.],\n        [3., 5., 7.]])\ntensor([[9., 1., 7.]])\n')
  })
  it('in-place dtype and shape rules', () => {
    expect(msg('x = torch.tensor([1, 2])\nx.sqrt_()')).toBe("RuntimeError: result type Float can't be cast to the desired output type Long")
    expect(msg('x = torch.tensor([1, 2])\nx.add_(0.5)')).toBe("RuntimeError: result type Float can't be cast to the desired output type Long")
    expect(msg('x = torch.ones(1)\nx += torch.ones(3)')).toBe("RuntimeError: output with shape [1] doesn't match the broadcast shape [3]")
    expect(msg('w = torch.ones(2, requires_grad=True)\nw[0].add_(1)')).toBe('RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.')
    expect(msg('w = torch.ones(2, requires_grad=True)\nw.sqrt_()')).toBe('RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.')
    expect(out('w = torch.ones(2, requires_grad=True)\ny = w * 2\nv = y[0]\nv.mul_(3)\nprint(y, v)\ny.sum().backward()\nprint(w.grad)')).toBe('tensor([6., 2.], grad_fn=<CopySlices>) tensor(6., grad_fn=<AsStridedBackward0>)\ntensor([6., 2.])\n')
  })
  it('the common in-place forms', () => {
    expect(out(`x = torch.tensor([4., 9.])
print(x.clone().sqrt_(), x.clone().clamp_(max=5.), x.clone().neg_(), x.clone().pow_(2), x.clone().round_(), x.clone().exp_().log_(), x.clone().sub_(1, alpha=2), x.clone().add_(torch.ones(2), alpha=3))
y = torch.ones(2, 3)
print(y.t_().shape, y.unsqueeze_(0).shape, y.squeeze_().shape, y.transpose_(0, 1).shape)
z = torch.zeros(2, 2).fill_diagonal_(1.)
print(z)
print(z.zero_().sum(), z.fill_(2.).sum())
m = torch.zeros(3).masked_fill_(torch.tensor([True, False, True]), 1.)
print(m, m.detach_().requires_grad)
print(torch.ones(2, 2).normal_(3., 0.).tolist(), torch.ones(2).uniform_(5., 5.).tolist(), torch.ones(2).bernoulli_(1.).tolist(), torch.ones(2).exponential_().min().item() > 0, torch.zeros(3).random_(2, 3).tolist())
`)).toBe(`tensor([2., 3.]) tensor([4., 5.]) tensor([-4., -9.]) tensor([16., 81.]) tensor([4., 9.]) tensor([4., 9.]) tensor([2., 7.]) tensor([ 7., 12.])
torch.Size([3, 2]) torch.Size([1, 3, 2]) torch.Size([3, 2]) torch.Size([2, 3])
tensor([[1., 0.],
        [0., 1.]])
tensor(0.) tensor(8.)
tensor([1., 0., 1.]) False
[[3.0, 3.0], [3.0, 3.0]] [5.0, 5.0] [1.0, 1.0] True [2.0, 2.0, 2.0]
`)
  })
})

describe('tensor basics', () => {
  it('new tensors like this one', () => {
    expect(out(`x = torch.ones(2, dtype=torch.float64)
print(x.new_zeros(2, 2), x.new_ones((1, 2)), x.new_full((2,), 3.), x.new_tensor([1, 2]), x.new_zeros(1, dtype=torch.int64), x.new_empty(0).shape)
print(torch.ones(2).type_as(x).dtype, torch.ones(2).to(x).dtype, torch.ones(2).type(torch.LongTensor).dtype, torch.ones(2).type('torch.BoolTensor').dtype, torch.ones(2).byte().dtype, torch.ones(2).type())
print(torch.FloatTensor([1, 2]), torch.LongTensor([1, 2]), torch.Tensor(2, 3).shape, torch.DoubleTensor(2).dtype, isinstance(torch.ones(1), torch.FloatTensor), isinstance(torch.ones(1), torch.LongTensor))
`)).toBe(`tensor([[0., 0.],
        [0., 0.]], dtype=torch.float64) tensor([[1., 1.]], dtype=torch.float64) tensor([3., 3.], dtype=torch.float64) tensor([1., 2.], dtype=torch.float64) tensor([0]) torch.Size([0])
torch.float64 torch.float64 torch.int64 torch.bool torch.uint8 torch.FloatTensor
tensor([1., 2.]) tensor([1, 2]) torch.Size([2, 3]) torch.float64 True False
`)
  })
  it('allclose and equal', () => {
    expect(out(`a = torch.tensor([1., 2.])
print(torch.equal(a, a), torch.equal(a, a.view(2, 1)), a.equal(a + 1), torch.equal(torch.tensor([1, 2]), torch.tensor([1., 2.])))
print(torch.allclose(a, a + 1e-7), torch.allclose(a, a + 1), torch.allclose(a, a + 1, atol=2.), torch.allclose(a, a * 1.1, rtol=0.2))
n = torch.tensor([float('nan')])
print(torch.allclose(n, n), torch.allclose(n, n, equal_nan=True), n.isnan().item(), torch.isinf(n).item())
`)).toBe('True False False True\nTrue False True True\nFalse True True False\n')
    expect(msg('torch.allclose(torch.ones(1), torch.ones(1, dtype=torch.float64))')).toBe('RuntimeError: Float did not match Double')
  })
  it('random creators: shapes, ranges and generators', () => {
    expect(out(`torch.manual_seed(1)
p = torch.randperm(8)
print(sorted(p.tolist()), p.dtype)
g = torch.Generator().manual_seed(5)
a = torch.rand(3, generator=g)
g.manual_seed(5)
print(torch.equal(a, torch.rand(3, generator=g)), g.initial_seed())
g.manual_seed(5)
b = torch.randperm(6, generator=g)
g.manual_seed(5)
print(torch.equal(b, torch.randperm(6, generator=g)))
print(torch.normal(1., 0., (2,)).tolist(), torch.normal(torch.tensor([1., 2.]), 0.).tolist(), torch.normal(0., torch.zeros(2)).tolist(), torch.bernoulli(torch.tensor([0., 1.])).tolist(), torch.multinomial(torch.tensor([0., 1.]), 2, replacement=True).tolist(), sorted(torch.multinomial(torch.tensor([1., 1., 1.]), 3).tolist()))
print(torch.randint(5, (2,)).shape, torch.randint(0, 5, (3,)).max().item() < 5, torch.randint(2, 3, size=(2,)).tolist(), torch.randint(low=1, high=2, size=(1,)).tolist(), torch.randint_like(torch.ones(2), 3, 4).tolist())
`)).toBe(`[0, 1, 2, 3, 4, 5, 6, 7] torch.int64
True 5
True
[1.0, 1.0] [1.0, 2.0] [0.0, 0.0] [0.0, 1.0] [1, 1] [0, 1, 2]
torch.Size([2]) True [2, 2] [1] [3.0, 3.0]
`)
    expect(msg('torch.multinomial(torch.tensor([1., 1.]), 3)')).toBe('RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement')
    expect(msg('torch.rand(2, dtype=torch.int64)')).toBe(`NotImplementedError: "check_uniform_bounds" not implemented for 'Long'`)
  })
  it('0-d tensors: dim -1 and dim 0 both name their only axis', () => {
    expect(out('x = torch.tensor(5.)\nprint(x.sort(-1).values, x.sum(-1), torch.topk(x, 1).values, x.cumsum(0), x.softmax(-1), x.mode().values, x.median(0).values, x.unsqueeze(-1).shape, x.prod(0))')).toBe('tensor(5.) tensor(5.) tensor(5.) tensor(5.) tensor(1.) tensor(5.) tensor(5.) torch.Size([1]) tensor(5.)\n')
  })
  it('return types print and unpack as PyTorch does', () => {
    expect(out('v = torch.tensor([3., 1.])\nprint(v.sort())\nvalues, indices = torch.max(v, 0)\nprint(values, indices)\nprint(v.topk(1), v.median(0), v.mode(), v.kthvalue(1), sep="\\n")')).toBe(`torch.return_types.sort(
values=tensor([1., 3.]),
indices=tensor([1, 0]))
tensor(3.) tensor(0)
torch.return_types.topk(
values=tensor([3.]),
indices=tensor([0]))
torch.return_types.median(
values=tensor(1.),
indices=tensor(1))
torch.return_types.mode(
values=tensor(1.),
indices=tensor(1))
torch.return_types.kthvalue(
values=tensor(1.),
indices=tensor(1))
`)
  })
  it('positional keepdim and unbiased, once ignored', () => {
    expect(out('a = torch.tensor([[1., 2.], [3., 4.]])\nprint(a.max(1, True).values.shape, a.argmax(1, True).shape, a.min(0, True).indices.shape, torch.sum(a, 0, True).shape, torch.mean(a, 1, True).shape, a.std(0, True, True).shape)\nprint(a.all(), a.any(1))')).toBe('torch.Size([2, 1]) torch.Size([2, 1]) torch.Size([1, 2]) torch.Size([1, 2]) torch.Size([2, 1]) torch.Size([1, 2])\ntensor(True) tensor([True, True])\n')
  })
  it('a view of a non-contiguous tensor works when PyTorch says it can', () => {
    expect(out('x = torch.arange(6).view(2, 3).t()\nprint(x.view(3, 2).shape, x.view(3, 1, 2)[2, 0, 1].item(), x.view(1, 3, 2).t().shape if False else "skip")\ny = torch.arange(8).view(2, 2, 2).permute(2, 0, 1)\nprint(y.view(2, 4).shape)')).toBe('torch.Size([3, 2]) 5 skip\ntorch.Size([2, 4])\n')
    expect(msg('torch.ones(2, 3).t().view(6)')).toMatch("view size is not compatible with input tensor's size and stride")
  })
})

describe('details that used to slip', () => {
  it('NumPy spellings: axis and keepdims', () => {
    expect(out('a = torch.tensor([[1., 2.], [3., 4.]])\nprint(a.sum(axis=0, keepdims=True), a.max(axis=1).values, torch.max(a, axis=0).indices, a.argmax(axis=1), a.prod(axis=0), torch.mean(a, axis=1, keepdims=True).shape, a.all(axis=0))')).toBe('tensor([[4., 6.]]) tensor([2., 4.]) tensor([1, 1]) tensor([1, 1]) tensor([3., 8.]) torch.Size([2, 1]) tensor([True, True])\n')
  })
  it('a Python float keeps double precision next to a float64 tensor', () => {
    expect(out('x = torch.tensor([1., 2.], dtype=torch.float64)\nprint((x * 0.1).tolist(), (0.1 + x)[0].item(), x.clone().add_(0.1)[0].item())\nx[0] = 0.1\nprint(x[0].item())\nf = torch.tensor([1.])\nprint((f * 0.1).item(), (f * 0.1).dtype, torch.where(f > 0, 1.0, 0.0).dtype)')).toBe('[0.1, 0.2] 1.1 1.1\n0.1\n0.10000000149011612 torch.float32 torch.float32\n')
  })
  it('grad modes', () => {
    expect(out(`x = torch.ones(1, requires_grad=True)
with torch.set_grad_enabled(False):
    print((x * 2).requires_grad)
print((x * 2).requires_grad, torch.is_grad_enabled())
torch.set_grad_enabled(False)
print((x * 2).requires_grad)
torch.set_grad_enabled(True)
print(torch.no_grad()(lambda t: (t * 2).requires_grad)(x), torch.no_grad(lambda t: (t * 2).requires_grad)(x), (x * 2).requires_grad)
with torch.inference_mode():
    print((x * 2).requires_grad)
with torch.no_grad():
    with torch.enable_grad():
        print((x * 2).requires_grad)
    print(torch.is_grad_enabled())
`)).toBe('False\nTrue True\nFalse\nFalse False True\nFalse\nTrue\nFalse\n')
  })
  it('float(loss) and friends', () => {
    expect(out('x = torch.tensor([2.], requires_grad=True)\nl = (x ** 2).sum()\nprint(float(l), round(float(l), 2), x.detach().numpy().tolist(), x.numpy(force=True).tolist())')).toBe('4.0 4.0 [2.0] [2.0]\n')
    expect(msg('torch.ones(2, requires_grad=True).numpy()')).toMatch("Can't call numpy() on Tensor that requires grad")
  })
  it('keyword spellings and shared memory', () => {
    expect(out(`a = torch.zeros(2, 3)
print(a.size(dim=1), a.permute(dims=(1, 0)).shape, torch.permute(a, (1, 0)).shape, a.transpose(dim0=0, dim1=1).shape, torch.reshape(a, shape=(3, 2)).shape, a.view(size=(6,)).shape)
t = torch.zeros(1)
p = nn.Parameter(t)
p.data.add_(1)
print(t, torch.as_tensor(t) is t, torch.as_tensor(t, dtype=torch.float64).dtype)
d = torch.zeros(1)
d.data = torch.ones(1, dtype=torch.float64)
print(d)
`)).toBe('3 torch.Size([3, 2]) torch.Size([3, 2]) torch.Size([3, 2]) torch.Size([3, 2]) torch.Size([6])\ntensor([1.]) True torch.float64\ntensor([1.], dtype=torch.float64)\n')
  })
  it('inplace=True changes the input', () => {
    expect(out('x = torch.tensor([-1., 2.])\ny = nn.ReLU(inplace=True)(x)\nprint(x, y is x)\nz = torch.tensor([-1., 2.])\nF.relu(z, inplace=True)\nprint(z)\nw = torch.tensor([-1., 2.])\nF.elu(w)\nprint(w)\nv = torch.ones(4)\nnn.Dropout(0.0, inplace=True)(v)\nprint(v.sum())')).toBe('tensor([0., 2.]) True\ntensor([0., 2.])\ntensor([-1.,  2.])\ntensor(4.)\n')
  })
  it('the model and the data must agree on dtype, as in PyTorch', () => {
    expect(out('m = nn.Linear(2, 1).double()\nprint(m(torch.ones(1, 2, dtype=torch.float64)).dtype, nn.Linear(2, 1, dtype=torch.float64).weight.dtype, nn.Conv1d(1, 1, 1, dtype=torch.float64)(torch.ones(1, 1, 1, dtype=torch.float64)).dtype)')).toBe('torch.float64 torch.float64 torch.float64\n')
  })
  it('ModuleList.insert', () => {
    expect(out('l = nn.ModuleList([nn.ReLU()])\nl.insert(0, nn.Tanh())\nprint(l)')).toBe('ModuleList(\n  (0): Tanh()\n  (1): ReLU()\n)\n')
  })
})

describe('layers print as PyTorch prints them', () => {
  it.each([
    ['nn.Conv1d(1, 2, 3)', 'Conv1d(1, 2, kernel_size=(3,), stride=(1,))'],
    ['nn.Conv1d(1, 2, 3, stride=2, padding=1, dilation=2, bias=False)', 'Conv1d(1, 2, kernel_size=(3,), stride=(2,), padding=(1,), dilation=(2,), bias=False)'],
    ['nn.Conv2d(2, 4, (3, 5), padding="same", groups=2, padding_mode="reflect")', 'Conv2d(2, 4, kernel_size=(3, 5), stride=(1, 1), padding=same, groups=2, padding_mode=reflect)'],
    ['nn.ConvTranspose2d(1, 2, 3, stride=2, padding=1, output_padding=1)', 'ConvTranspose2d(1, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))'],
    ['nn.MaxPool1d(2)', 'MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)'],
    ['nn.MaxPool2d(2)', 'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)'],
    ['nn.MaxPool2d((2, 3), stride=1, padding=(1, 0))', 'MaxPool2d(kernel_size=(2, 3), stride=1, padding=(1, 0), dilation=1, ceil_mode=False)'],
    ['nn.AvgPool2d(3, padding=1)', 'AvgPool2d(kernel_size=3, stride=3, padding=1)'],
    ['nn.AvgPool1d(2)', 'AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))'],
    ['nn.BatchNorm1d(4, momentum=None, affine=False)', 'BatchNorm1d(4, eps=1e-05, momentum=None, affine=False, bias=False, track_running_stats=True)'],
    ['nn.BatchNorm2d(16)', 'BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)'],
    ['nn.LayerNorm((2, 3))', 'LayerNorm((2, 3), eps=1e-05, elementwise_affine=True, bias=True)'],
    ['nn.GroupNorm(2, 4)', 'GroupNorm(2, 4, eps=1e-05, affine=True, bias=True)'],
    ['nn.Dropout2d(0.3)', 'Dropout2d(p=0.3, inplace=False)'],
    ['nn.Dropout(0.3, inplace=True)', 'Dropout(p=0.3, inplace=True)'],
    ['nn.Unflatten(1, (2, 3))', 'Unflatten(dim=1, unflattened_size=(2, 3))'],
    ['nn.ReLU()', 'ReLU()'], ['nn.ReLU(inplace=True)', 'ReLU(inplace=True)'], ['nn.ELU()', 'ELU(alpha=1.0)'], ['nn.GELU()', "GELU(approximate='none')"], ['nn.Softplus()', 'Softplus(beta=1.0, threshold=20.0)'], ['nn.LeakyReLU()', 'LeakyReLU(negative_slope=0.01)'], ['nn.Hardtanh()', 'Hardtanh(min_val=-1.0, max_val=1.0)'], ['nn.PReLU()', 'PReLU(num_parameters=1)'], ['nn.Softmax(dim=1)', 'Softmax(dim=1)'], ['nn.Softmax()', 'Softmax(dim=None)'],
    ['nn.SmoothL1Loss()', 'SmoothL1Loss()'], ['nn.HuberLoss()', 'HuberLoss()'],
    ['nn.Embedding(10, 3)', 'Embedding(10, 3)'], ['nn.Embedding(10, 3, padding_idx=0)', 'Embedding(10, 3, padding_idx=0)'],
    ['nn.RNN(1, 2)', 'RNN(1, 2)'], ['nn.LSTM(3, 4, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)', 'LSTM(3, 4, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)'], ['nn.GRU(1, 2, bias=False)', 'GRU(1, 2, bias=False)'], ['nn.LSTMCell(1, 2)', 'LSTMCell(1, 2)'],
    ['nn.Upsample(scale_factor=2)', "Upsample(scale_factor=2.0, mode='nearest')"], ['nn.Upsample(size=(4, 4), mode="bilinear")', "Upsample(size=(4, 4), mode='bilinear')"],
    ['nn.ZeroPad2d(1)', 'ZeroPad2d((1, 1, 1, 1))'], ['nn.ZeroPad2d((1, 2, 3, 4))', 'ZeroPad2d((1, 2, 3, 4))'], ['nn.ConstantPad2d(1, 0.5)', 'ConstantPad2d(padding=(1, 1, 1, 1), value=0.5)'], ['nn.ReflectionPad2d(1)', 'ReflectionPad2d((1, 1, 1, 1))'],
    ['nn.AdaptiveAvgPool2d((7, 7))', 'AdaptiveAvgPool2d(output_size=(7, 7))'], ['nn.AdaptiveAvgPool2d(1)', 'AdaptiveAvgPool2d(output_size=1)'], ['nn.AdaptiveMaxPool2d((None, 2))', 'AdaptiveMaxPool2d(output_size=(None, 2))'],
    ['nn.Identity()', 'Identity()'], ['nn.Sequential()', 'Sequential()'],
  ])('%s', (make, want) => expect(out(`print(${make})`).trim()).toBe(want))
  it('containers', () => {
    expect(out(`print(nn.Sequential(nn.Linear(1, 2), nn.Sequential(nn.ReLU())))
print(nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1), nn.ReLU()]))
print(nn.ModuleDict({'a': nn.ReLU(), 'b': nn.Linear(1, 1)}))
print(nn.MultiheadAttention(4, 2))
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for k in range(3)])
        self.norm = nn.BatchNorm1d(2)
print(Block())
`)).toBe(`Sequential(
  (0): Linear(in_features=1, out_features=2, bias=True)
  (1): Sequential(
    (0): ReLU()
  )
)
ModuleList(
  (0-1): 2 x Linear(in_features=1, out_features=1, bias=True)
  (2): ReLU()
)
ModuleDict(
  (a): ReLU()
  (b): Linear(in_features=1, out_features=1, bias=True)
)
MultiheadAttention(
  (out_proj): NonDynamicallyQuantizableLinear(in_features=4, out_features=4, bias=True)
)
Block(
  (layers): ModuleList(
    (0-2): 3 x Linear(in_features=2, out_features=2, bias=True)
  )
  (norm): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
)
`)
  })
  it('optimisers print their defaults, sorted', () => {
    expect(out('p = [nn.Parameter(torch.zeros(1))]\nprint(torch.optim.SGD(p, lr=0.1))\nprint(torch.optim.Adam(p))\nprint(torch.optim.RMSprop(p))\nprint(torch.optim.Adagrad(p))')).toBe(`SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    fused: None
    lr: 0.1
    maximize: False
    momentum: 0
    nesterov: False
    weight_decay: 0
)
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
RMSprop (
Parameter Group 0
    alpha: 0.99
    capturable: False
    centered: False
    differentiable: False
    eps: 1e-08
    foreach: None
    lr: 0.01
    maximize: False
    momentum: 0
    weight_decay: 0
)
Adagrad (
Parameter Group 0
    differentiable: False
    eps: 1e-10
    foreach: None
    fused: None
    initial_accumulator_value: 0
    lr: 0.01
    lr_decay: 0
    maximize: False
    weight_decay: 0
)
`)
  })
})

describe('shapes through the new layers', () => {
  it.each([
    ['nn.Conv1d(2, 4, 3)', 'torch.zeros(5, 2, 10)', [5, 4, 8]],
    ['nn.Conv1d(2, 4, 3, padding="same")', 'torch.zeros(2, 10)', [4, 10]],
    ['nn.Conv1d(2, 4, 4, padding="same")', 'torch.zeros(1, 2, 10)', [1, 4, 10]],
    ['nn.Conv2d(1, 2, 2, padding="same")', 'torch.zeros(1, 1, 5, 5)', [1, 2, 5, 5]],
    ['nn.Conv2d(1, 2, 3, padding="valid")', 'torch.zeros(1, 1, 5, 5)', [1, 2, 3, 3]],
    ['nn.Conv2d(1, 2, 3, dilation=2)', 'torch.zeros(1, 1, 7, 7)', [1, 2, 3, 3]],
    ['nn.ConvTranspose2d(3, 2, 2, stride=2)', 'torch.zeros(4, 3, 5, 5)', [4, 2, 10, 10]],
    ['nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, output_padding=1)', 'torch.zeros(3, 5, 5)', [2, 10, 10]],
    ['nn.MaxPool1d(2)', 'torch.zeros(3, 2, 8)', [3, 2, 4]],
    ['nn.MaxPool1d(2)', 'torch.zeros(2, 8)', [2, 4]],
    ['nn.AdaptiveAvgPool2d((7, 7))', 'torch.zeros(1, 3, 32, 32)', [1, 3, 7, 7]],
    ['nn.AdaptiveAvgPool2d((5, 7))', 'torch.zeros(3, 2, 2)', [3, 5, 7]],
    ['nn.AdaptiveAvgPool1d(3)', 'torch.zeros(1, 2, 9)', [1, 2, 3]],
    ['nn.Unflatten(1, (2, 3))', 'torch.zeros(4, 6)', [4, 2, 3]],
    ['nn.Unflatten(-1, (2, -1))', 'torch.zeros(4, 6)', [4, 2, 3]],
    ['nn.Embedding(5, 3)', 'torch.zeros(2, 4, dtype=torch.int64)', [2, 4, 3]],
    ['nn.Upsample(scale_factor=2)', 'torch.zeros(1, 2, 3)', [1, 2, 6]],
    ['nn.Upsample(scale_factor=(2, 3))', 'torch.zeros(1, 1, 2, 3)', [1, 1, 4, 9]],
    ['nn.ZeroPad2d(2)', 'torch.zeros(1, 1, 3, 4)', [1, 1, 7, 8]],
    ['nn.Dropout2d(0.5)', 'torch.ones(4, 2, 3, 3)', [4, 2, 3, 3]],
    ['nn.Flatten()', 'torch.zeros(2, 3, 4)', [2, 12]],
    ['nn.GroupNorm(1, 2)', 'torch.ones(3, 2, 4, 4)', [3, 2, 4, 4]],
    ['nn.LayerNorm(3)', 'torch.ones(2, 4, 3)', [2, 4, 3]],
  ])('%s on %s', (make, input, want) => expect(out(`print(${make}(${input}).shape)`).trim()).toBe(`torch.Size([${(want as number[]).join(', ')}])`))
  it('recurrent layers: outputs and states', () => {
    expect(out(`x = torch.zeros(5, 2, 3)
o, h = nn.RNN(3, 4)(x)
print(o.shape, h.shape)
o, (h, c) = nn.LSTM(3, 4, 2, batch_first=True)(torch.zeros(2, 5, 3))
print(o.shape, h.shape, c.shape)
o, h = nn.GRU(3, 4, 2, bidirectional=True)(x)
print(o.shape, h.shape)
o, h = nn.GRU(3, 4)(torch.zeros(5, 3))
print(o.shape, h.shape)
o, (h, c) = nn.LSTM(3, 4)(torch.zeros(5, 3), (torch.zeros(1, 4), torch.zeros(1, 4)))
print(o.shape, h.shape, c.shape)
print([n for n, _ in nn.LSTM(1, 2, 2, bidirectional=True, bias=False).named_parameters()])
print([tuple(p.shape) for p in nn.LSTM(3, 4, 2).parameters()], [tuple(p.shape) for p in nn.GRU(3, 4).parameters()], [tuple(p.shape) for p in nn.RNN(3, 4).parameters()])
print([n for n, _ in nn.GRUCell(1, 2).named_parameters()], nn.LSTMCell(1, 2)(torch.zeros(3, 1))[1].shape, nn.RNNCell(1, 2)(torch.zeros(1)).shape)
print([n for n, _ in nn.MultiheadAttention(4, 2).named_parameters()])
o, w = nn.MultiheadAttention(4, 2, batch_first=True)(torch.zeros(2, 3, 4), torch.zeros(2, 5, 4), torch.zeros(2, 5, 4))
print(o.shape, w.shape)
o, w = nn.MultiheadAttention(4, 2)(torch.zeros(3, 4), torch.zeros(3, 4), torch.zeros(3, 4), average_attn_weights=False)
print(o.shape, w.shape)
print(nn.MultiheadAttention(4, 2)(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), need_weights=False)[1])
`)).toBe(`torch.Size([5, 2, 4]) torch.Size([1, 2, 4])
torch.Size([2, 5, 4]) torch.Size([2, 2, 4]) torch.Size([2, 2, 4])
torch.Size([5, 2, 8]) torch.Size([4, 2, 4])
torch.Size([5, 4]) torch.Size([1, 4])
torch.Size([5, 4]) torch.Size([1, 4]) torch.Size([1, 4])
['weight_ih_l0', 'weight_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse', 'weight_ih_l1', 'weight_hh_l1', 'weight_ih_l1_reverse', 'weight_hh_l1_reverse']
[(16, 3), (16, 4), (16,), (16,), (16, 4), (16, 4), (16,), (16,)] [(12, 3), (12, 4), (12,), (12,)] [(4, 3), (4, 4), (4,), (4,)]
['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'] torch.Size([3, 2]) torch.Size([2])
['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']
torch.Size([2, 3, 4]) torch.Size([2, 3, 5])
torch.Size([3, 4]) torch.Size([2, 3, 3])
None
`)
  })
  it('Dropout2d zeroes whole channels and rescales the rest; eval leaves it all', () => {
    expect(out(`torch.manual_seed(2)
d = nn.Dropout2d(0.5)
y = d(torch.ones(50, 4, 3, 3))
per = y.view(200, -1)
whole = ((per == 0).all(dim=1) | (per == 2).all(dim=1)).all()
print(whole.item(), 60 < (per[:, 0] == 0).sum().item() < 140)
d.eval()
print(d(torch.ones(1, 1, 2, 2)).tolist())
print(nn.Dropout2d(1.0)(torch.ones(1, 1, 1, 1)).item(), F.dropout2d(torch.ones(1, 2, 1, 1), p=0.0).tolist())
`)).toBe('True True\n[[[[1.0, 1.0], [1.0, 1.0]]]]\n0.0 [[[[1.0]], [[1.0]]]]\n')
  })
})

describe('Module', () => {
  it('BatchNorm: state_dict entries, buffers, train and eval', () => {
    expect(out(`m = nn.BatchNorm2d(2)
print(list(m.state_dict().keys()))
print(m.running_mean, m.running_var, m.num_batches_tracked)
print([n for n, _ in m.named_parameters()], [n for n, _ in m.named_buffers()], len(list(m.buffers())))
print(m.state_dict()['num_batches_tracked'].dtype, m.training)
m(torch.arange(16.).view(2, 2, 2, 2))
print(m.num_batches_tracked.item(), m.state_dict()['num_batches_tracked'].item())
m.eval()
m(torch.arange(16.).view(2, 2, 2, 2))
print(m.num_batches_tracked.item(), m.training)
m2 = nn.BatchNorm2d(2)
print(m2.load_state_dict(m.state_dict()), m2.num_batches_tracked.item(), torch.equal(m2.running_mean, m.running_mean))
print(list(nn.BatchNorm1d(1, affine=False).state_dict().keys()), list(nn.BatchNorm1d(1, track_running_stats=False).state_dict().keys()))
net = nn.Sequential(nn.Linear(1, 1), nn.BatchNorm1d(1))
print(list(net.state_dict().keys()))
net.eval()
print(net[1].training, net.training)
`)).toBe(`['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
tensor([0., 0.]) tensor([1., 1.]) tensor(0)
['weight', 'bias'] ['running_mean', 'running_var', 'num_batches_tracked'] 3
torch.int64 True
1 1
1 False
<All keys matched successfully> 1 True
['running_mean', 'running_var', 'num_batches_tracked'] ['weight', 'bias']
['0.weight', '0.bias', '1.weight', '1.bias', '1.running_mean', '1.running_var', '1.num_batches_tracked']
False False
`)
  })
  it('register_buffer on a user module, and assigning to it', () => {
    expect(out(`class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.register_buffer('count', torch.tensor(0))
        self.register_buffer('scratch', torch.zeros(1), persistent=False)
        self.register_parameter('gain', nn.Parameter(torch.ones(1)))
    def forward(self, x):
        self.count = self.count + 1
        return self.fc(x) * self.gain
n = Net()
n(torch.zeros(1))
n(torch.zeros(1))
print(n.count, list(n.state_dict().keys()))
print([k for k, _ in n.named_buffers()], [k for k, _ in n.named_parameters()], [k for k, _ in n.named_modules()])
print(n.state_dict()['count'], [p.requires_grad for p in n.requires_grad_(False).parameters()], [p.requires_grad for p in n.requires_grad_().parameters()])
m = Net()
print(m.load_state_dict(n.state_dict()), m.count)
print(m.double().fc.weight.dtype, m.count.dtype, m.gain.dtype, m.float().gain.dtype)
print(m.to(torch.float64).gain.dtype, m.to('cpu').gain.dtype)
`)).toBe(`tensor(2) ['gain', 'count', 'fc.weight', 'fc.bias']
['count', 'scratch'] ['gain', 'fc.weight', 'fc.bias'] ['', 'fc']
tensor(2) [False, False, False] [True, True, True]
<All keys matched successfully> tensor(2)
torch.float64 torch.int64 torch.float64 torch.float32
torch.float64 torch.float64
`)
  })
  it('requires_grad_ freezes a layer: no gradient, and the optimiser leaves it', () => {
    expect(out(`torch.manual_seed(0)
m = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
m[0].requires_grad_(False)
before = m[0].weight.clone()
opt = torch.optim.SGD(m.parameters(), lr=0.1)
m(torch.ones(1)).sum().backward()
opt.step()
print(m[0].weight.grad, torch.equal(before, m[0].weight), m[1].weight.grad is not None)
`)).toBe('None True True\n')
  })
  it('load_state_dict reports what PyTorch reports', () => {
    expect(msg('nn.Linear(1, 1).load_state_dict({})')).toBe('RuntimeError: Error(s) in loading state_dict for Linear:\n\tMissing key(s) in state_dict: "weight", "bias". ')
    expect(msg('m = nn.Linear(1, 1)\nsd = m.state_dict()\nsd["x"] = torch.zeros(1)\nm.load_state_dict(sd)')).toBe('RuntimeError: Error(s) in loading state_dict for Linear:\n\tUnexpected key(s) in state_dict: "x". ')
    expect(msg('nn.Linear(1, 1).load_state_dict(nn.Linear(2, 1).state_dict())')).toBe('RuntimeError: Error(s) in loading state_dict for Linear:\n\tsize mismatch for weight: copying a param with shape torch.Size([1, 2]) from checkpoint, the shape in current model is torch.Size([1, 1]).')
    expect(out('r = nn.Linear(1, 1).load_state_dict({"weight": torch.zeros(1, 1), "x": torch.zeros(1)}, strict=False)\nprint(r, r.missing_keys, r.unexpected_keys)')).toBe("_IncompatibleKeys(missing_keys=['bias'], unexpected_keys=['x']) ['bias'] ['x']\n")
  })
  it('ModuleDict and the list containers', () => {
    expect(out(`d = nn.ModuleDict({'a': nn.Linear(1, 2)})
d['b'] = nn.ReLU()
print(len(d), 'a' in d, 'z' in d, list(d.keys()), [k for k in d], [k for k, _ in d.named_parameters()], len(list(d.parameters())))
print(d['b'], d.a.out_features, d['a'](torch.zeros(1)).shape)
print(len(d.values()), [k for k, v in d.items()], d.pop('b'), len(d))
d.update({'c': nn.Tanh()})
print(list(d.keys()))
l = nn.ModuleList([nn.Linear(1, 1)])
l.append(nn.ReLU())
l.extend([nn.Tanh()])
print(len(l), l[-1], l[0:2][1], len(l[1:]))
s = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Tanh())
print(s[1:], s[-1])
s[1] = nn.Sigmoid()
s.add_module('extra', nn.Identity())
print(list(k for k, _ in s.named_children()), s.extra)
`)).toBe(`2 True False ['a', 'b'] ['a', 'b'] ['a.weight', 'a.bias'] 2
ReLU() 2 torch.Size([2])
2 ['a', 'b'] ReLU() 1
['a', 'c']
3 Tanh() ReLU() 2
Sequential(
  (1): ReLU()
  (2): Tanh()
) Tanh()
['0', '1', '2', 'extra'] Identity()
`)
    expect(msg('nn.ModuleDict({})["x"]')).toBe("KeyError: 'x'")
    expect(msg('nn.ModuleList([])(torch.zeros(1))')).toBe('NotImplementedError: Module [ModuleList] is missing the required "forward" function')
    expect(msg('nn.ModuleDict({})(torch.zeros(1))')).toBe('NotImplementedError: Module [ModuleDict] is missing the required "forward" function')
  })
  it('assigning parameters, tying weights, apply', () => {
    expect(out(`m = nn.Linear(2, 1)
m.weight = nn.Parameter(torch.ones(1, 2))
m.bias = None
print(m(torch.ones(2)), [k for k, _ in m.named_parameters()], m)
a = nn.Linear(2, 2)
b = nn.Linear(2, 2)
b.weight = a.weight
net = nn.Sequential(a, b)
print(len(list(net.parameters())), len(net.state_dict()))
seen = []
def init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.zeros_(layer.bias)
        nn.init.ones_(layer.weight)
    seen.append(type(layer).__name__)
nn.Sequential(nn.Linear(1, 1), nn.Sequential(nn.ReLU())).apply(init)
print(seen)
`)).toBe(`tensor([2.], grad_fn=<SqueezeBackward4>) ['weight'] Linear(in_features=2, out_features=1, bias=False)
3 4
['Linear', 'ReLU', 'Sequential', 'Sequential']
`)
    expect(msg('nn.Linear(1, 1).weight = 3')).toBe("TypeError: cannot assign 'int' as parameter 'weight' (torch.nn.Parameter or None expected)")
    expect(msg('m = nn.Linear(1, 1)\nm.weight = torch.ones(1, 1)')).toBe("TypeError: cannot assign 'torch.FloatTensor' as parameter 'weight' (torch.nn.Parameter or None expected)")
  })
  it('torch.save keeps a snapshot, for tensors, state dicts and whole models', () => {
    expect(out(`torch.manual_seed(0)
m = nn.Linear(1, 1)
torch.save(m.state_dict(), 'w.pt')
saved = m.weight.clone()
with torch.no_grad():
    m.weight.add_(1)
m.load_state_dict(torch.load('w.pt'))
print(torch.equal(saved, m.weight))
torch.save(m, 'm.pt')
with torch.no_grad():
    m.weight.add_(1)
n = torch.load('m.pt', weights_only=False)
print(torch.equal(saved, n.weight), n, isinstance(n, nn.Linear), n.weight is m.weight)
n(torch.zeros(1))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, x):
        return self.fc(x)
p = Net()
w = p.fc.weight.clone()
torch.save(p, 'p.pt')
with torch.no_grad():
    p.fc.weight.add_(1)
q = torch.load('p.pt', weights_only=False)
print(torch.equal(w, q.fc.weight), q(torch.zeros(1)).shape, q.training, q)
torch.save({'x': torch.ones(1), 'n': 3}, 'd.pt')
print(torch.load('d.pt'))
big = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(2, 2))
lstm = nn.LSTM(2, 2)
torch.save(big, 'b.pt')
torch.save(lstm, 'l.pt')
keep = big[0].weight.clone()
with torch.no_grad():
    big[0].weight.add_(1)
    big[1].running_mean.add_(1)
    lstm.weight_ih_l0.add_(1)
b = torch.load('b.pt', weights_only=False)
print(torch.equal(b[0].weight, keep), b[1].running_mean.sum().item(), b[0].kernel_size, b(torch.zeros(2, 1, 4, 4)).shape, b[0].weight is big[0].weight)
print(torch.equal(torch.load('l.pt', weights_only=False).weight_ih_l0, lstm.weight_ih_l0), torch.load('l.pt', weights_only=False)(torch.zeros(3, 1, 2))[0].shape)
`)).toBe(`True
True Linear(in_features=1, out_features=1, bias=True) True False
True torch.Size([1]) True Net(
  (fc): Linear(in_features=1, out_features=1, bias=True)
)
{'x': tensor([1.]), 'n': 3}
True 0.0 (3, 3) torch.Size([2, 2]) False
False torch.Size([3, 1, 2])
`)
  })
})

describe('nn.init', () => {
  const STATS = `torch.manual_seed(0)
t = torch.empty(200, 100)
def info(t):
    return round(t.mean().item(), 2), round(t.std().item(), 3), round(t.abs().max().item(), 3)
`
  it.each([
    ['xavier_uniform_(t)', Math.sqrt(2 / 300), Math.sqrt(3) * Math.sqrt(2 / 300)],
    ['xavier_uniform_(t, gain=2.0)', 2 * Math.sqrt(2 / 300), 2 * Math.sqrt(3) * Math.sqrt(2 / 300)],
    ['kaiming_uniform_(t)', Math.sqrt(2 / (1 + 0)) / 10, Math.sqrt(3) * Math.SQRT2 / 10],
    ['kaiming_uniform_(t, mode="fan_out", nonlinearity="relu")', Math.SQRT2 / Math.sqrt(200), Math.sqrt(3) * Math.SQRT2 / Math.sqrt(200)],
    ['kaiming_uniform_(t, a=1.0)', 1 / 10, Math.sqrt(3) / 10],
    ['uniform_(t, -2.0, 2.0)', 4 / Math.sqrt(12), 2],
  ])('%s has the documented spread and bound', (call, std, bound) => {
    const [, got, max] = JSON.parse(out(`${STATS}nn.init.${call}\nprint(list(info(t)))`))
    expect(Math.abs(got - std) / std).toBeLessThan(0.03)
    expect(max).toBeLessThanOrEqual(bound + 1e-3)
    expect(max).toBeGreaterThan(bound * 0.98)
  })
  it.each([
    ['xavier_normal_(t)', Math.sqrt(2 / 300)],
    ['kaiming_normal_(t)', Math.SQRT2 / 10],
    ['kaiming_normal_(t, nonlinearity="tanh")', 5 / 3 / 10],
    ['normal_(t, 0.0, 0.5)', 0.5],
    ['trunc_normal_(t, std=0.5, a=-5.0, b=5.0)', 0.5],
  ])('%s has the documented spread', (call, std) => {
    const [mean, got] = JSON.parse(out(`${STATS}nn.init.${call}\nprint(list(info(t)))`))
    expect(Math.abs(got - std) / std).toBeLessThan(0.03)
    expect(Math.abs(mean)).toBeLessThan(0.05)
  })
  it('the fixed ones, and they may touch a parameter', () => {
    expect(out(`m = nn.Linear(2, 2)
print(nn.init.zeros_(m.bias), nn.init.ones_(m.bias), nn.init.constant_(m.bias, 3.0), nn.init.eye_(m.weight), m.weight.requires_grad, m.weight.is_leaf)
print(nn.init.calculate_gain('relu'), nn.init.calculate_gain('tanh'), nn.init.calculate_gain('linear'), nn.init.calculate_gain('leaky_relu', 0.0), nn.init.calculate_gain('selu'))
nn.init.trunc_normal_(m.weight, a=-0.1, b=0.1)
print(m.weight.abs().max().item() <= 0.1 + 1e-6)
`)).toBe(`Parameter containing:
tensor([3., 3.], requires_grad=True) Parameter containing:
tensor([3., 3.], requires_grad=True) Parameter containing:
tensor([3., 3.], requires_grad=True) Parameter containing:
tensor([[1., 0.],
        [0., 1.]], requires_grad=True) True True
1.4142135623730951 1.6666666666666667 1 1.4142135623730951 0.75
True
`)
    expect(msg('nn.init.xavier_uniform_(torch.zeros(3))')).toBe('ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions')
    expect(msg('nn.init.kaiming_normal_(torch.zeros(2, 2), mode="x")')).toBe("ValueError: Mode x not supported, please use one of ['fan_in', 'fan_out']")
    expect(msg('nn.init.calculate_gain("x")')).toBe('ValueError: Unsupported nonlinearity x')
  })
})

describe('torch.utils.data', () => {
  const IMPORT = 'from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset, ConcatDataset, random_split, SequentialSampler, RandomSampler, SubsetRandomSampler, WeightedRandomSampler\n'
  it('DataLoader: batch_size, drop_last, shuffle', () => {
    expect(out(`${IMPORT}torch.manual_seed(0)
ds = TensorDataset(torch.arange(10.), torch.arange(10))
print(len(ds), ds[2], ds[1:3], len(ds.tensors))
print([len(y) for x, y in DataLoader(ds, batch_size=4)], len(DataLoader(ds, batch_size=4)))
print([len(y) for x, y in DataLoader(ds, batch_size=4, drop_last=True)], len(DataLoader(ds, batch_size=4, drop_last=True)))
print(len(DataLoader(ds, 3)), DataLoader(ds, 3).batch_size, len(DataLoader(ds, batch_size=3).dataset))
seen = []
for x, y in DataLoader(ds, batch_size=3, shuffle=True):
    seen += y.tolist()
print(sorted(seen), seen != list(range(10)))
g = torch.Generator().manual_seed(7)
a = [y.tolist() for x, y in DataLoader(ds, batch_size=5, shuffle=True, generator=g)]
g.manual_seed(7)
print(a == [y.tolist() for x, y in DataLoader(ds, batch_size=5, shuffle=True, generator=g)])
batch = [b for b in DataLoader(TensorDataset(torch.arange(2)), batch_size=2)][0]
print(type(batch).__name__, len(batch), batch[0])
print(len(list(DataLoader(ds, batch_size=4, num_workers=2, pin_memory=True))))
`)).toBe(`10 (tensor(2.), tensor(2)) (tensor([1., 2.]), tensor([1, 2])) 2
[4, 4, 2] 3
[4, 4] 2
4 3 10
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] True
True
list 1 tensor([0, 1])
3
`)
  })
  it('a custom Dataset and default_collate', () => {
    expect(out(`${IMPORT}class D(Dataset):
    def __init__(self):
        self.n = 3
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return torch.tensor([float(i)]), i, float(i) / 2, 'abc'[i], {'k': i}, (i, i + 1)
d = D()
print(isinstance(d, Dataset), len(DataLoader(d, batch_size=2)))
for b in DataLoader(d, batch_size=2):
    print(b)
print([b for b in DataLoader(TensorDataset(torch.ones(2, 2)), batch_size=2, collate_fn=lambda s: len(s))])
`)).toBe(`True 2
[tensor([[0.],
        [1.]]), tensor([0, 1]), tensor([0.0000, 0.5000], dtype=torch.float64), ('a', 'b'), {'k': tensor([0, 1])}, [tensor([0, 1]), tensor([1, 2])]]
[tensor([[2.]]), tensor([2]), tensor([1.], dtype=torch.float64), ('c',), {'k': tensor([2])}, [tensor([2]), tensor([3])]]
[2]
`)
  })
  it('random_split, Subset, ConcatDataset', () => {
    expect(out(`${IMPORT}ds = TensorDataset(torch.arange(10))
a, b = random_split(ds, [7, 3])
print(len(a), len(b), sorted(a.indices + b.indices), a.dataset is ds, isinstance(a, Dataset))
print(sorted(int(a[k][0]) for k in range(7)) == sorted(a.indices), len(DataLoader(a, batch_size=2)))
p, q, r = random_split(ds, [0.5, 0.3, 0.2])
print(len(p), len(q), len(r))
p, q = random_split(TensorDataset(torch.arange(5)), [0.5, 0.5])
print(len(p), len(q))
g = torch.Generator().manual_seed(1)
one = random_split(ds, [5, 5], generator=g)[0].indices
g.manual_seed(1)
print(one == random_split(ds, [5, 5], generator=g)[0].indices)
s = Subset(ds, [8, 1])
print(s[0], s[-1], len(s), s[[0, 1]])
c = ConcatDataset([s, TensorDataset(torch.tensor([100]))])
print(len(c), c[1], c[2], c[-1])
print([y.tolist() for y, in DataLoader(c, batch_size=2)])
`)).toBe(`7 3 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] True True
True 4
5 3 2
3 2
True
(tensor(8),) (tensor(1),) 2 (tensor([8, 1]),)
3 (tensor(1),) (tensor(100),) (tensor(100),)
[[8, 1], [100]]
`)
    expect(msg(`${IMPORT}random_split(TensorDataset(torch.arange(3)), [1, 1])`)).toBe('ValueError: Sum of input lengths does not equal the length of the input dataset!')
    expect(msg(`${IMPORT}TensorDataset(torch.zeros(2), torch.zeros(3))`)).toBe('AssertionError: Size mismatch between tensors')
  })
  it('samplers', () => {
    expect(out(`${IMPORT}torch.manual_seed(0)
ds = TensorDataset(torch.arange(6))
print(list(SequentialSampler(ds)), sorted(list(RandomSampler(ds))), len(RandomSampler(ds)), sorted(list(SubsetRandomSampler([4, 1]))), len(list(RandomSampler(ds, replacement=True, num_samples=3))))
print([b[0].tolist() for b in DataLoader(ds, batch_size=2, sampler=SequentialSampler(ds))], len(DataLoader(ds, batch_size=4, sampler=SubsetRandomSampler([0, 1, 2]))))
print(list(WeightedRandomSampler([0., 1., 0.], 4)), len(WeightedRandomSampler([1., 1.], 5)))
print(sorted(b[0].item() for b in DataLoader(ds, sampler=SubsetRandomSampler([4, 1]))))
`)).toBe(`[0, 1, 2, 3, 4, 5] [0, 1, 2, 3, 4, 5] 6 [1, 4] 3
[[0, 1], [2, 3], [4, 5]] 1
[1, 1, 1, 1] 5
[1, 4]
`)
    expect(msg(`${IMPORT}ds = TensorDataset(torch.arange(6))\nDataLoader(ds, shuffle=True, sampler=SequentialSampler(ds))`)).toBe('ValueError: sampler option is mutually exclusive with shuffle')
    expect(msg(`${IMPORT}DataLoader(TensorDataset(torch.arange(6)), batch_size=0)`)).toBe('ValueError: batch_size should be a positive integer value, but got batch_size=0')
  })
})

describe('optimisers in use', () => {
  it('param_groups are live: writing lr changes the next step', () => {
    expect(out(`w = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)
(w * 1).backward()
opt.step()
print(round(w.item(), 6))
opt.param_groups[0]['lr'] = 0.5
for g in opt.param_groups:
    g['lr'] *= 2
opt.step()
print(round(w.item(), 6), opt.param_groups[0]['lr'], sorted(opt.param_groups[0].keys())[:3])
`)).toBe("0.9\n-0.1 1.0 ['dampening', 'differentiable', 'foreach']\n")
  })
  it('parameter groups with their own settings, and named_parameters', () => {
    expect(out(`a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.SGD([{'params': [a]}, {'params': [b], 'lr': 0.5}], lr=0.1, momentum=0.0)
(a + b).backward()
opt.step()
print(round(a.item(), 4), round(b.item(), 4), len(opt.param_groups))
c = torch.tensor(1.0, requires_grad=True)
opt.add_param_group({'params': c, 'lr': 1.0})
print(len(opt.param_groups), opt.param_groups[2]['momentum'])
m = nn.Linear(1, 1)
print(len(torch.optim.Adam(m.named_parameters()).param_groups[0]['params']))
`)).toBe('0.9 0.5 2\n3 0.0\n2\n')
  })
  it('a parameter without a gradient does not count a step, and step(closure) evaluates it', () => {
    expect(out(`a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.Adam([a, b], lr=0.1)
a.backward()
opt.step()
print(round(a.item(), 4), b.item())
b.backward()
opt.zero_grad()
b.backward(retain_graph=True) if False else None
c = torch.tensor(2.0, requires_grad=True)
sgd = torch.optim.SGD([c], lr=0.1)
def closure():
    sgd.zero_grad()
    loss = c ** 2
    loss.backward()
    return loss
print(sgd.step(closure), round(c.item(), 4))
print(list(opt.state_dict().keys()), sorted(opt.state_dict()['state'][0].keys()), opt.state_dict()['state'][0]['step'], opt.state_dict()['param_groups'][0]['params'])
`)).toBe("0.9 1.0\ntensor(4., grad_fn=<PowBackward0>) 1.6\n['state', 'param_groups'] ['exp_avg', 'exp_avg_sq', 'step'] tensor(1.) [0, 1]\n")
  })
  it('state_dict round-trips momentum', () => {
    const train = (restore: string) => out(`w = torch.tensor([1.0], requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1, momentum=0.9)
for k in range(2):
    opt.zero_grad()
    (w ** 2).sum().backward()
    opt.step()
sd = opt.state_dict()
${restore}
for k in range(2):
    opt.zero_grad()
    (w ** 2).sum().backward()
    opt.step()
print(round(w.item(), 5))
`)
    expect(train('opt = torch.optim.SGD([w], lr=0.5)\nopt.load_state_dict(sd)')).toBe(train('pass'))
  })
  it('zero_grad and set_to_none', () => {
    expect(out(`w = torch.ones(1, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)
w.backward(torch.ones(1))
opt.zero_grad(set_to_none=False)
print(w.grad)
opt.zero_grad()
print(w.grad)
m = nn.Linear(1, 1)
m(torch.ones(1)).backward()
m.zero_grad(set_to_none=False)
print(m.bias.grad)
m.zero_grad()
print(m.bias.grad)
`)).toBe('tensor([0.])\nNone\ntensor([0.])\nNone\n')
  })
  it('bad settings', () => {
    const p = '[nn.Parameter(torch.zeros(1))]'
    expect(msg(`torch.optim.SGD(${p}, lr=-1)`)).toBe('ValueError: Invalid learning rate: -1')
    expect(msg(`torch.optim.Adam(${p}, betas=(1.0, 0.9))`)).toBe('ValueError: Invalid beta parameter at index 0: 1.0')
    expect(msg(`torch.optim.SGD(${p}, lr=0.1, nesterov=True)`)).toBe('ValueError: Nesterov momentum requires a momentum and zero dampening')
    expect(msg('torch.optim.SGD([], lr=0.1)')).toBe('ValueError: optimizer got an empty parameter list')
    expect(msg('torch.optim.SGD(torch.ones(1), lr=0.1)')).toBe('TypeError: params argument given to the optimizer should be an iterable of Tensors or dicts, but got torch.FloatTensor')
    expect(msg(`x = torch.ones(1, requires_grad=True)\ntorch.optim.SGD([x * 2], lr=0.1)`)).toBe("ValueError: can't optimize a non-leaf Tensor")
    expect(msg(`x = nn.Parameter(torch.ones(1))\ntorch.optim.SGD([x, x], lr=0.1).add_param_group({'params': x})`)).toMatch('some parameters appear in more than one parameter group')
    expect(msg(`torch.optim.SGD(${p}, lr=0.1, differentiable=True)`)).toMatch('NotImplementedError')
  })
  it('schedulers: attributes and the group they drive', () => {
    expect(out(`from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
w = nn.Parameter(torch.zeros(1))
opt = torch.optim.Adam([w], lr=0.1)
s = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
print(s.last_epoch, s.base_lrs, s.get_last_lr(), opt.param_groups[0]['initial_lr'])
opt.step()
s.step()
print(s.last_epoch, opt.param_groups[0]['lr'])
opt.param_groups[0]['lr'] = 0.3
opt.step()
s.step()
print(opt.param_groups[0]['lr'], s.base_lrs)
p = ReduceLROnPlateau(torch.optim.SGD([w], lr=0.1), patience=0)
p.step(1.0)
p.step(torch.tensor(1.0))
print(p.get_last_lr(), p.best, p.num_bad_epochs)
`)).toBe('0 [0.1] [0.1] 0.1\n1 0.05\n0.15 [0.1]\n[0.010000000000000002] 1.0 0\n')
    expect(msg('torch.optim.lr_scheduler.StepLR(nn.Linear(1, 1), 1)')).toBe('TypeError: Linear is not an Optimizer')
    expect(msg('w = nn.Parameter(torch.zeros(1))\ntorch.optim.lr_scheduler.ReduceLROnPlateau(torch.optim.SGD([w], lr=0.1), factor=1.0)')).toBe('ValueError: Factor should be < 1.0.')
  })
  it('clip_grad_norm_ takes the parameters of a model', () => {
    expect(out('m = nn.Linear(1, 1, bias=False)\nm(torch.tensor([3.])).backward()\nprint(nn.utils.clip_grad_norm_(m.parameters(), 0.5).item() > 0, m.weight.grad.norm().item() <= 0.5 + 1e-5)')).toBe('True True\n')
  })
})

describe('whole models train', () => {
  const TRAIN = (model: string, x: string, y: string, loss = 'nn.CrossEntropyLoss()', opt = 'torch.optim.Adam(model.parameters(), lr=0.05)', steps = 40, forward = 'model(x)') => out(`torch.manual_seed(1)
x = ${x}
y = ${y}
model = ${model}
opt = ${opt}
lossf = ${loss}
first = None
for k in range(${steps}):
    opt.zero_grad()
    l = lossf(${forward}, y)
    l.backward()
    opt.step()
    if first is None:
        first = l.item()
print(l.item() < first / 2)`).trim()
  const CH = 'torch.tensor([[[1., 1., 1., 0., 0., 0.]], [[0., 0., 0., 1., 1., 1.]]])'
  const IMG2 = 'torch.cat([torch.ones(4, 1, 4, 4), torch.zeros(4, 1, 4, 4)])'
  const Y2 = 'torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])'
  it.each([
    ['Conv1d, MaxPool1d, Flatten', TRAIN('nn.Sequential(nn.Conv1d(1, 2, 3), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(4, 2))', CH, 'torch.tensor([0, 1])')],
    ['Conv2d, BatchNorm2d, Dropout2d, AdaptiveAvgPool2d', TRAIN('nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.BatchNorm2d(3), nn.GELU(), nn.Dropout2d(0.1), nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(12, 2))', IMG2, Y2)],
    ['Linear, BatchNorm1d, LayerNorm, ELU, AdamW', TRAIN('nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8), nn.ELU(), nn.LayerNorm(8), nn.Softplus(), nn.Linear(8, 2))', 'torch.tensor([[0., 1.], [1., 0.], [0., 0.], [1., 1.]])', 'torch.tensor([0, 0, 1, 1])', 'nn.CrossEntropyLoss()', 'torch.optim.AdamW(model.parameters(), lr=0.05)')],
    ['ConvTranspose2d, Upsample, ZeroPad2d: an autoencoder', TRAIN('nn.Sequential(nn.Conv2d(1, 2, 3, stride=2, padding=1), nn.ReLU(), nn.ConvTranspose2d(2, 2, 2, stride=2), nn.ReLU(), nn.Upsample(scale_factor=2), nn.ZeroPad2d(1), nn.Conv2d(2, 1, 3, stride=2))', 'torch.arange(16.).view(1, 1, 4, 4) / 16', 'torch.arange(16.).view(1, 1, 4, 4) / 16', 'nn.MSELoss()', 'torch.optim.RMSprop(model.parameters(), lr=0.01)', 60)],
    ['Embedding and pooling', TRAIN('nn.Sequential(nn.Embedding(4, 3), nn.Flatten(), nn.Linear(6, 2))', 'torch.tensor([[0, 1], [2, 3]])', 'torch.tensor([0, 1])')],
    ['RNN', TRAIN('nn.RNN(2, 4, batch_first=True)', 'torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])', 'torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]], [[0., 1., 0., 0.], [1., 0., 0., 0.]]])', 'nn.MSELoss()', 'torch.optim.Adagrad(model.parameters(), lr=0.1)', 40, 'model(x)[0]')],
    ['LSTM', TRAIN('nn.LSTM(2, 4, batch_first=True)', 'torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])', 'torch.tensor([[[.5, 0., 0., 0.], [0., .5, 0., 0.]], [[0., .5, 0., 0.], [.5, 0., 0., 0.]]])', 'nn.SmoothL1Loss()', 'torch.optim.Adam(model.parameters(), lr=0.05)', 40, 'model(x)[0]')],
    ['GRU', TRAIN('nn.GRU(2, 4, 2, bidirectional=True)', 'torch.tensor([[[1., 0.]], [[0., 1.]]])', 'torch.zeros(2, 1, 8) + 0.25', 'nn.HuberLoss()', 'torch.optim.Adam(model.parameters(), lr=0.05)', 40, 'model(x)[0]')],
  ])('%s', (_name, result) => expect(result).toBe('True'))
  it('a user module with nn.init, a scheduler, clipping and a DataLoader, all together', () => {
    expect(out(`from torch.utils.data import TensorDataset, DataLoader, random_split
torch.manual_seed(0)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 2))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).repeat(5, 1)
Y = torch.tensor([0, 1, 1, 0]).repeat(5)
train, val = random_split(TensorDataset(X, Y), [16, 4])
loader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)
model = Net()
opt = torch.optim.AdamW(model.parameters(), lr=0.05)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
for epoch in range(30):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        lossf(model(xb), yb).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    sch.step()
model.eval()
with torch.no_grad():
    pred = model(X).argmax(dim=1)
print((pred == Y).float().mean().item() > 0.9, sch.get_last_lr()[0] < 1e-3)
`)).toBe('True True\n')
  })
})

describe('the step list and the shape trace', () => {
  const calls = (src: string) => run(src).calls.map((c) => [c.api, c.kind, c.axis, c.groups ? c.groups.length : 0])
  it('new tensor functions appear as steps, with the input behind each output cell', () => {
    expect(calls('a = torch.tensor([[3., 1.], [2., 4.]])\nb = a.flip(0)\nc = a.sort(dim=1)\nd = a.cumsum(1)\ne = a.prod(dim=0)\nf = torch.log2(a)\ng = a.gather(1, torch.tensor([[0], [1]]))\nh = a.unique()\nk = torch.linalg.inv(a)').slice(1)).toEqual([
      ['Tensor.flip', 'move', null, 4],
      ['Tensor.sort', 'sort', 1, 4],
      ['Tensor.cumsum', 'scan', 1, 0],
      ['Tensor.prod', 'reduce', 0, 0],
      ['Tensor.log2', 'elementwise', null, 0],
      ['torch.tensor', 'create', null, 0],
      ['Tensor.gather', 'move', null, 2],
      ['Tensor.unique', 'unique', null, 0],
      ['torch.linalg.inv', 'linalg', null, 0],
    ])
    const r = run('a = torch.tensor([[3., 1.], [2., 4.]])\nb = a.flip(0)\nc = torch.bincount(torch.tensor([1, 1]))\nd = F.elu(a)\ne = a.expand(2, 2, 2)')
    const by = (api: string) => r.calls.find((c) => c.api === api)!
    expect(by('Tensor.flip').groups).toEqual([[[0, 2]], [[0, 3]], [[0, 0]], [[0, 1]]])
    expect(by('Tensor.bincount').groups).toEqual([[], [[0, 0], [0, 1]]])
    expect(by('F.elu').kind).toBe('elementwise')
    expect(by('Tensor.expand').groups?.length).toBe(8)
    expect(by('torch.tensor').kind).toBe('create')
  })
  it('joins and pieces say where each cell came from', () => {
    const r = run('a = torch.tensor([1., 2.])\nb = torch.tensor([3.])\nc = torch.cat([a, b])\nd = torch.hstack((a, b))\ne = torch.stack([a, a])\nf, g = a.split(1)\nh, k = torch.meshgrid(a, b, indexing="ij")\nm = a.median(0)')
    const by = (api: string) => r.calls.filter((c) => c.api === api)
    expect(by('torch.cat')[0].groups).toEqual([[[0, 0]], [[0, 1]], [[1, 0]]])
    expect(by('torch.hstack')[0].operands.map((o) => o.label)).toEqual(['a', 'b'])
    expect(by('torch.stack')[0].groups).toEqual([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    expect(by('Tensor.split').map((c) => c.groups)).toEqual([[[[0, 0]]], [[[0, 1]]]])
    expect(by('torch.meshgrid').map((c) => c.groups)).toEqual([[[[0, 0]], [[0, 1]]], [[[0, 0]], [[0, 0]]]])
    expect(by('Tensor.median')[0].axis).toBe(0)
  })
  it('random creators and nn.init are steps too', () => {
    const r = run('a = torch.randperm(3)\nb = torch.logspace(0, 1, 2)\nc = nn.init.zeros_(torch.empty(2, 2))')
    expect(r.calls.map((c) => [c.api, c.kind])).toEqual([['torch.randperm', 'random'], ['torch.logspace', 'create'], ['torch.empty', 'create'], ['nn.init.zeros_', 'random']])
  })
  it('a forward pass lists the new layers with their shapes', () => {
    const r = run(`class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 2, 3, padding=1)
        self.bn = nn.BatchNorm1d(2)
        self.pool = nn.MaxPool1d(2)
        self.rnn = nn.LSTM(2, 3, batch_first=True)
        self.fc = nn.Linear(3, 1)
    def forward(self, x):
        x = self.pool(self.bn(self.conv(x)))
        out, _ = self.rnn(x.transpose(1, 2))
        return self.fc(out[:, -1])
m = Net()
y = m(torch.zeros(2, 1, 8))`)
    expect(r.error).toBeNull()
    const flow = r.events.find((e) => e.type === 'flow')
    expect(flow && flow.type === 'flow' && flow.rows.map((x) => [x.name, x.input, x.output])).toEqual([
      ['conv', [2, 1, 8], [2, 2, 8]],
      ['bn', [2, 2, 8], [2, 2, 8]],
      ['pool', [2, 2, 8], [2, 2, 4]],
      ['rnn', [2, 4, 2], [2, 4, 3]],
      ['fc', [2, 3], [2, 1]],
    ])
    expect(flow && flow.type === 'flow' && flow.total).toBe(8 + 4 + 4 * 3 * 2 + 4 * 3 * 3 + 12 + 12 + 4)
  })
  it('backward() shows the layers grad_fn names in the graph', () => {
    const r = run('m = nn.BatchNorm1d(2)\nout = m(torch.ones(3, 2) * torch.tensor([[1.], [2.], [3.]]))\nout.sum().backward()')
    const g = r.events.find((e) => e.type === 'graph')
    expect(g && g.type === 'graph' && g.nodes.map((n) => n.op)).toContain('NativeBatchNormBackward0')
    expect(g && g.type === 'graph' && g.nodes.map((n) => n.label)).toEqual(expect.arrayContaining(['weight', 'bias']))
  })
})

describe('parameters that change the result are never silently ignored', () => {
  it.each([
    ['F.interpolate(torch.zeros(1, 1, 2, 2), scale_factor=2, mode="bilinear", antialias=True)', 'NotImplementedError: F.interpolate: antialias=True is not available in this sandbox'],
    ['torch.topk(torch.zeros(2), 1, out=(torch.zeros(1), torch.zeros(1, dtype=torch.int64)))', 'NotImplementedError: topk(): the out= argument is not available in this sandbox'],
    ['nn.EmbeddingBag(2, 2, max_norm=1.0)', 'NotImplementedError: EmbeddingBag: max_norm= is not available in this sandbox'],
    ['nn.EmbeddingBag(2, 2, include_last_offset=True)', 'NotImplementedError: EmbeddingBag: include_last_offset=True is not available in this sandbox'],
    ['nn.MultiheadAttention(2, 1, add_bias_kv=True)', 'NotImplementedError: MultiheadAttention: add_bias_kv=True is not available in this sandbox'],
    ['nn.MultiheadAttention(2, 1, kdim=3)', 'NotImplementedError: MultiheadAttention: kdim and vdim different from embed_dim are not available in this sandbox'],
    ['torch.optim.Adam([nn.Parameter(torch.ones(1))], differentiable=True)', 'NotImplementedError: Adam: differentiable=True is not available in this sandbox'],
    ['torch.optim.LBFGS([nn.Parameter(torch.ones(1))], line_search_fn="strong_wolfe")', "NotImplementedError: LBFGS: line_search_fn='strong_wolfe' is not available in this sandbox"],
    ['F.grid_sample(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 2), mode="bicubic")', "NotImplementedError: F.grid_sample: mode='bicubic' is not available in this sandbox"],
    ['torch.ones(1, requires_grad=True).sum().backward(inputs=[torch.ones(1)])', 'NotImplementedError: backward(inputs=…) is not available in this sandbox'],
    ['torch.linalg.eig(torch.tensor([[0., 1.], [-1., 0.]]))', 'NotImplementedError: linalg.eig: eigenvectors of a matrix with complex eigenvalues are not available in this sandbox; torch.linalg.eigvals gives the values'],
    ['x = torch.ones(2, 2, requires_grad=True)\ntorch.linalg.qr(x)[0].sum().backward()', 'NotImplementedError: the gradient through linalg_qr is not available in this sandbox'],
    ['x = torch.ones(1, requires_grad=True)\ng, = torch.autograd.grad(torch.sinh(x).sum(), x, create_graph=True)', 'NotImplementedError: create_graph=True: the gradient through SinhBackward0 cannot be differentiated again in this sandbox'],
  ])('%s', (src, want) => expect(msg(src)).toBe(want))
  it('an unknown keyword is an error', () => {
    expect(msg('torch.sort(torch.ones(1), wibble=1)')).toMatch(/^TypeError: sort\(\) received an invalid combination of arguments - got \(Tensor, wibble=int\), but expected/)
    expect(msg('nn.Linear(1, 1, wibble=1)')).toBe("TypeError: Linear.__init__() got an unexpected keyword argument 'wibble'")
    expect(msg('nn.Conv2d(1, 1, 1, wibble=1)')).toBe("TypeError: Conv2d.__init__() got an unexpected keyword argument 'wibble'")
    expect(msg('torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=1, wibble=1)')).toBe("TypeError: SGD.__init__() got an unexpected keyword argument 'wibble'")
    expect(msg('F.mse_loss(torch.ones(1), torch.ones(1), wibble=1)')).toBe("TypeError: mse_loss() got an unexpected keyword argument 'wibble'")
    expect(msg('torch.utils.data.DataLoader([1], wibble=1)')).toBe("TypeError: DataLoader.__init__() got an unexpected keyword argument 'wibble'")
  })
})

describe('errors students meet', () => {
  it.each([
    ['torch.ones(2, 3).expand(4, 3)', 'RuntimeError: The expanded size of the tensor (4) must match the existing size (2) at non-singleton dimension 0.  Target sizes: [4, 3].  Tensor sizes: [2, 3]'],
    ['torch.ones(2, 3).expand(3)', 'RuntimeError: expand(torch.FloatTensor{[2, 3]}, size=[3]): the number of sizes provided (1) must be greater or equal to the number of dimensions in the tensor (2)'],
    ['torch.ones(3).expand(-1, 3)', "RuntimeError: The expanded size of the tensor (-1) isn't allowed in a leading, non-existing dimension 0"],
    ['torch.ones(2, 3).repeat(2)', 'RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor'],
    ['torch.ones(3).narrow(0, 2, 5)', 'RuntimeError: start (2) + length (5) exceeds dimension size (3).'],
    ['torch.ones(3).split([1, 1])', "RuntimeError: split_with_sizes expects split_sizes to sum exactly to 3 (input tensor's size at dimension 0), but got split_sizes=[1, 1]"],
    ['torch.ones(3).split(-1)', 'RuntimeError: split expects split_size be non-negative, but got split_size=-1'],
    ['torch.ones(3).chunk(0)', 'RuntimeError: chunk expects `chunks` to be greater than 0, got: 0'],
    ['torch.ones(3).topk(5)', 'RuntimeError: selected index k out of range'],
    ['torch.ones(2, 2).gather(1, torch.tensor([[0.]]))', 'RuntimeError: gather(): Expected dtype int32/int64 for index'],
    ['torch.ones(2, 2).gather(1, torch.tensor([0]))', 'RuntimeError: Index tensor must have the same number of dimensions as input tensor'],
    ['torch.ones(2, 2).gather(1, torch.tensor([[5]]))', 'RuntimeError: index 5 is out of bounds for dimension 1 with size 2'],
    ['torch.ones(2, 2).gather(1, torch.zeros(3, 1, dtype=torch.int64))', 'RuntimeError: Size does not match at dimension 0 expected index [3, 1] to be no larger than self [2, 2] apart from dimension 1'],
    ['torch.ones(2).index_select(0, torch.tensor([5]))', 'IndexError: index out of range in self'],
    ['torch.ones(2).masked_select(torch.tensor([1, 0]))', 'RuntimeError: masked_select: expected BoolTensor for mask'],
    ['torch.ones(2).masked_fill(torch.tensor([1, 0]), 0.)', 'RuntimeError: masked_fill_ only supports boolean masks, but got mask with dtype long long'],
    ['torch.tensor([-1]).bincount()', 'RuntimeError: bincount only supports 1-d non-negative integral inputs.'],
    ['torch.tensor([1.]).bincount()', `NotImplementedError: "bincount_cpu" not implemented for 'Float'`],
    ['torch.tensor([1.]).cumsum()', 'TypeError: cumsum() missing 1 required positional arguments: "dim"'],
    ['torch.ones(2, 2).flip(0, 0)', 'RuntimeError: dim 0 appears multiple times in the list of dims'],
    ['torch.ones(2, 2).roll((1, 1), 0)', 'RuntimeError: shifts and dimensions must align. shifts: 2, dims:1'],
    ['torch.ones(2, 2, 2).trace()', 'RuntimeError: trace: expected a matrix, but got tensor with dim 3'],
    ['torch.ones(2, 2, 2).diag()', 'RuntimeError: diag(): Supports 1D or 2D tensors. Got 3D'],
    ['torch.ones(2).tril()', 'RuntimeError: tril: input tensor must have at least 2 dimensions'],
    ['torch.outer(torch.ones(2, 2), torch.ones(2))', 'RuntimeError: outer: Expected 1-D argument self, but got 2-D'],
    ['torch.inner(torch.ones(2), torch.ones(3))', 'RuntimeError: inner() the last dimension must match on both input tensors but got shapes [2] and [3]'],
    ['torch.dot(torch.ones(2, 2), torch.ones(2, 2))', 'RuntimeError: 1D tensors expected, but got 2D and 2D tensors'],
    ['torch.tensor([1, 2]) @ torch.tensor([1., 2.])', 'RuntimeError: dot : expected both vectors to have same dtype, but found Long and Float'],
    ['torch.matmul(torch.ones(2, 2, dtype=torch.int64), torch.ones(2, 2))', 'RuntimeError: expected m1 and m2 to have the same dtype, but got: long long != float'],
    ['torch.linalg.inv(torch.tensor([[1., 2.], [2., 4.]]))', '_LinAlgError: linalg.inv: The diagonal element 2 is zero, the inversion could not be completed because the input matrix is singular.'],
    ['torch.linalg.inv(torch.ones(2, 3))', 'RuntimeError: linalg.inv: A must be batches of square matrices, but they are 2 by 3 matrices'],
    ['torch.linalg.inv(torch.tensor([[1, 2], [3, 4]]))', 'RuntimeError: linalg.inv: Expected a floating point or complex tensor as input. Got Long'],
    ['torch.linalg.det(torch.ones(2, 3))', 'RuntimeError: linalg.det: A must be batches of square matrices, but they are 2 by 3 matrices'],
    ['torch.linalg.solve(torch.tensor([[1., 2.], [2., 4.]]), torch.ones(2))', '_LinAlgError: torch.linalg.solve: The solver failed because the input matrix is singular.'],
    ['torch.linalg.solve(torch.eye(2), torch.ones(2, dtype=torch.float64))', 'RuntimeError: linalg.solve: Expected A and B to have the same dtype, but found A of type Float and B of type Double instead'],
    ['torch.linalg.norm(torch.ones(2, 2, 2), ord=1)', 'RuntimeError: linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D. Got 3D.'],
    ['torch.norm(torch.tensor([1, 2]))', 'RuntimeError: linalg.vector_norm: Expected a floating point or complex tensor as input. Got Long'],
    ['torch.clamp(torch.ones(1))', "RuntimeError: torch.clamp: At least one of 'min' or 'max' must not be None"],
    ['torch.lerp(torch.tensor([1]), torch.tensor([2]), 0.5)', `NotImplementedError: "lerp_kernel_scalar" not implemented for 'Long'`],
    ['torch.ones(2).normal_(0., -1.)', 'RuntimeError: normal expects std >= 0.0, but found std -1'],
    ['torch.tensor([1]).normal_()', `NotImplementedError: "normal_kernel_cpu" not implemented for 'Long'`],
    ['torch.meshgrid(torch.ones(2, 2))', 'RuntimeError: torch.meshgrid: Expected 0D or 1D tensor in the tensor list but got:  1  1\n 1  1\n[ torch.FloatTensor{2,2} ]'],
    ['torch.meshgrid(torch.ones(2), indexing="x")', 'RuntimeError: torch.meshgrid: indexing must be one of "xy" or "ij", but received: x'],
    ['x = torch.ones(2)\nx[0:1] = torch.ones(3)', 'RuntimeError: The expanded size of the tensor (1) must match the existing size (3) at non-singleton dimension 0.  Target sizes: [1].  Tensor sizes: [3]'],
    ['nn.Linear(2, 1)(torch.tensor([[1, 2]]))', 'RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float'],
    ['nn.Linear(2, 1)(torch.ones(1, 2, dtype=torch.float64))', 'RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float'],
    ['nn.Conv2d(1, 1, 1)(torch.ones(1, 1, 2, 2, dtype=torch.float64))', 'RuntimeError: Input type (double) and bias type (float) should be the same'],
    ['nn.Conv2d(1, 1, 1, bias=False)(torch.ones(1, 1, 2, 2, dtype=torch.float64))', 'RuntimeError: expected scalar type Double but found Float'],
    ['nn.Conv2d(1, 1, 1)(torch.ones(1, 1, 2, 2, dtype=torch.uint8))', 'RuntimeError: Input type (unsigned char) and bias type (float) should be the same'],
    ['nn.Conv2d(1, 2, 1)(torch.ones(2, 2))', 'RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2, 2]'],
    ['nn.Conv1d(1, 2, 3)(torch.zeros(1, 2, 8))', 'RuntimeError: Given groups=1, weight of size [2, 1, 3], expected input[1, 2, 8] to have 1 channels, but got 2 channels instead'],
    ['nn.Conv1d(1, 2, 3)(torch.zeros(1, 1, 2))', "RuntimeError: Calculated padded input size per channel: (2). Kernel size: (3). Kernel size can't be greater than actual input size"],
    ['nn.Conv1d(1, 2, 3)(torch.zeros(2))', 'RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [2]'],
    ['nn.Conv2d(2, 4, 1, groups=2)(torch.zeros(1, 3, 2, 2))', 'RuntimeError: Given groups=2, weight of size [4, 1, 1, 1], expected input[1, 3, 2, 2] to have 2 channels, but got 3 channels instead'],
    ['nn.Conv2d(3, 3, 1, groups=2)', 'ValueError: in_channels must be divisible by groups'],
    ['nn.Conv2d(1, 1, 3, stride=2, padding="same")', "ValueError: padding='same' is not supported for strided convolutions"],
    ['nn.ConvTranspose2d(1, 2, 2)(torch.zeros(1, 3, 2, 2))', 'RuntimeError: Given transposed=1, weight of size [1, 2, 2, 2], expected input[1, 3, 2, 2] to have 1 channels, but got 3 channels instead'],
    ['nn.BatchNorm2d(2)(torch.zeros(2, 2))', 'ValueError: expected 4D input (got 2D input)'],
    ['nn.BatchNorm1d(2)(torch.zeros(2))', 'ValueError: expected 2D or 3D input (got 1D input)'],
    ['nn.BatchNorm1d(2)(torch.zeros(1, 2))', 'ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 2])'],
    ['nn.BatchNorm1d(2)(torch.zeros(2, 3))', 'RuntimeError: running_mean should contain 3 elements not 2'],
    ['nn.LayerNorm(3)(torch.zeros(2, 4))', 'RuntimeError: Given normalized_shape=[3], expected input with shape [*, 3], but got input of size[2, 4]'],
    ['nn.GroupNorm(3, 4)', 'ValueError: num_channels (4) must be divisible by num_groups (3)'],
    ['nn.Dropout2d(2.0)', 'ValueError: dropout probability has to be between 0 and 1, but got 2.0'],
    ['nn.Embedding(2, 2)(torch.tensor([5]))', 'IndexError: index out of range in self'],
    ['nn.Embedding(2, 2)(torch.tensor([.5]))', "RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)"],
    ['nn.LSTM(3, 4)(torch.zeros(3))', 'ValueError: LSTM: Expected input to be 2D or 3D, got 1D instead'],
    ['nn.GRU(3, 4)(torch.zeros(2, 1, 5))', 'RuntimeError: input.size(-1) must be equal to input_size. Expected 3, got 5'],
    ['nn.RNN(3, 4)(torch.zeros(2, 1, 3), torch.zeros(1, 2, 4))', 'RuntimeError: Expected hidden size (1, 1, 4), got [1, 2, 4]'],
    ['nn.LSTM(3, 4)(torch.zeros(2, 1, 3), (torch.zeros(2, 1, 4), torch.zeros(1, 1, 4)))', 'RuntimeError: Expected hidden[0] size (1, 1, 4), got [2, 1, 4]'],
    ['nn.RNN(3, 4)(torch.zeros(2, 1, 3, dtype=torch.float64))', 'ValueError: RNN input dtype (torch.float64) does not match weight dtype (torch.float32). Convert input: input.to(torch.float32), or convert model: model.to(torch.float64)'],
    ['nn.RNN(3, 4, nonlinearity="x")', "ValueError: Unknown nonlinearity 'x'. Select from 'tanh' or 'relu'."],
    ['nn.MultiheadAttention(5, 2)', 'AssertionError: embed_dim must be divisible by num_heads'],
    ['F.cross_entropy(torch.ones(2, 3), torch.tensor([0, 1]), label_smoothing=2.0)', 'RuntimeError: label_smoothing must be between 0.0 and 1.0. Got: 2'],
    ['F.cross_entropy(torch.ones(2, 3), torch.tensor([0, 1]), weight=torch.ones(2))', 'RuntimeError: weight tensor should be defined either for all 3 classes or no classes but got weight tensor of shape: [2]'],
    ['F.one_hot(torch.tensor([3]), 2)', 'RuntimeError: Class values must be smaller than num_classes.'],
    ['F.one_hot(torch.tensor([-1]))', 'RuntimeError: Class values must be non-negative.'],
    ['F.pad(torch.ones(2), (1,))', 'RuntimeError: Padding length must be divisible by 2'],
    ['F.pad(torch.ones(2), (1, 1, 1, 1))', 'RuntimeError: Padding length should be less than or equal to two times the input dimension but got padding length 4 and input of dimension 1'],
    ['F.pad(torch.ones(1, 1, 2, 2), (5, 5, 0, 0), mode="reflect")', 'RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (5, 5) at dimension 3 of input [1, 1, 2, 2]'],
    ['F.pad(torch.ones(1, 1, 2, 2), (5, 5, 0, 0), mode="circular")', 'RuntimeError: Padding value causes wrapping around more than once.'],
    ['F.pad(torch.ones(2), (1, 1), mode="x")', 'NotImplementedError: Unrecognised padding mode x'],
    ['F.interpolate(torch.zeros(1, 1, 2, 2))', 'ValueError: either size or scale_factor should be defined'],
    ['F.interpolate(torch.zeros(1, 1, 2, 2), size=2, scale_factor=2)', 'ValueError: only one of size or scale_factor should be defined'],
    ['F.interpolate(torch.zeros(1, 1, 2, 2), size=2, align_corners=True)', 'ValueError: align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear'],
    ['F.interpolate(torch.zeros(1, 1, 2, 2), size=2, mode="trilinear")', 'NotImplementedError: Got 4D input, but trilinear mode needs 5D input'],
    ['torch.ones(2).softmax()', 'TypeError: softmax() missing 1 required positional arguments: "dim"'],
    ['y = torch.tensor(1.)\ntorch.autograd.grad(y, y)', 'RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn'],
    ['x = torch.tensor(1., requires_grad=True)\nu = torch.tensor(1., requires_grad=True)\ntorch.autograd.grad(x * 2, u)', 'RuntimeError: The differentiated Tensor at index 0 appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.'],
  ])('%s', (src, want) => expect(msg(src)).toBe(want))
  it('autograd.grad with allow_unused', () => {
    expect(out('x = torch.tensor(1., requires_grad=True)\nu = torch.tensor(1., requires_grad=True)\nprint(torch.autograd.grad(x * 2, [x, u], allow_unused=True))')).toBe('(tensor(2.), None)\n')
  })
})

describe('adjoint identity', () => {
  it('ConvTranspose2d is the adjoint of Conv2d: ⟨conv(x), y⟩ = ⟨x, conv_t(y)⟩', () => {
    const got = out(`torch.manual_seed(0)
w = torch.rand(3, 2, 3, 3, dtype=torch.float64)
x = torch.rand(1, 2, 7, 7, dtype=torch.float64)
y = F.conv2d(x, w, stride=2, padding=1)
z = torch.rand_like(y)
left = (y * z).sum().item()
right = (x * F.conv_transpose2d(z, w, stride=2, padding=1)).sum().item()
print(abs(left - right) < 1e-9)`)
    expect(got).toBe('True\n')
  })
})
