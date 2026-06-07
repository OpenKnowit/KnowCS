// 极简反向模式自动求导引擎（micrograd 风格），用于 PyTorch autograd 可视化。
// 每个 Value 记录前驱与局部反传规则，backward() 按拓扑序累加梯度。

export class Value {
  data: number
  grad = 0
  label: string
  op: string
  prev: Value[]
  private _backward: () => void = () => {}

  constructor(data: number, label = '', op = '', prev: Value[] = []) {
    this.data = data
    this.label = label
    this.op = op
    this.prev = prev
  }

  add(other: Value, label = ''): Value {
    const out = new Value(this.data + other.data, label, '+', [this, other])
    out._backward = () => {
      this.grad += out.grad
      other.grad += out.grad
    }
    return out
  }

  mul(other: Value, label = ''): Value {
    const out = new Value(this.data * other.data, label, '×', [this, other])
    out._backward = () => {
      this.grad += other.data * out.grad
      other.grad += this.data * out.grad
    }
    return out
  }

  relu(label = ''): Value {
    const out = new Value(Math.max(0, this.data), label, 'ReLU', [this])
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad
    }
    return out
  }

  tanh(label = ''): Value {
    const t = Math.tanh(this.data)
    const out = new Value(t, label, 'tanh', [this])
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad
    }
    return out
  }

  /** 反向传播：拓扑排序后从输出回传梯度（输出节点 grad=1）。 */
  backward(): void {
    const topo: Value[] = []
    const visited = new Set<Value>()
    const build = (v: Value) => {
      if (visited.has(v)) return
      visited.add(v)
      v.prev.forEach(build)
      topo.push(v)
    }
    build(this)
    topo.forEach((v) => (v.grad = 0))
    this.grad = 1
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward()
  }
}

/** 数值梯度（中心差分），用于校验解析梯度。 */
export function numericalGrad(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - f(x - h)) / (2 * h)
}
