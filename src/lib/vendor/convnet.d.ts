// The part of ConvNetJS this site uses.
export interface Vol {
  sx: number
  sy: number
  depth: number
  w: Float64Array
  dw: Float64Array
}
export interface Layer {
  layer_type: string
  out_act: Vol
  filters?: Vol[]
  biases?: Vol
  sx?: number
  in_depth?: number
  num_inputs?: number
}
export interface ParamsAndGrads {
  params: Float64Array
  grads: Float64Array
}
export interface Net {
  layers: Layer[]
  makeLayers(defs: Record<string, unknown>[]): void
  forward(v: Vol, isTraining?: boolean): Vol
  /** fills the gradients for class y (the last layer is a softmax) and returns −log p(y) */
  backward(y: number): number
  getParamsAndGrads(): ParamsAndGrads[]
  getPrediction(): number
}
export interface TrainStats {
  loss: number
  cost_loss: number
}
export interface SGDTrainer {
  learning_rate: number
  train(x: Vol, y: number | number[]): TrainStats
}
declare const convnetjs: {
  Net: new () => Net
  Vol: { new (sx: number, sy: number, depth: number, c?: number): Vol; new (values: ArrayLike<number>): Vol }
  SGDTrainer: new (net: Net, options: { learning_rate: number; momentum: number; batch_size: number; l2_decay: number }) => SGDTrainer
}
export default convnetjs
