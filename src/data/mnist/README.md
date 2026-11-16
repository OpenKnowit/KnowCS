# MNIST for the /mnist/ page

The first 30,000 training images and the first 3,000 test images of MNIST (Yann LeCun, Corinna Cortes and
Christopher J. C. Burges; CC BY-SA 3.0), from Keras's copy `mnist.npz`
(sha256 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1).

- `train-0.webp` … `train-9.webp`: 3,000 training images each, `test.webp`: 3,000 test images. Lossless WebP,
  grey, 60 images of 28×28 per row (image `i` of a file at row `i // 60`, column `i % 60`).
- `labels.bin`: 33,000 bytes, the 30,000 training labels then the 3,000 test labels.

The batches load one after another, as in ConvNetJS's demo, so training starts after the first one.
