# CUDA implementation of Wavelet KAN

## See also

- Other CUDA implementations of KAN:
   - [Legendre Polynomials](https://github.com/Da1sypetals/Legendre-KAN-cuda)
   - [Chebyshev Polynomials](https://github.com/Da1sypetals/ChebyKan-cuda-op)
   - [RSWAF (variant of RBF)](https://github.com/Da1sypetals/faster-kan-cuda)


## Introduction

CUDA implementation of the paper introducing Wavelet KAN at https://arxiv.org/abs/2405.12832.

This is significantly faster than the original implementation, with ~50x performance forward and 5x performance backward, results given by benchmark scripts in https://github.com/Jerry-Master/KAN-benchmarking.

```
          |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
--------------------------------------------------------------------------------------------------------------------
cuda-gpu  |     29.10 ms  |     65.27 ms  |      0.28 GB  |      1.03 GB  |      3151362  |               3151362
orig-gpu  |    522.00 ms  |   1461.29 ms  |      5.53 GB  |      5.53 GB  |      3151362  |               3151362

```


## Note

- There are no optimizations in this implementation. I am a cuda beginner and willing to receive optimization suggestions : )

- Currently Mexican hat and Morlet are implemented.

## Start

1. Install

```bash
pip install -e .
```

> Make sure the version of nvcc in PATH is compatible with your current PyTorch version (it seems minor version difference is OK).

2. Run

   - Run test on MNIST:

   ```bash
   python test.py
   ```

3. Benchmark

```bash
python benchmark.py --method all --reps 100 --just-cuda
```

---

### Please remind:
1. Morlet wavelet performs badly in MNIST, but if you use a shallow net, you can observe it learn.

