# CUDA implementation of Wavelet KAN

- See Also other CUDA implementations of KAN:
   - [Legendre Polynomials](https://github.com/Da1sypetals/Legendre-KAN-cuda)
   - [Chebyshev Polynomials](https://github.com/Da1sypetals/ChebyKan-cuda-op)
   - [RSWAF (variant of RBF)](https://github.com/Da1sypetals/faster-kan-cuda)

- I am interested in the performance aspect of KAN, and willing to discuss / recieve more information on this topic :)
- This is for personal practice purposes, use at your own risk. Tested on my RTX3050 as well as a remote RTX3090 on CUDA 12.x .

## Update:
- A **much** faster implementation is updated, but please note:
   - Since the implementation uses tiling, assertions are opened by default to make sure that tensor conform to the restrictions.
   - If a NaN emerges during training, please first check that all dimensions are divisible by 64. If that does not solve, feel free to open an issue. 
   - Currently only forward is optimized, but optimizing backward is mathematically similar. Maybe it will be done after my examinations...
   - I am a cuda beginner, and I am grateful for any optimization suggestion : )
- Thanks https://github.com/siboehm/SGEMM_CUDA for the optimized `SGEMM` code, I adopted it with some modification and got the implementation.

results on RTX3050:
```
          |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
---------------------------------------------------------------------------------------------------------------------------
cuda-gpu  |    117.24 ms  |    651.21 ms  |      1.10 GB  |      4.12 GB  |     12787840  |              12787840
gemm-gpu  |     26.21 ms  |    678.70 ms  |      0.10 GB  |      4.12 GB  |     12787840  |              12787840
```

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
python benchmark.py --method all --reps 10 --just-cuda
```

---

### Please remind:
1. Morlet wavelet performs badly in MNIST, but if you use a shallow net, you can observe it learn.

