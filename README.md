# CUDA implementation of Wavelet KAN

## Note

- There are no optimizations in this implementation. I a cuda beginner and willing to receive optimization suggestions : )

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
1. Morlet wavelet performs badly in MNIST, but if you use shallow net, you can observe it learn.

