#include <torch/torch.h>
#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
#define INDEX3D(a, b, c, db, dc) (((a) * (db) * (dc) + (b) * (dc) + (c)))

#define COEFF 0.867325070f

#define MAX_DIM 2048

#define ALL_THREADS_IN_WARP 0xFFFFFFFF

__global__ void fwd_kernel(const torch::PackedTensorAccessor64<float, 3> x,
                           const torch::PackedTensorAccessor64<float, 2> scale,
                           const torch::PackedTensorAccessor64<float, 2> bias,
                           const torch::PackedTensorAccessor64<float, 2> weight,
                           torch::PackedTensorAccessor64<float, 3> result,
                           int batch_size, int batch_size_padded, int in_feats, int out_feats, int numThreads)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // batch_size_padded pads batch_size to a multiple of 32
    // x.size(): (out_feats, in_feats, batch_size)
    int ibatch = idx % batch_size_padded;
    int iin = (idx / batch_size_padded) % in_feats;
    int iout = idx / (in_feats * batch_size_padded);

    float s0, b0, w0;
    if (threadIdx.x % 32 == 0) // 1st thread in warp
    {
        s0 = scale[iout][iin];
        b0 = bias[iout][iin];
        w0 = weight[iout][iin];
    }

    __syncwarp();

    float s = __shfl_sync(ALL_THREADS_IN_WARP, s0, 0);
    float b = __shfl_sync(ALL_THREADS_IN_WARP, b0, 0);
    float w = __shfl_sync(ALL_THREADS_IN_WARP, w0, 0);

    if (idx < numThreads)
    {

        float x_val = x[iout][iin][ibatch];

        float z = s * s * (x_val - b) * (x_val - b);
        float u = COEFF * w * (z - 1) * expf(-0.5f * z);

        result[iout][iin][ibatch] = u;
    }
}

__global__ void bwd_kernel()
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

void fwd_launcher(int batch_size, int batch_size_padded)
{
    int numThreads = batch_size_padded * in_feats * out_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    fwd_kernel<<<blockSize, threadSize>>>();
}

void bwd_launcher()
{

    int numThreads =
        dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    bwd_kernel<<<blockSize, threadSize>>>();
}
