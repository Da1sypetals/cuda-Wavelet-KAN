#include <torch/torch.h>
#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
#define INDEX3D(a, b, c, db, dc) (((a) * (db) * (dc) + (b) * (dc) + (c)))

#define COEFF 0.867325070f

#define MAX_DIM 2048

#define ALL_THREADS_IN_WARP 0xFFFFFFFF

__global__ void fwd_kernel(const torch::PackedTensorAccessor64<float, 2> x,
                           const torch::PackedTensorAccessor64<float, 2> scale,
                           const torch::PackedTensorAccessor64<float, 2> bias,
                           const torch::PackedTensorAccessor64<float, 2> weight,
                           torch::PackedTensorAccessor64<float, 3> result,
                           int batch_size, int in_feats, int out_feats, int numThreads)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // x.size(): (batch_size, in_feats)

    if (idx < numThreads)
    {
        int ibatch = idx / (out_feats * in_feats);
        int iout = (idx / in_feats) % out_feats;
        int iin = idx % in_feats;

        /* optimization: should not access memory one time for each thread */
        float x_val = x[ibatch][iin];
        float s = scale[iout][iin];
        float b = bias[iout][iin];
        float w = weight[iout][iin];

        float y2 = s * s * (x_val + b) * (x_val + b);
        float u = COEFF * w * (y2 - 1) * expf(-0.5f * y2);

        result[ibatch][iout][iin] = u;
    }
}

__global__ void bwd_kernel(const torch::PackedTensorAccessor64<float, 2> gout,
                           const torch::PackedTensorAccessor64<float, 2> x,
                           const torch::PackedTensorAccessor64<float, 2> scale,
                           const torch::PackedTensorAccessor64<float, 2> bias,
                           const torch::PackedTensorAccessor64<float, 2> weight,
                           torch::PackedTensorAccessor64<float, 3> grad_x,
                           torch::PackedTensorAccessor64<float, 3> grad_s_expand,
                           torch::PackedTensorAccessor64<float, 3> grad_b_expand,
                           torch::PackedTensorAccessor64<float, 3> grad_w_expand,
                           int batch_size, int in_feats,
                           int out_feats, int numThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numThreads)
    {
        int ibatch = idx / (out_feats * in_feats);
        int iout = (idx / in_feats) % out_feats;
        int iin = idx % in_feats;

        float gout_val = gout[ibatch][iout];
        float x_val = x[ibatch][iin];
        float s = scale[iout][iin];
        float b = bias[iout][iin];
        float w = weight[iout][iin];

        float y_val = s * (x_val + b);

        float exp_val = expf(-0.5f * y_val * y_val);
        float share = COEFF * w * exp_val * (3.0f - y_val * y_val) * y_val;

        float g_x = share * s;
        float g_s = share * x_val;
        // float g_b = g_x;
        float g_w = COEFF * exp_val * (y_val * y_val - 1);

        grad_x[ibatch][iout][iin] = gout_val * g_x;
        grad_s_expand[ibatch][iout][iin] = gout_val * g_s;
        grad_b_expand[ibatch][iout][iin] = gout_val * g_x;
        grad_w_expand[ibatch][iout][iin] = gout_val * g_w;
    }
}

void fwd_launcher(const torch::PackedTensorAccessor64<float, 2> x,
                  const torch::PackedTensorAccessor64<float, 2> scale,
                  const torch::PackedTensorAccessor64<float, 2> bias,
                  const torch::PackedTensorAccessor64<float, 2> weight,
                  torch::PackedTensorAccessor64<float, 3> result,
                  int batch_size, int in_feats, int out_feats)
{
    int numThreads = batch_size * in_feats * out_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    fwd_kernel<<<blockSize, threadSize>>>(x, scale, bias, weight, result,
                                          batch_size, in_feats, out_feats, numThreads);
}

void bwd_launcher(const torch::PackedTensorAccessor64<float, 2> gout,
                  const torch::PackedTensorAccessor64<float, 2> x,
                  const torch::PackedTensorAccessor64<float, 2> scale,
                  const torch::PackedTensorAccessor64<float, 2> bias,
                  const torch::PackedTensorAccessor64<float, 2> weight,
                  torch::PackedTensorAccessor64<float, 3> grad_x,
                  torch::PackedTensorAccessor64<float, 3> grad_s_expand,
                  torch::PackedTensorAccessor64<float, 3> grad_b_expand,
                  torch::PackedTensorAccessor64<float, 3> grad_w_expand,
                  int batch_size, int in_feats,
                  int out_feats)
{

    int numThreads = batch_size * in_feats * out_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    bwd_kernel<<<blockSize, threadSize>>>(gout, x, scale, bias, weight,
                                          grad_x, grad_s_expand, grad_b_expand, grad_w_expand,
                                          batch_size, in_feats, out_feats, numThreads);
}
