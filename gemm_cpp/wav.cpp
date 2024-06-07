#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <cuda_runtime.h>
#include <utility>

constexpr float COEFF = 0.867325070f;

#define checkCudaErrors(call)                                       \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess)                                     \
        {                                                           \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void fwd_gemm_launcher(int batch_size, int in_feats, int out_feats,
                       float *x, float *scale, float *bias,
                       float *weight, float *result);

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
                  int out_feats);

using Tensor4 = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

torch::Tensor mexhat_fwd(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, torch::Tensor weight)
{

    // first check inputs!
    CHECK_INPUT(x);
    CHECK_INPUT(scale);
    CHECK_INPUT(bias);
    CHECK_INPUT(weight);

    // get metadata
    // x: (batch_size, in_feats)
    // params: (in_feats, out_feats)
    int batch_size = x.size(0);
    int in_feats = x.size(1);
    int out_feats = scale.size(1);

    // get data accesser
    float *x_ptr = x.data_ptr<float>();
    float *scale_ptr = scale.data_ptr<float>();
    float *bias_ptr = bias.data_ptr<float>();
    float *weight_ptr = weight.data_ptr<float>();

    // create result tensor
    torch::Tensor result = torch::empty({batch_size, out_feats}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    float *result_ptr = result.data_ptr<float>();

    fwd_gemm_launcher(batch_size, in_feats, out_feats, x_ptr, scale_ptr, bias_ptr, weight_ptr, result_ptr);
    cudaDeviceSynchronize();

    return result;
}

Tensor4 mexhat_bwd(torch::Tensor gout, torch::Tensor x, torch::Tensor scale, torch::Tensor bias, torch::Tensor weight)
{

    // first check inputs!
    CHECK_INPUT(gout);
    CHECK_INPUT(x);
    CHECK_INPUT(scale);
    CHECK_INPUT(bias);
    CHECK_INPUT(weight);

    // get metadata
    int batch_size = x.size(0);
    int in_feats = x.size(1);
    int out_feats = scale.size(1);

    // get data accesser
    const auto gout_acc = gout.packed_accessor64<float, 2>();
    const auto x_acc = x.packed_accessor64<float, 2>();
    const auto scale_acc = scale.packed_accessor64<float, 2>();
    const auto bias_acc = bias.packed_accessor64<float, 2>();
    const auto weight_acc = weight.packed_accessor64<float, 2>();

    std::vector<int64_t> shape = {batch_size, in_feats, out_feats};

    torch::Tensor grad_x = torch::empty(shape, torch::device(torch::kCUDA).dtype(torch::kFloat));
    torch::Tensor grad_s_expand = torch::empty(shape, torch::device(torch::kCUDA).dtype(torch::kFloat));
    torch::Tensor grad_b_expand = torch::empty(shape, torch::device(torch::kCUDA).dtype(torch::kFloat));
    torch::Tensor grad_w_expand = torch::empty(shape, torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto grad_x_acc = grad_x.packed_accessor64<float, 3>();
    auto grad_s_expand_acc = grad_s_expand.packed_accessor64<float, 3>();
    auto grad_b_expand_acc = grad_b_expand.packed_accessor64<float, 3>();
    auto grad_w_expand_acc = grad_w_expand.packed_accessor64<float, 3>();

    bwd_launcher(gout_acc, x_acc, scale_acc, bias_acc, weight_acc,
                 grad_x_acc, grad_s_expand_acc, grad_b_expand_acc, grad_w_expand_acc,
                 batch_size, in_feats, out_feats);

    // cudaDeviceSynchronize();

    return Tensor4(
        grad_x.sum({2}),
        grad_s_expand.sum({0}),
        grad_b_expand.sum({0}),
        grad_w_expand.sum({0}));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &mexhat_fwd, "mexican hat forward");
    m.def("backward", &mexhat_bwd, "mexican hat backward");
}