#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void fwd_launcher();

void bwd_launcher();

torch::Tensor faster_fwd()
{

    // first check inputs!

    // get metadata

    fwd_launcher(//);

    cudaDeviceSynchronize();

    return // something
}

torch::Tensor faster_bwd()
{

    // first check inputs!

    // get metadata

    bwd_launcher(//);

    cudaDeviceSynchronize();

    return // something;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &faster_fwd, "leg forward");
    m.def("backward", &faster_bwd, "leg backward");
}