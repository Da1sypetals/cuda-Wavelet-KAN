import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-12'


from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    'nvcc': ['-g', '--generate-line-info'],  # CUDA编译器的调试选项
    'cxx': ['-g']  # C++编译器的调试选项，对于GCC或Clang，'-g'开启调试信息
}

setup(
    name='WaveletOps',
    packages=find_packages(),
    version='0.0.0',
    author='Yuxue Yang',
    ext_modules=[
        # CUDAExtension(
        #     'mexhat_ops', # operator name
        #     ['./baseline_cpp/wav.cpp',
        #      './baseline_cpp/wav_cuda.cu',],
        #       extra_compile_args=extra_compile_args
        # ),
        # # CUDAExtension(
        # #     'optim_mexhat_ops', # operator name
        # #     ['./cpp/wav.cpp',
        # #      './cpp/wav_cuda.cu',]
        # # ),
        # # CUDAExtension(
        # #     'morlet_ops', # operator name
        # #     ['./morlet/morlet.cpp',
        # #      './morlet/morlet_cuda.cu',]
        # # ),
        # CUDAExtension(
        #     'gemm_ops', # operator name
        #     ['./gemm_cpp/wav.cpp',
        #      './gemm_cpp/wav_cuda.cu',
        #      './gemm_cpp/gemm.cu',
        #      ],
        # ),
        CUDAExtension(
            'mmops', # operator name
            ['./matmul/mm.cpp',
             './matmul/mm_cuda.cu',
             './matmul/mm_autotune.cu',
             './matmul/mm_new.cu',
             ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)