import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-12'


from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    name='WaveletOps',
    packages=find_packages(),
    version='0.0.0',
    author='Yuxue Yang',
    ext_modules=[
        CUDAExtension(
            'mexhat_ops', # operator name
            ['./baseline_cpp/wav.cpp',
             './baseline_cpp/wav_cuda.cu',]
        ),
        CUDAExtension(
            'morlet_ops', # operator name
            ['./morlet/morlet.cpp',
             './morlet/morlet_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)