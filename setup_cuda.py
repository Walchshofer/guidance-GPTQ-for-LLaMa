from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', ['quant_cuda.cpp', 'quant_cuda_kernel.cu'],
        extra_compile_args={
            'cxx': ['-std=c++17'],
            'nvcc': ['-O3', '-arch=compute_86', '-code=sm_86', '-std=c++17']
        }
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
