#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

'''
python3.5 torch1 with cuda9.0 is ok
python3.5 torch1.1.0 with cuda10.0 is ok
use: python setup.py install --user 
original python:   /usr/bin/python3 setup.py install --user
k36torch14:        /home/luokunming/anaconda3/envs/k36torch14/bin/python3 setup.py install --user

'''
cxx_args = ['-std=c++11']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_61,code=compute_61',
    # '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_75,code=compute_75' # from: https://github.com/visinf/irr/issues/27
    # '-ccbin', '/usr/bin/gcc-4.9'
    '-ccbin', '/usr/bin/gcc'
]

# setup(
#     name='correlation_cuda',
#     ext_modules=[
#         CUDAExtension('correlation_cuda', [
#             'correlation_cuda.cc',
#             'correlation_cuda_kernel.cu'
#         ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args, 'cuda-path': ['/usr/local/cuda-9.0']})
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args, 'cuda-path': ['/data/cuda/cuda-10.0/cuda']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

