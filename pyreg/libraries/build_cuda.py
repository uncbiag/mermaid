from __future__ import print_function
import os
import sys
import torch
from torch.utils.ffi import create_extension

if sys.version_info >= (3, 0):
    target_dir = '_ext'
else:
    target_dir = '_ext'

#this_file = os.path.dirname(__file__)
sources_1D = []
headers_1D = []
extra_objects_1D=[]

sources_2D = []
headers_2D = []
extra_objects_2D=[]
sources_3D = []
headers_3D = []
extra_objects_3D =[]
sources_nn = []
headers_nn = []
extra_objects_nn =[]
defines = []

with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources_1D += ['src/my_lib_cuda_1D.c']
    headers_1D += ['src/my_lib_cuda_1D.h']
    sources_2D += ['src/my_lib_cuda_2D.c']
    headers_2D += ['src/my_lib_cuda_2D.h']
    sources_3D += ['src/my_lib_cuda_3D.c']
    headers_3D += ['src/my_lib_cuda_3D.h']
    sources_nn += ['src/nn_interpolation.c']
    headers_nn += ['src/nn_interpolation.h']

    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects_1D += ['src/my_lib_cuda_1D.cu.o']
extra_objects_2D += ['src/my_lib_cuda_2D.cu.o']
extra_objects_3D += ['src/my_lib_cuda_3D.cu.o']
extra_objects_nn += ['src/nn_interpolation.cu.o']
extra_objects_1D = [os.path.join(this_file, fname) for fname in extra_objects_1D]
extra_objects_2D = [os.path.join(this_file, fname) for fname in extra_objects_2D]
extra_objects_3D = [os.path.join(this_file, fname) for fname in extra_objects_3D]
extra_objects_nn = [os.path.join(this_file, fname) for fname in extra_objects_nn]

ffi_1D = create_extension(
    target_dir + '.my_lib_1D',
    headers=headers_1D,
    sources=sources_1D,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects_1D,
extra_compile_args=["-std=c99"]
)


ffi_2D = create_extension(
    target_dir + '.my_lib_2D',
    headers=headers_2D,
    sources=sources_2D,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects_2D,
extra_compile_args=["-std=c99"]
)


ffi_3D = create_extension(
    target_dir + '.my_lib_3D',
    headers=headers_3D,
    sources=sources_3D,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects_3D,
extra_compile_args=["-std=c99"]
)

ffi_nn = create_extension(
    target_dir + '.nn_interpolation',
    headers=headers_nn,
    sources=sources_nn,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects_nn,
extra_compile_args=["-std=c99"]
)

if __name__ == '__main__':
    ffi_1D.build()
    ffi_2D.build()
    ffi_3D.build()
    ffi_nn.build()
 
