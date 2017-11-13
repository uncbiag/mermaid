from __future__ import print_function
import os
import torch
from torch.utils.ffi import create_extension

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
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects_1D += ['src/my_lib_cuda_1D.cu.o']
extra_objects_2D += ['src/my_lib_cuda_2D.cu.o']
extra_objects_3D += ['src/my_lib_cuda_3D.cu.o']
extra_objects_1D = [os.path.join(this_file, fname) for fname in extra_objects_1D]
extra_objects_2D = [os.path.join(this_file, fname) for fname in extra_objects_2D]
extra_objects_3D = [os.path.join(this_file, fname) for fname in extra_objects_3D]

ffi_1D = create_extension(
    '_ext.my_lib_1D',
    headers=headers_1D,
    sources=sources_1D,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects_1D
)


ffi_2D = create_extension(
    '_ext.my_lib_2D',
    headers=headers_2D,
    sources=sources_2D,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects_2D
)


ffi_3D = create_extension(
    '_ext.my_lib_3D',
    headers=headers_3D,
    sources=sources_3D,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects_3D
)


if __name__ == '__main__':
    ffi_1D.build()
    ffi_2D.build()
    ffi_3D.build()
 
