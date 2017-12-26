import os
import torch
from torch.utils.ffi import create_extension

#this_file = os.path.dirname(__file__)

sources_nd = ['src/my_lib_nd.c']
headers_nd = ['src/my_lib_nd.h']
sources_nn = ['src/my_lib_nn.c']
headers_nn = ['src/my_lib_nn.h']
defines = []
with_cuda = False

extra_compile_args = []
extra_link_args = []
with_openmp = False # set this to false if you are using clang on OSX or install gcc



if with_openmp:
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
else:
    extra_compile_args += ['-Dno_openmp']
    
this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)


extra_objects = []

ffi_nd = create_extension(
    '_ext.my_lib_nd',
    headers=headers_nd,
    sources=sources_nd,
    verbose=True,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)



ffi_nn = create_extension(
    '_ext.my_lib_nn',
    headers=headers_nn,
    sources=sources_nn,
    verbose=True,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)
if __name__ == '__main__':
    ffi_nd.build()
    ffi_nn.build()
