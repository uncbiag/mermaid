#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "nn_interpolation_kernel.h"

#define real float

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids



int nearestNeighBCW_updateOutput_cuda_1D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int * device)
{
// not right here, just for projection, refer to mycuda
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

  cudaSetDevice(device[0]);
  int success = 0;
  success = nearestNeighBCW_updateOutput_cuda_kernel_1D(THCudaTensor_size(state,output,2),
                                               THCudaTensor_size(state,output,1),
                                               THCudaTensor_size(state,output,0),
                                               THCudaTensor_size(state, inputImages, 1),
                                               THCudaTensor_size(state, inputImages, 2),
                                               THCudaTensor_size(state, output, 2),
                                               THCudaTensor_data(state, inputImages),
                                               THCudaTensor_stride(state, inputImages, 0),
                                               THCudaTensor_stride(state, inputImages, 1),
                                               THCudaTensor_stride(state, inputImages, 2),
                                               THCudaTensor_data(state, grids),
                                               THCudaTensor_stride(state, grids, 0),
                                               THCudaTensor_stride(state, grids, 1),
                                               THCudaTensor_stride(state, grids, 2),
                                               THCudaTensor_data(state, output),
                                               THCudaTensor_stride(state, output, 0),
                                               THCudaTensor_stride(state, output, 1),
                                               THCudaTensor_stride(state, output, 2),
                                               THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}




int nearestNeighBCWH_updateOutput_cuda_2D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int * device)
{
// not right here, just for projection, refer to mycuda
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

  cudaSetDevice(device[0]);
  int success = 0;
  success = nearestNeighBCWH_updateOutput_cuda_kernel_2D(THCudaTensor_size(state,output,2),
                                               THCudaTensor_size(state,output,1),
                                               THCudaTensor_size(state,output,0),
                                               THCudaTensor_size(state, inputImages, 1),
                                               THCudaTensor_size(state, inputImages, 2),
                                               THCudaTensor_size(state, inputImages, 3),
                                               THCudaTensor_size(state, output, 2),
                                               THCudaTensor_size(state, output, 3),
                                               THCudaTensor_data(state, inputImages),
                                               THCudaTensor_stride(state, inputImages, 0),
                                               THCudaTensor_stride(state, inputImages, 1),
                                               THCudaTensor_stride(state, inputImages, 2),
                                               THCudaTensor_stride(state, inputImages, 3),
                                               THCudaTensor_data(state, grids),
                                               THCudaTensor_stride(state, grids, 0),
                                               THCudaTensor_stride(state, grids, 1),
                                               THCudaTensor_stride(state, grids, 2),
                                               THCudaTensor_stride(state, grids, 3),
                                               THCudaTensor_data(state, output),
                                               THCudaTensor_stride(state, output, 0),
                                               THCudaTensor_stride(state, output, 1),
                                               THCudaTensor_stride(state, output, 2),
                                               THCudaTensor_stride(state, output, 3),
                                               THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}




int nearestNeighBCWHD_updateOutput_cuda_3D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int * device)
{
// not right here, just for projection, refer to mycuda
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
// /*THCudaTensor_size(state,output,2)*/int szw,
//                                                  /*THCudaTensor_size(state,output,1)*/int szc,
//                                                  /*THCudaTensor_size(state,output,0)*/int szb,
//                                                  /*THCudaTensor_size(state, inputImages, 1)*/int ic,
//                                                  /*THCudaTensor_size(state, inputImages, 2)*/int iw,
//                                                  /*THCudaTensor_size(state, inputImages, 3)*/int ih,
//                                                  THCudaTensor_size(state, inputImages, 4)int id,
//                                                  /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, int isd, 
//                                                  /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh, int gsd, 
//                                                  /*THCudaTensor *output*/float *output, int osb, int osc, int osw, int osh, int osd, 
//                                                  /*THCState_getCurrentStream(state)*/cudaStream_t stream)


  cudaSetDevice(device[0]);
  int success = 0;
  success = nearestNeighBCWHD_updateOutput_cuda_kernel_3D(THCudaTensor_size(state,output,2),
                                               THCudaTensor_size(state,output,1),
                                               THCudaTensor_size(state,output,0),
                                               THCudaTensor_size(state, inputImages, 1),
                                               THCudaTensor_size(state, inputImages, 2),
                                               THCudaTensor_size(state, inputImages, 3),
                                               THCudaTensor_size(state, inputImages, 4),
                                               THCudaTensor_size(state, output, 2),
                                               THCudaTensor_size(state, output, 3),
                                               THCudaTensor_size(state, output, 4),
                                               THCudaTensor_data(state, inputImages),
                                               THCudaTensor_stride(state, inputImages, 0),
                                               THCudaTensor_stride(state, inputImages, 1),
                                               THCudaTensor_stride(state, inputImages, 2),
                                               THCudaTensor_stride(state, inputImages, 3),
                                               THCudaTensor_stride(state, inputImages, 4),
                                               THCudaTensor_data(state, grids),
                                               THCudaTensor_stride(state, grids, 0),
                                               THCudaTensor_stride(state, grids, 1),
                                               THCudaTensor_stride(state, grids, 2),
                                               THCudaTensor_stride(state, grids, 3),
                                               THCudaTensor_stride(state, grids, 4),
                                               THCudaTensor_data(state, output),
                                               THCudaTensor_stride(state, output, 0),
                                               THCudaTensor_stride(state, output, 1),
                                               THCudaTensor_stride(state, output, 2),
                                               THCudaTensor_stride(state, output, 3),
                                               THCudaTensor_stride(state, output, 4),
                                               THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}