#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "my_lib_cuda_kernel_3D.h"

#define real float

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids



int BilinearSamplerBCWHD_updateOutput_cuda_3D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int * device)
{
// not right here, just for projection, refer to mycuda
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
// /*output->size[2]*/int szw,
//                                                  /*output->size[1]*/int szc,
//                                                  /*output->size[0]*/int szb,
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
  success = BilinearSamplerBCWHD_updateOutput_cuda_kernel_3D(output->size[2],
                                               output->size[1],
                                               output->size[0],
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



int BilinearSamplerBCWHD_updateGradInput_cuda_3D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int * device)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  // /*gradOutput->size[2]*/int szw, 
  //                                                   /*gradOutput->size[1]*/int szc,
  //                                                   /*gradOutput->size[0]*/int szb,
  //                                                   /*THCudaTensor_size(state, inputImages, 1)*/int ic,
  //                                                   /*THCudaTensor_size(state, inputImages, 2)*/int iw,
  //                                                   /*THCudaTensor_size(state, inputImages, 3)*/int ih,
  //                                                   /*THCudaTensor_size(state, inputImages, 4)*/int id,
  //                                                   /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, int isd, 
  //                                                   /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh,  int gsd,
  //                                                   /*THCudaTensor *gradInputImages*/float *gradInputImages, int gisb, int gisc, int gisw, int gish, int gisd,
  //                                                   /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsw, int ggsh, int ggsd,
  //                                                   /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosw, int gosh, int gosd,
  //                                                   /*THCState_getCurrentStream(state)*/cudaStream_t stream);

  cudaSetDevice(device[0]);
  int success = 0;
  success = BilinearSamplerBCWHD_updateGradInput_cuda_kernel_3D(gradOutput->size[2],
                                                  gradOutput->size[1],
                                                  gradOutput->size[0],
                                                  THCudaTensor_size(state, inputImages, 1),
                                                  THCudaTensor_size(state, inputImages, 2),
                                                  THCudaTensor_size(state, inputImages, 3),
                                                  THCudaTensor_size(state, inputImages, 4),
                                                  THCudaTensor_size(state, gradOutput, 2),
                                                  THCudaTensor_size(state, gradOutput, 3),
                                                  THCudaTensor_size(state, gradOutput, 4),
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
                                                  THCudaTensor_data(state, gradInputImages),
                                                  THCudaTensor_stride(state, gradInputImages, 0),
                                                  THCudaTensor_stride(state, gradInputImages, 1),
                                                  THCudaTensor_stride(state, gradInputImages, 2),
                                                  THCudaTensor_stride(state, gradInputImages, 3),
                                                  THCudaTensor_stride(state, gradInputImages, 4),
                                                  THCudaTensor_data(state, gradGrids),
                                                  THCudaTensor_stride(state, gradGrids, 0),
                                                  THCudaTensor_stride(state, gradGrids, 1),
                                                  THCudaTensor_stride(state, gradGrids, 2),
                                                  THCudaTensor_stride(state, gradGrids, 3),
                                                  THCudaTensor_stride(state, gradGrids, 4),
                                                  THCudaTensor_data(state, gradOutput),
                                                  THCudaTensor_stride(state, gradOutput, 0),
                                                  THCudaTensor_stride(state, gradOutput, 1),
                                                  THCudaTensor_stride(state, gradOutput, 2),
                                                  THCudaTensor_stride(state, gradOutput, 3),
                                                  THCudaTensor_stride(state, gradOutput, 4),
                                                  THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

int BilinearSamplerBCWHD_updateGradInputOnlyGrid_cuda_3D(THCudaTensor *inputImages, THCudaTensor *grids,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int * device)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  cudaSetDevice(device[0]);
  int success = 0;
  success = BilinearSamplerBCWHD_updateGradInputOnlyGrid_cuda_kernel_3D(
                                                  gradOutput->size[2],
                                                  gradOutput->size[1],
                                                  gradOutput->size[0],
                                                  THCudaTensor_size(state, inputImages, 1),
                                                  THCudaTensor_size(state, inputImages, 2),
                                                  THCudaTensor_size(state, inputImages, 3),
                                                  THCudaTensor_size(state, inputImages, 4),
                                                  THCudaTensor_size(state, gradOutput, 2),
                                                  THCudaTensor_size(state, gradOutput, 3),
                                                  THCudaTensor_size(state, gradOutput, 4),
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
                                                  THCudaTensor_data(state, gradGrids),
                                                  THCudaTensor_stride(state, gradGrids, 0),
                                                  THCudaTensor_stride(state, gradGrids, 1),
                                                  THCudaTensor_stride(state, gradGrids, 2),
                                                  THCudaTensor_stride(state, gradGrids, 3),
                                                  THCudaTensor_stride(state, gradGrids, 4),
                                                  THCudaTensor_data(state, gradOutput),
                                                  THCudaTensor_stride(state, gradOutput, 0),
                                                  THCudaTensor_stride(state, gradOutput, 1),
                                                  THCudaTensor_stride(state, gradOutput, 2),
                                                  THCudaTensor_stride(state, gradOutput, 3),
                                                  THCudaTensor_stride(state, gradOutput, 4),
                                                  THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}




