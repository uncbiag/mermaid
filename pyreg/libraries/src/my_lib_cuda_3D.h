// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBCWHD_updateOutput_cuda_3D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int *);

int BilinearSamplerBCWHD_updateGradInput_cuda_3D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int *);

int BilinearSamplerBCWHD_updateGradInputOnlyGrid_cuda_3D(THCudaTensor *inputImages, THCudaTensor *grids,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int *);
