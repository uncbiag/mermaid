// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBCWH_updateOutput_cuda_2D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int *);

int BilinearSamplerBCWH_updateGradInput_cuda_2D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int *);

int BilinearSamplerBCWH_updateGradInputOnlyGrid_cuda_2D(THCudaTensor *inputImages, THCudaTensor *grids,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int *);
