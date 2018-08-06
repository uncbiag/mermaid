// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBCW_updateOutput_cuda_1D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int *, int);

int BilinearSamplerBCW_updateGradInput_cuda_1D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int *,int);

int BilinearSamplerBCW_updateGradInputOnlyGrid_cuda_1D(THCudaTensor *inputImages, THCudaTensor *grids,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int *,int);
