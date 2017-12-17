
int nearestNeighBCW_updateOutput_cuda_1D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int *);

int nearestNeighBCWH_updateOutput_cuda_2D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int *);

int nearestNeighBCWHD_updateOutput_cuda_3D(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int *);