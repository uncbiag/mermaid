int BilinearSamplerBXC_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary);

int BilinearSamplerBXC_updateGradInput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary);

int BilinearSamplerBXYC_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary);

int BilinearSamplerBXYC_updateGradInput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary);

int BilinearSamplerBXYZC_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary);

int BilinearSamplerBXYZC_updateGradInput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary);

int BilinearSamplerBCX_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary);

int BilinearSamplerBCX_updateGradInput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary);

int BilinearSamplerBCXY_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary);

int BilinearSamplerBCXY_updateOutput_2D_old(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary);

int BilinearSamplerBCXY_updateGradInput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary);

int BilinearSamplerBCXYZ_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary);

int BilinearSamplerBCXYZ_updateGradInput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary);

// the dimension-generic interfaces

int BilinearSamplerBXYZC_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim, int zero_boundary);

int BilinearSamplerBXYZC_updateGradInput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int ndim, int zero_boundary);

int BilinearSamplerBCXYZ_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim, int zero_boundary);

int BilinearSamplerBCXYZ_updateGradInput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int ndim, int zero_boundary);

// here is the original code from pytorch.stn

int BilinearSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary);

int BilinearSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary);





