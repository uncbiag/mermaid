int BilinearSamplerBXC_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBXC_updateGradInput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput);

int BilinearSamplerBXYC_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBXYC_updateGradInput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput);

int BilinearSamplerBXYZC_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBXYZC_updateGradInput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput);

int BilinearSamplerBCX_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBCX_updateGradInput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput);

int BilinearSamplerBCXY_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBCXY_updateGradInput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput);

int BilinearSamplerBCXYZ_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBCXYZ_updateGradInput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput);

// the dimension-generic interfaces

int BilinearSamplerBXYZC_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim);

int BilinearSamplerBXYZC_updateGradInput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int ndim);

int BilinearSamplerBCXYZ_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim);

int BilinearSamplerBCXYZ_updateGradInput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, int ndim);

// here is the original code from pytorch.stn

int BilinearSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput);





