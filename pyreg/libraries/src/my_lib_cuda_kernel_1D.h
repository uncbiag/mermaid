#ifdef __cplusplus
extern "C" {
#endif



int BilinearSamplerBCW_updateOutput_cuda_kernel_1D(/*output->size[2]*/int szw,
                                                 /*output->size[1]*/int szc,
                                                 /*output->size[0]*/int sz3,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                                 /*THCudaTensor_size(state, output, 2)*/int ow,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, 
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, 
                                                 /*THCudaTensor *output*/float *output, int osb, int osc, int osw,
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream);




int BilinearSamplerBCW_updateGradInput_cuda_kernel_1D(/*gradOutput->size[2]*/int szw, 
                                                    /*gradOutput->size[1]*/int szc,
                                                    /*gradOutput->size[0]*/int sz3,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                    /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                                    /*THCudaTensor_size(state, gradOutput, 2)*/int gow,
                                                    /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw,
                                                    /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw,
                                                    /*THCudaTensor *gradInputImages*/float *gradInputImages, int gisb, int gisc, int gisw,
                                                    /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsw,
                                                    /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosw,
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream);

int BilinearSamplerBCW_updateGradInputOnlyGrid_cuda_kernel_1D(
                                        /*gradOutput->size[2]*/int szw, 
                                        /*gradOutput->size[1]*/int szc,
                                        /*gradOutput->size[0]*/int sz3,
                                        /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                        /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                        /*THCudaTensor_size(state, gradOutput, 2)*/int gow,
                                        /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw,
                                        /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw,
                                        /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsw, 
                                        /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosw, 
                                        /*THCState_getCurrentStream(state)*/cudaStream_t stream);

#ifdef __cplusplus
}
#endif
