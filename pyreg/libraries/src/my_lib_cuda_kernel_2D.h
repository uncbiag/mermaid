#ifdef __cplusplus
extern "C" {
#endif



int BilinearSamplerBCWH_updateOutput_cuda_kernel_2D(/*output->size[2]*/int szw,
                                                 /*output->size[1]*/int szc,
                                                 /*output->size[0]*/int sz3,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                                 /*THCudaTensor_size(state, inputImages, 2)*/int ih,
                                                 /*THCudaTensor_size(state, output, 2)*/int ow,
                                                 /*THCudaTensor_size(state, output, 2)*/int oh,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, 
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh, 
                                                 /*THCudaTensor *output*/float *output, int osb, int osc, int osw, int osh, 
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream);




int BilinearSamplerBCWH_updateGradInput_cuda_kernel_2D(/*gradOutput->size[2]*/int szw, 
                                                    /*gradOutput->size[1]*/int szc,
                                                    /*gradOutput->size[0]*/int sz3,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                    /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                                    /*THCudaTensor_size(state, inputImages, 2)*/int ih,
                                                    /*THCudaTensor_size(state, gradOutput, 2)*/int gow,
                                                    /*THCudaTensor_size(state, gradOutput, 2)*/int goh,
                                                    /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, 
                                                    /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh, 
                                                    /*THCudaTensor *gradInputImages*/float *gradInputImages, int gisb, int gisc, int gisw, int gish,
                                                    /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsw, int ggsh,
                                                    /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosw, int gosh,
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream);

int BilinearSamplerBCWH_updateGradInputOnlyGrid_cuda_kernel_2D(
                                        /*gradOutput->size[2]*/int szw, 
                                        /*gradOutput->size[1]*/int szc,
                                        /*gradOutput->size[0]*/int sz3,
                                        /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                        /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                        /*THCudaTensor_size(state, inputImages, 2)*/int ih,
                                        /*THCudaTensor_size(state, gradOutput, 2)*/int gow,
                                        /*THCudaTensor_size(state, gradOutput, 2)*/int goh,
                                        /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, 
                                        /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh, 
                                        /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsw, int ggsh,
                                        /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosw, int gosh,
                                        /*THCState_getCurrentStream(state)*/cudaStream_t stream);

#ifdef __cplusplus
}
#endif
