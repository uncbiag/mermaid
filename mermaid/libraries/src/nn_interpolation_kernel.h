#ifdef __cplusplus
extern "C" {
#endif



int nearestNeighBCW_updateOutput_cuda_kernel_1D(/*THCudaTensor_size(state,output,2)*/int szw,
                                                 /*THCudaTensor_size(state,output,1)*/int szc,
                                                 /*THCudaTensor_size(state,output,0)*/int sz3,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                                 /*THCudaTensor_size(state, output, 2)*/int ow,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, 
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, 
                                                 /*THCudaTensor *output*/float *output, int osb, int osc, int osw,
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream);
                                                 
                                                 
int nearestNeighBCWH_updateOutput_cuda_kernel_2D(/*THCudaTensor_size(state,output,2)*/int szw,
                                                 /*THCudaTensor_size(state,output,1)*/int szc,
                                                 /*THCudaTensor_size(state,output,0)*/int sz3,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                                 /*THCudaTensor_size(state, inputImages, 2)*/int ih,
                                                 /*THCudaTensor_size(state, output, 2)*/int ow,
                                                 /*THCudaTensor_size(state, output, 2)*/int oh,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, 
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh, 
                                                 /*THCudaTensor *output*/float *output, int osb, int osc, int osw, int osh, 
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream);


int nearestNeighBCWHD_updateOutput_cuda_kernel_3D(/*THCudaTensor_size(state,output,2)*/int szw,
                                                 /*THCudaTensor_size(state,output,1)*/int szc,
                                                 /*THCudaTensor_size(state,output,0)*/int szb,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 2)*/int iw,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int ih,
                                                 /*THCudaTensor_size(state, inputImages, 4)*/int id,
                                                 /*THCudaTensor_size(state, inputImages, 2)*/int ow,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int oh,
                                                 /*THCudaTensor_size(state, inputImages, 4)*/int od,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, int isd, 
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh, int gsd, 
                                                 /*THCudaTensor *output*/float *output, int osb, int osc, int osw, int osh, int osd, 
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream);


#ifdef __cplusplus
}
#endif
