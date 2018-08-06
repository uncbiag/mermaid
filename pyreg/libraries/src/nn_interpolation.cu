#include <stdbool.h>
#include <stdio.h>
#include "nn_interpolation_kernel.h"


const int bdx =16;
const int bdy = 16;
//const int bdz =4;
#define real float


__device__ void getRound(float x, int width, int& point)
{
   float xcoord = (x + 1) * (width - 1) / 2;
   point = round(xcoord);
}

__device__ bool between(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}



__global__ void nearestNeighFromGrid_1D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideChannels, int grids_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideWidth,
                                         int inputImages_channels, int inputImages_width, int output_width)

{
    const int wOut = blockIdx.x*blockDim.x+threadIdx.x;
   const bool withinImageBounds = wOut < output_width; // asume the size of input is the same as the output

   const int b = blockIdx.y;
   float xf=0;
   if(withinImageBounds){
      int grid_address = b*grids_strideBatch + wOut*grids_strideWidth; // here we use the address of the 0th channel
      xf = grids_data[grid_address];
   }
   else
      return;
   int xInRound;
   getRound(xf, inputImages_width, xInRound);

   const int outAddress = output_strideBatch * b  + output_strideWidth * wOut;
   const int inRoundAddress = inputImages_strideBatch * b  + inputImages_strideWidth * xInRound;


   float v=0;
   float inRound=0;

   bool RoundIsIn = between(xInRound, 0, inputImages_width-1);


   // interpolation happens here
   for(int t=0; t<inputImages_channels; t++)
   {
      if(RoundIsIn) inRound = inputImages_data[inRoundAddress + t*inputImages_strideChannels];
      v = inRound;
      output_data[outAddress + t*output_strideChannels] = v;
    }

  }



__global__ void nearestNeighFromGrid_2D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideHeight, int output_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int output_height, int output_width)

{
    const int wOut = blockIdx.y*blockDim.y+threadIdx.y;
   const int hOut = blockIdx.x*blockDim.x+threadIdx.x;
   const bool withinImageBounds = wOut < output_width && hOut < output_height; // asume the size of input is the same as the output

   const int b = blockIdx.z;
   float yf=0;
   float xf=0;
   if(withinImageBounds){
      int grid_address =b*grids_strideBatch + hOut*grids_strideHeight + wOut*grids_strideWidth; // here we use the address of the 0th channel
      xf = grids_data[grid_address];
      yf = grids_data[grid_address + grids_strideYX];  // address of the 1st channel
   }
   else
      return;
   int yInRound, xInRound;
   getRound(xf, inputImages_width, xInRound);
   getRound(yf, inputImages_height, yInRound);

   const int outAddress = output_strideBatch * b + output_strideHeight * hOut + output_strideWidth * wOut;
   const int inRoundAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInRound + inputImages_strideWidth * xInRound;

   float v=0;
   float inRound=0;


   bool RoundIsIn = between(xInRound, 0, inputImages_width-1) && between(yInRound, 0, inputImages_height-1);

   // interpolation happens here
   for(int t=0; t<inputImages_channels; t++)
   {
      if(RoundIsIn) inRound = inputImages_data[inRoundAddress + t*inputImages_strideChannels];


      v = inRound;
       output_data[outAddress + t*output_strideChannels] = v;
    }
  }

__global__ void nearestNeighFromGrid_3D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, 
                                          int inputImages_strideDepth, int inputImages_strideHeight, int inputImages_strideWidth,
                                          float* grids_data, int grids_strideBatch, int grids_strideChannels, 
                                          int grids_strideDepth, int grids_strideHeight, int grids_strideWidth,
                                          float* output_data, int output_strideBatch, int output_strideChannels,
                                          int output_strideDepth, int output_strideHeight, int output_strideWidth,
                                          int inputImages_channels, int inputImages_depth, int inputImages_height, int inputImages_width,
                                          int outputImages_depth, int outputImages_height, int outputImages_width)
  
{
   const int wOut = blockIdx.z % (outputImages_width);
   const int hOut = blockIdx.y*blockDim.y + threadIdx.y;
   const int dOut = blockIdx.x*blockDim.x + threadIdx.x;

   const bool withinImageBounds = dOut < outputImages_depth && hOut < outputImages_height; // asume the size of input is the same as the output
   
   const int batchIdx = blockIdx.z /(outputImages_width);
   float zf=0;
   float yf=0;
   float xf=0;
   if(withinImageBounds){
      int grid_address = batchIdx*grids_strideBatch +  dOut*grids_strideDepth + hOut*grids_strideHeight + wOut*grids_strideWidth; // here we use the address of the 0th channel
      xf = grids_data[grid_address];  // changed zf yf xz to xf yf zf, to adpat the cpu version, assume in grid the 3*1 vector is stored as width,height and depth
      yf = grids_data[grid_address + grids_strideChannels];
      zf = grids_data[grid_address + grids_strideChannels*2];  // address of the 1st channel
   }
   else
      return;
   int zInRound, yInRound, xInRound;  // zInTopFrontRound
   getRound(xf, inputImages_width, xInRound);
   getRound(yf, inputImages_height, yInRound);
   getRound(zf, inputImages_depth, zInRound);
   
   const int outAddress = output_strideBatch * batchIdx + output_strideDepth * dOut + output_strideHeight * hOut + output_strideWidth * wOut;  // here assume the channel will be calculated later
   const int inRoundAddress = inputImages_strideBatch * batchIdx + inputImages_strideDepth*zInRound+ inputImages_strideHeight * yInRound + inputImages_strideWidth * xInRound;

   float v=0;
   float inRound=0;
 

    bool RoundIsIn = between(xInRound, 0, inputImages_width-1) && between(yInRound, 0, inputImages_height-1)&& between(zInRound, 0, inputImages_depth-1);


   // interpolation happens here
   for(int t=0; t<inputImages_channels; t++)
   {
      if(RoundIsIn)  inRound = inputImages_data[inRoundAddress + t*inputImages_strideChannels];
     

      v =  inRound;

       output_data[outAddress + t*output_strideChannels] = v;
    }
  }




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
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{
  // batch channel x y
  //  0      1     2 3 
   //dim3 blocks((THCudaTensor_size(state,output,2)+15)/16, THCudaTensor_size(state,output,1), THCudaTensor_size(state,output,0));
   dim3 blocks((ow+bdx-1)/bdx, sz3);
   dim3 threads(bdx);

   /* assume BHWD */
   nearestNeighFromGrid_1D <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                    /*THCudaTensor_data(state, inputImages)*/inputImages, 
                                                    /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                    /*THCudaTensor_stride(state, inputImages, 3)*/isc, 
                                                    /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                    /*THCudaTensor_data(state, grids)*/grids, 
                                                    /*THCudaTensor_stride(state, grids, 0)*/gsb, 
                                                    /*THCudaTensor_stride(state, grids, 3)*/gsc, 
                                                    /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                    /*THCudaTensor_data(state, output)*/output,  
                                                    /*THCudaTensor_stride(state, output, 0)*/osb, 
                                                    /*THCudaTensor_stride(state, output, 3)*/osc,
                                                    /*THCudaTensor_stride(state, output, 2)*/osw,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/ic, 
                                                    /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                    /*THCudaTensor_size(state, output, 2)*/ow);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in nearestNeigh.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}




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
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{
  // batch channel x y
  //  0      1     2 3 
   //dim3 blocks((THCudaTensor_size(state,output,2)+15)/16, THCudaTensor_size(state,output,1), THCudaTensor_size(state,output,0));
   dim3 blocks((oh+bdx-1)/bdx, (ow+bdy-1)/bdy, sz3);
   dim3 threads(bdx,bdy);
   //printf(" iw, ih, ow, oh  %d %d %d %d",iw,ih,ow,oh);

   /* assume BHWD */
   nearestNeighFromGrid_2D <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                    /*THCudaTensor_data(state, inputImages)*/inputImages, 
                                                    /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                    /*THCudaTensor_stride(state, inputImages, 3)*/isc, 
                                                    /*THCudaTensor_stride(state, inputImages, 1)*/ish, 
                                                    /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                    /*THCudaTensor_data(state, grids)*/grids, 
                                                    /*THCudaTensor_stride(state, grids, 0)*/gsb, 
                                                    /*THCudaTensor_stride(state, grids, 3)*/gsc,
                                                    /*THCudaTensor_stride(state, grids, 1)*/gsh, 
                                                    /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                    /*THCudaTensor_data(state, output)*/output,  
                                                    /*THCudaTensor_stride(state, output, 0)*/osb, 
                                                    /*THCudaTensor_stride(state, output, 3)*/osc,
                                                    /*THCudaTensor_stride(state, output, 1)*/osh, 
                                                    /*THCudaTensor_stride(state, output, 2)*/osw,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/ic,
                                                    /*THCudaTensor_size(state, inputImages, 1)*/ih, 
                                                    /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                    /*THCudaTensor_size(state, output, 2)*/oh, ow);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in nearestNeigh.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}






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
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{
  // batch channel x y  z
  //  0      1     2 3  4
   //dim3 blocks((THCudaTensor_size(state,output,2)+15)/16, THCudaTensor_size(state,output,1), THCudaTensor_size(state,output,0));
   dim3 blocks((od+bdx-1)/bdx, (oh+bdy-1)/bdy, szw*szb);
   dim3 threads(bdx,bdy);
   //printf(" gsh %d  gsc %d osh %d osc %d  ic %d id %d ih %d\n!!!!",gsh,gsc,osh,osc,ic,id,ih);

   /* assume BHWD */
   //                                       (float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, 
   //                                        int inputImages_strideDepth, int inputImages_strideHeight, int inputImages_strideWidth,
   //                                        float* grids_data, int grids_strideBatch, int grids_strideChannels, 
   //                                        int grids_strideDepth, int grids_strideHeight, int grids_strideWidth,
   //                                        float* output_data, int output_strideBatch, int output_strideChannels,
   //                                        int output_strideDepth, int output_strideHeight, int output_strideWidth,
   //                                        int inputImages_channels, int inputImages_depth, int inputImages_height, int inputImages_width)
   nearestNeighFromGrid_3D <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                    inputImages, isb, isc, isd, ish, isw,
                                                    grids,       gsb, gsc, gsd, gsh, gsw,
                                                    output,      osb, osc, osd, osh, osw,
                                                    ic,  id,  ih, iw, od, oh, ow);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in nearestNeigh.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}

#ifdef __cplusplus
}
#endif
