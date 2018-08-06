#include <stdbool.h>
#include <stdio.h>
#include "my_lib_cuda_kernel_1D.h"


const int bdx =256;
//const int bdz =4;
#define real float

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

__device__ void getLeft(float x, int width, int& point, float& weight)
{
   /* for interpolation :
      stores in point and weight :
      - the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
      - the weight for interpolating
   */

   float xcoord = (x + 1) * (width - 1) / 2;
   point = floor(xcoord);
   weight = 1 - (xcoord - point);
}

__device__ bool between_1D(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}



__global__ void bilinearSamplingFromGrid_1D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideChannels, int grids_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideWidth,
                                         int inputImages_channels, int inputImages_width, int output_width, int zero_boundary)
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   /////////////////////////////// threadIdx.x : used for features (coalescing is trivial)
   
   // dim3 blocks((sz1+15)/16, sz2, sz3);
   // dim3 threads(32,16);
   ///*output->size[2]*/int sz1
   /*output->size[1] int sz2,
   output->size[0]int sz3,*/
   //dim3 blocks((sz1+15)/16, (sz2+15)/16, sz3);  x: w/8, y: h/16, z: batch
   //dim3 threads(4,8,16);  x: c 4, y w:8, z h 16
   // threadIdx.x: only 2 of the blockDim.x is used in the grids part, but all of the threadIdx.x can be used in the sampling parts, so it is used in two parts for different propose
   {
  // block(h,w, b)
  //threads(h,w);
  // batch channel  x  y
  //  0      1      2  3 

    const int wOut = blockIdx.x*blockDim.x+threadIdx.x;
   //const int idInBlock = threadIdx.x + threadIdx.y*blockDim.x;
   const bool withinImageBounds = wOut < output_width; // asume the size of input is the same as the output
   //const bool withinGridBounds = blockIdx.x*blockDim.y + idInBlock < output_width;//.........................

   const int b = blockIdx.y;
   float xf=0;
   if(withinImageBounds){
      int grid_address = b*grids_strideBatch + wOut*grids_strideWidth; // here we use the address of the 0th channel
      xf = grids_data[grid_address];
   }
   else
      return;
   int xInLeft;
   float xWeightLeft;
   getLeft(xf, inputImages_width, xInLeft, xWeightLeft);
   bool zero_boundary_bool = zero_boundary == 1;


   bool xBeyondLow = xInLeft < 0;
   bool xBeyondHigh = xInLeft+1 > inputImages_width-1;

    ///////////////  using  non zero border condition

    if (zero_boundary_bool) {
        if (xBeyondLow)
            xInLeft = 0;
        if (xBeyondHigh)
            xInLeft = inputImages_width-2;
    }

   
   const int outAddress = output_strideBatch * b  + output_strideWidth * wOut;
   const int inLeftAddress = inputImages_strideBatch * b  + inputImages_strideWidth * xInLeft;
   const int inRightAddress = inLeftAddress + inputImages_strideWidth;

   float v=0;
   float inLeft=0;
   float inRight=0;



   // interpolation happens here
   for(int t=0; t<inputImages_channels; t++)
   {
      if (zero_boundary_bool || (! (xBeyondLow || xBeyondHigh))){

          inLeft = inputImages_data[inLeftAddress + t*inputImages_strideChannels];
          inRight = inputImages_data[inRightAddress + t*inputImages_strideChannels];
      }


      v = xWeightLeft * inLeft + (1 - xWeightLeft) * inRight;
      output_data[outAddress + t*output_strideChannels] = v;
    }

  }




template<bool onlyGrid> __global__ void backwardBilinearSampling_1D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideWidth,
                                         float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels, int gradInputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideChannels, int grids_strideWidth,
                                         float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideChannels, int gradGrids_strideWidth,
                                         float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, int gradOutput_strideWidth,
                                         int inputImages_channels, int inputImages_width, int output_width, int zero_boundary)
{
   const int wOut = blockIdx.x*blockDim.x+threadIdx.x;
   //const int idInBlock = threadIdx.x + threadIdx.y*blockDim.x;
   const bool withinImageBounds = wOut < output_width;
   //const bool withinGridBounds = blockIdx.x*blockDim.y + idInBlock < output_width;//.........................

   const int b = blockIdx.y;
   float xf;
   int grid_address =b*grids_strideBatch + wOut*grids_strideWidth; // here we use the address of the 0th channel
   float gradxf=0;

   if(withinImageBounds)
   {
      xf = grids_data[grid_address];
      
      int xInLeft;
      float xWeightLeft;
      getLeft(xf, inputImages_width, xInLeft, xWeightLeft);
        bool zero_boundary_bool = zero_boundary == 1;


      bool xBeyondLow = xInLeft < 0;
       bool xBeyondHigh = xInLeft+1 > inputImages_width-1;

        ///////////////  using  non zero border condition

        if (zero_boundary_bool) {
            if (xBeyondLow)
                xInLeft = 0;
            if (xBeyondHigh)
                xInLeft = inputImages_width-2;
        }

      
      const int inLeftAddress = inputImages_strideBatch * b  + inputImages_strideWidth * xInLeft;
      const int inRightAddress = inLeftAddress + inputImages_strideWidth;


      const int gradInputImagesLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideWidth * xInLeft;
      const int gradInputImagesRightAddress = gradInputImagesLeftAddress + gradInputImages_strideWidth;


      const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideWidth * wOut;

      float LeftDotProduct = 0;
      float RightDotProduct = 0;




      for(int t=0; t<inputImages_channels; t++)
      {
        int tch = t*gradInputImages_strideChannels;
         float gradOutValue = gradOutput_data[gradOutputAddress + t*gradOutput_strideChannels];
         // bool between_1D(int value, int lowerBound, int upperBound)
        if (zero_boundary_bool || (! (xBeyondLow || xBeyondHigh))){
            float inLeft = inputImages_data[inLeftAddress + tch];
            LeftDotProduct += inLeft * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesLeftAddress + tch], xWeightLeft * gradOutValue);

            float inRight = inputImages_data[inRightAddress + tch];
            RightDotProduct += inRight * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesRightAddress + tch], (1 - xWeightLeft)  * gradOutValue);
         }

      }
      gradxf = - LeftDotProduct + RightDotProduct;
      gradGrids_data[grid_address] = gradxf * (inputImages_width-1) / 2;
    }
     
}



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
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
  // batch channel x y
  //  0      1     2 3 
   //dim3 blocks((output->size[2]+15)/16, output->size[1], output->size[0]);
   dim3 blocks((ow+bdx-1)/bdx, sz3);
   dim3 threads(bdx);

   /* assume BHWD */
   bilinearSamplingFromGrid_1D <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
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
                                                    /*THCudaTensor_size(state, output, 2)*/ow,
                                                    zero_boundary);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}







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
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

   //dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 blocks((gow+bdx-1)/bdx,sz3);
   dim3 threads(bdx);
   //int grids_channels=2;
   //printf("ggsc %d\n szc %d\n, sz3 %d",ggsc, szc,sz3);

   backwardBilinearSampling_1D <false> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      /*THCudaTensor_data(state, inputImages)*/inputImages, 
                                                      /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                      /*THCudaTensor_stride(state, inputImages, 3)*/isc,
                                                      /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                      /*THCudaTensor_data(state, gradInputImages)*/gradInputImages, 
                                                      /*THCudaTensor_stride(state, gradInputImages, 0)*/gisb,
                                                      /*THCudaTensor_stride(state, gradInputImages, 3)*/gisc,
                                                      /*THCudaTensor_stride(state, gradInputImages, 2)*/gisw,
                                                      /*THCudaTensor_data(state, grids)*/grids, 
                                                      /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                      /*THCudaTensor_stride(state, grids, 3)*/gsc,
                                                      /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                      /*THCudaTensor_data(state, gradGrids)*/gradGrids, 
                                                      /*THCudaTensor_stride(state, gradGrids, 0)*/ggsb,
                                                      /*THCudaTensor_stride(state, gradGrids, 3)*/ggsc,
                                                      /*THCudaTensor_stride(state, gradGrids, 2)*/ggsw,
                                                      /*THCudaTensor_data(state, gradOutput)*/gradOutput, 
                                                      /*THCudaTensor_stride(state, gradOutput, 0)*/gosb,
                                                      /*THCudaTensor_stride(state, gradOutput, 3)*/gosc,
                                                      /*THCudaTensor_stride(state, gradOutput, 2)*/gosw,
                                                      /*THCudaTensor_size(state, inputImages, 3)*/ic,
                                                      /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                      /*THCudaTensor_size(state, gradOutput, 2)*/gow,
                                                      zero_boundary);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}

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
                                        /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

   //dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 blocks((gow+bdx-1)/bdx, szc, sz3);
   dim3 threads(bdx);
   //int grids_channels=2;
   //printf("ow %d gsh %d  gsc %d osh %d osc %d\n",ow,gsh,gsc,osh,osc);

   backwardBilinearSampling_1D <true> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      /*THCudaTensor_data(state, inputImages)*/inputImages, 
                                                      /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                      /*THCudaTensor_stride(state, inputImages, 3)*/isc,
                                                      /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      /*THCudaTensor_data(state, grids)*/grids, 
                                                      /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                      /*THCudaTensor_stride(state, grids, 3)*/gsc,
                                                      /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                      /*THCudaTensor_data(state, gradGrids)*/gradGrids, 
                                                      /*THCudaTensor_stride(state, gradGrids, 0)*/ggsb,
                                                      /*THCudaTensor_stride(state, gradGrids, 3)*/ggsc,
                                                      /*THCudaTensor_stride(state, gradGrids, 2)*/ggsw,
                                                      /*THCudaTensor_data(state, gradOutput)*/gradOutput, 
                                                      /*THCudaTensor_stride(state, gradOutput, 0)*/gosb,
                                                      /*THCudaTensor_stride(state, gradOutput, 3)*/gosc,
                                                      /*THCudaTensor_stride(state, gradOutput, 2)*/gosw,
                                                      /*THCudaTensor_size(state, inputImages, 3)*/ic,
                                                      /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                      /*THCudaTensor_size(state, gradOutput, 2)*/gow,
                                                      zero_boundary);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}





#ifdef __cplusplus
}
#endif
