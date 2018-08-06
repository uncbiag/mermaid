#include <stdbool.h>
#include <stdio.h>
#include "my_lib_cuda_kernel_2D.h"


const int bdx =8;
const int bdy = 16;
//const int bdz =4;
#define real float

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

__device__ void getTopLeft(float x, int width, int& point, float& weight)
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

__device__ bool between_2D(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}

__device__ void sumReduceShMem(volatile float s[])
{
   /* obviously only works for 32 elements */
   /* sums up a shared memory array of 32 elements, stores it in s[0] */
   /* whole warp can then read first element (broadcasting) */
   if(threadIdx.x<16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16]; }
   if(threadIdx.x<8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
   if(threadIdx.x<4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
   if(threadIdx.x<2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
   if(threadIdx.x<1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
}



__global__ void bilinearSamplingFromGrid_2D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideHeight, int output_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int output_height, int output_width, int zero_boundary)
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   /////////////////////////////// threadIdx.x : used for features (coalescing is trivial)
   
   // dim3 blocks((sz1+15)/16, sz2, sz3);
   // dim3 threads(32,16);
   ///*THCudaTensor_size(state,output,2)*/int sz1
   /*THCudaTensor_size(state,output,1) int sz2,
   THCudaTensor_size(state,output,0)int sz3,*/
   //dim3 blocks((sz1+15)/16, (sz2+15)/16, sz3);  x: w/8, y: h/16, z: batch
   //dim3 threads(4,8,16);  x: c 4, y w:8, z h 16
   // threadIdx.x: only 2 of the blockDim.x is used in the grids part, but all of the threadIdx.x can be used in the sampling parts, so it is used in two parts for different propose
   {
  // block(h,w, b)
  //threads(h,w);
  // batch channel  x  y
  //  0      1      2  3 


    const int wOut = blockIdx.y*blockDim.y+threadIdx.y;
   const int hOut = blockIdx.x*blockDim.x+threadIdx.x;
   //const int idInBlock = threadIdx.x + threadIdx.y*blockDim.x;
   const bool withinImageBounds = wOut < output_width && hOut < output_height; // asume the size of input is the same as the output
   //const bool withinGridBounds = blockIdx.x*blockDim.y + idInBlock < output_width;//.........................
   
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
   int yInTopLeft, xInTopLeft;
   float yWeightTopLeft, xWeightTopLeft;
   getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
   getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);
     bool zero_boundary_bool = zero_boundary == 1;



   bool xBeyondLow = xInTopLeft < 0;
    bool yBeyondLow = yInTopLeft < 0;
    bool xBeyondHigh = xInTopLeft+1 > inputImages_width-1;
    bool yBeyondHigh = yInTopLeft+1 > inputImages_height-1;

    ///////////////  using  non zero border condition

    if (zero_boundary_bool) {
    if (xBeyondLow)
        xInTopLeft = 0;
    if (xBeyondHigh)
        xInTopLeft = inputImages_width-2;
    if (yBeyondLow)
        yInTopLeft = 0;
    if (yBeyondHigh)
        yInTopLeft = inputImages_height-2;
    }


   
   const int outAddress = output_strideBatch * b + output_strideHeight * hOut + output_strideWidth * wOut;
   const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
   const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
   const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
   const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

   float v=0;
   float inTopLeft=0;
   float inTopRight=0;
   float inBottomLeft=0;
   float inBottomRight=0;



   // interpolation happens here
   for(int t=0; t<inputImages_channels; t++)
   {

      if (zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh))){
          inTopLeft = inputImages_data[inTopLeftAddress + t*inputImages_strideChannels];
          inTopRight = inputImages_data[inTopRightAddress + t*inputImages_strideChannels];
          inBottomLeft = inputImages_data[inBottomLeftAddress + t*inputImages_strideChannels];
          inBottomRight = inputImages_data[inBottomRightAddress + t*inputImages_strideChannels];
      }

      v = xWeightTopLeft * yWeightTopLeft * inTopLeft
       + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
       + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
       + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;
       output_data[outAddress + t*output_strideChannels] = v;
    }
  }




template<bool onlyGrid> __global__ void backwardBilinearSampling_2D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideYX, int gradGrids_strideHeight, int gradGrids_strideWidth,
                                         float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, int gradOutput_strideHeight, int gradOutput_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int output_height, int output_width, int zero_boundary)
{
   const int wOut = blockIdx.y*blockDim.y+threadIdx.y;
   const int hOut = blockIdx.x*blockDim.x+threadIdx.x;
   //const int idInBlock = threadIdx.x + threadIdx.y*blockDim.x;
   const bool withinImageBounds = wOut < output_width && hOut < output_height;
   //const bool withinGridBounds = blockIdx.x*blockDim.y + idInBlock < output_width;//.........................
   
   const int b = blockIdx.z;
   float yf,xf;
   int grid_address =b*grids_strideBatch + hOut*grids_strideHeight + wOut*grids_strideWidth; // here we use the address of the 0th channel
   float gradyf=0;
   float gradxf=0;

   if(withinImageBounds)
   {
      xf = grids_data[grid_address];
      yf = grids_data[grid_address + grids_strideYX];  // address of the 1st channel

      
      int yInTopLeft, xInTopLeft;
      float yWeightTopLeft, xWeightTopLeft;
      getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
      getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);
        bool zero_boundary_bool = zero_boundary == 1;



      bool xBeyondLow = xInTopLeft < 0;
        bool yBeyondLow = yInTopLeft < 0;
        bool xBeyondHigh = xInTopLeft+1 > inputImages_width-1;
        bool yBeyondHigh = yInTopLeft+1 > inputImages_height-1;

        ///////////////  using  non zero border condition

        if (zero_boundary_bool) {
        if (xBeyondLow)
            xInTopLeft = 0;
        if (xBeyondHigh)
            xInTopLeft = inputImages_width-2;
        if (yBeyondLow)
            yInTopLeft = 0;
        if (yBeyondHigh)
            yInTopLeft = inputImages_height-2;
        }


      
      const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
      const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
      const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
      const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

      const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
      const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
      const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
      const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

      const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * hOut + gradOutput_strideWidth * wOut;

      float topLeftDotProduct = 0;
      float topRightDotProduct = 0;
      float bottomLeftDotProduct = 0;
      float bottomRightDotProduct = 0;



      for(int t=0; t<inputImages_channels; t++)
      {
        int tch = t*gradInputImages_strideChannels;
         float gradOutValue = gradOutput_data[gradOutputAddress + t*gradOutput_strideChannels];
         // bool between_2D(int value, int lowerBound, int upperBound)
         if (zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh))){
            float inTopLeft = inputImages_data[inTopLeftAddress + tch];
            topLeftDotProduct += inTopLeft * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopLeftAddress + tch], xWeightTopLeft * yWeightTopLeft * gradOutValue);


            float inTopRight = inputImages_data[inTopRightAddress + tch];
            topRightDotProduct += inTopRight * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopRightAddress + tch], (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue);

            float inBottomLeft = inputImages_data[inBottomLeftAddress + tch];
            bottomLeftDotProduct += inBottomLeft * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftAddress + tch], xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue);


            float inBottomRight = inputImages_data[inBottomRightAddress + tch];
            bottomRightDotProduct += inBottomRight * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomRightAddress + tch], (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue);
         }
      }

      gradyf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
      gradxf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;
      gradGrids_data[grid_address] = gradxf * (inputImages_width-1) / 2;
      gradGrids_data[grid_address+grids_strideYX] = gradyf * (inputImages_height-1) / 2;  
    }
     
}



#ifdef __cplusplus
extern "C" {
#endif




int BilinearSamplerBCWH_updateOutput_cuda_kernel_2D(/*THCudaTensor_size(state,output,2)*/int szw,
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
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
  // batch channel x y
  //  0      1     2 3 
   //dim3 blocks((THCudaTensor_size(state,output,2)+15)/16, THCudaTensor_size(state,output,1), THCudaTensor_size(state,output,0));
   dim3 blocks((oh+bdx-1)/bdx, (ow+bdy-1)/bdy, sz3);
   dim3 threads(bdx,bdy);
   //printf(" iw, ih, ow, oh  %d %d %d %d",iw,ih,ow,oh);

   /* assume BHWD */
   bilinearSamplingFromGrid_2D <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
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
                                                    /*THCudaTensor_size(state, output, 2)*/oh, ow,zero_boundary);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}







int BilinearSamplerBCWH_updateGradInput_cuda_kernel_2D(/*THCudaTensor_size(state,gradOutput,2)*/int szw,
                                                    /*THCudaTensor_size(state,gradOutput,1)*/int szc,
                                                    /*THCudaTensor_size(state,gradOutput,0)*/int sz3,
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
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

   //dim3 blocks((THCudaTensor_size(state,gradOutput,2)+15)/16, THCudaTensor_size(state,gradOutput,1), THCudaTensor_size(state,gradOutput,0));
   dim3 blocks((goh+bdx-1)/bdx, (gow+bdy-1)/bdy, sz3);
   dim3 threads(bdx,bdy);
   //int grids_channels=2;
   //printf("ow %d gsh %d  gsc %d osh %d osc %d\n",ow,gsh,gsc,osh,osc);

   backwardBilinearSampling_2D <false> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      /*THCudaTensor_data(state, inputImages)*/inputImages, 
                                                      /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                      /*THCudaTensor_stride(state, inputImages, 3)*/isc,
                                                      /*THCudaTensor_stride(state, inputImages, 1)*/ish,
                                                      /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                      /*THCudaTensor_data(state, gradInputImages)*/gradInputImages, 
                                                      /*THCudaTensor_stride(state, gradInputImages, 0)*/gisb,
                                                      /*THCudaTensor_stride(state, gradInputImages, 3)*/gisc,
                                                      /*THCudaTensor_stride(state, gradInputImages, 1)*/gish,
                                                      /*THCudaTensor_stride(state, gradInputImages, 2)*/gisw,
                                                      /*THCudaTensor_data(state, grids)*/grids, 
                                                      /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                      /*THCudaTensor_stride(state, grids, 3)*/gsc,
                                                      /*THCudaTensor_stride(state, grids, 1)*/gsh,
                                                      /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                      /*THCudaTensor_data(state, gradGrids)*/gradGrids, 
                                                      /*THCudaTensor_stride(state, gradGrids, 0)*/ggsb,
                                                      /*THCudaTensor_stride(state, gradGrids, 3)*/ggsc,
                                                      /*THCudaTensor_stride(state, gradGrids, 1)*/ggsh,
                                                      /*THCudaTensor_stride(state, gradGrids, 2)*/ggsw,
                                                      /*THCudaTensor_data(state, gradOutput)*/gradOutput, 
                                                      /*THCudaTensor_stride(state, gradOutput, 0)*/gosb,
                                                      /*THCudaTensor_stride(state, gradOutput, 3)*/gosc,
                                                      /*THCudaTensor_stride(state, gradOutput, 1)*/gosh,
                                                      /*THCudaTensor_stride(state, gradOutput, 2)*/gosw,
                                                      /*THCudaTensor_size(state, inputImages, 3)*/ic,
                                                      /*THCudaTensor_size(state, inputImages, 1)*/ih, 
                                                      /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                      /*THCudaTensor_size(state, gradOutput, 2)*/goh,gow,zero_boundary);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}

int BilinearSamplerBCWH_updateGradInputOnlyGrid_cuda_kernel_2D(
                                        /*THCudaTensor_size(state,gradOutput,2)*/int szw,
                                        /*THCudaTensor_size(state,gradOutput,1)*/int szc,
                                        /*THCudaTensor_size(state,gradOutput,0)*/int sz3,
                                        /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                        /*THCudaTensor_size(state, inputImages, 1)*/int iw,
                                        /*THCudaTensor_size(state, inputImages, 2)*/int ih,
                                        /*THCudaTensor_size(state, gradOutput, 2)*/int gow,
                                        /*THCudaTensor_size(state, gradOutput, 2)*/int goh,
                                        /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, 
                                        /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh, 
                                        /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsw, int ggsh,
                                        /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosw, int gosh,
                                        /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

   //dim3 blocks((THCudaTensor_size(state,gradOutput,2)+15)/16, THCudaTensor_size(state,gradOutput,1), THCudaTensor_size(state,gradOutput,0));
   dim3 blocks((goh+bdx-1)/bdx, (gow+bdy-1)/bdy, sz3);
   dim3 threads(bdx,bdy);
   //int grids_channels=2;
   //printf("ow %d gsh %d  gsc %d osh %d osc %d\n",ow,gsh,gsc,osh,osc);

   backwardBilinearSampling_2D <true> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      /*THCudaTensor_data(state, inputImages)*/inputImages, 
                                                      /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                      /*THCudaTensor_stride(state, inputImages, 3)*/isc,
                                                      /*THCudaTensor_stride(state, inputImages, 1)*/ish,
                                                      /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      /*THCudaTensor_data(state, grids)*/grids, 
                                                      /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                      /*THCudaTensor_stride(state, grids, 3)*/gsc,
                                                      /*THCudaTensor_stride(state, grids, 1)*/gsh,
                                                      /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                      /*THCudaTensor_data(state, gradGrids)*/gradGrids, 
                                                      /*THCudaTensor_stride(state, gradGrids, 0)*/ggsb,
                                                      /*THCudaTensor_stride(state, gradGrids, 3)*/ggsc,
                                                      /*THCudaTensor_stride(state, gradGrids, 1)*/ggsh,
                                                      /*THCudaTensor_stride(state, gradGrids, 2)*/ggsw,
                                                      /*THCudaTensor_data(state, gradOutput)*/gradOutput, 
                                                      /*THCudaTensor_stride(state, gradOutput, 0)*/gosb,
                                                      /*THCudaTensor_stride(state, gradOutput, 3)*/gosc,
                                                      /*THCudaTensor_stride(state, gradOutput, 1)*/gosh,
                                                      /*THCudaTensor_stride(state, gradOutput, 2)*/gosw,
                                                      /*THCudaTensor_size(state, inputImages, 3)*/ic,
                                                      /*THCudaTensor_size(state, inputImages, 1)*/ih, 
                                                      /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                      /*THCudaTensor_size(state, gradOutput, 2)*/goh, gow,zero_boundary);



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
