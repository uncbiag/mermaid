#include <stdbool.h>
#include <stdio.h>
#include "my_lib_cuda_kernel_3D.h"


const int bdx =16;
const int bdy = 16;
//const int bdz =4;
#define real float

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

__device__ void getTopLeftFront(float x, int width, int& point, float& weight)
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

__device__ bool between_3D(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}



__global__ void bilinearSamplingFromGrid_3D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, 
                                          int inputImages_strideDepth, int inputImages_strideHeight, int inputImages_strideWidth,
                                          float* grids_data, int grids_strideBatch, int grids_strideChannels, 
                                          int grids_strideDepth, int grids_strideHeight, int grids_strideWidth,
                                          float* output_data, int output_strideBatch, int output_strideChannels,
                                          int output_strideDepth, int output_strideHeight, int output_strideWidth,
                                          int inputImages_channels, int inputImages_depth, int inputImages_height, int inputImages_width,
                                          int outputImages_depth, int outputImages_height, int outputImages_width , int zero_boundary)
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
   // batch channel x y  z
   //  0      1     2 3  4
   //dim3 blocks((id+bdx-1)/bdx, (ih+bdy-1)/bdy, szwb);
   //dim3 threads(bdx,bdy);


   const int wOut = blockIdx.z % (outputImages_width);
   const int hOut = blockIdx.y*blockDim.y + threadIdx.y;
   const int dOut = blockIdx.x*blockDim.x + threadIdx.x;

   //const int idInBlock = threadIdx.x + threadIdx.y*blockDim.x;
   const bool withinImageBounds = dOut < outputImages_depth && hOut < outputImages_height; // asume the size of input is the same as the output
   //const bool withinGridBounds = blockIdx.x*blockDim.y + idInBlock < output_width;//.........................
   
   //const int channelIdx = blockDim.z /(inputImages_width);
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
   int zInTopLeft, yInTopLeft, xInTopLeft;  // zInTopFrontLeft
   float zWeightTopLeft, yWeightTopLeft, xWeightTopLeft;
   getTopLeftFront(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
   getTopLeftFront(yf, inputImages_height, yInTopLeft, yWeightTopLeft);
   getTopLeftFront(zf, inputImages_depth, zInTopLeft, zWeightTopLeft);
     bool zero_boundary_bool = zero_boundary == 1;



   bool xBeyondLow = xInTopLeft < 0;
   bool yBeyondLow = yInTopLeft < 0;
   bool zBeyondLow = zInTopLeft < 0;
   bool xBeyondHigh = xInTopLeft+1 > inputImages_width-1;
   bool yBeyondHigh = yInTopLeft+1 > inputImages_height-1;
   bool zBeyondHigh = zInTopLeft+1 > inputImages_depth-1;



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
    if (zBeyondLow)
        zInTopLeft = 0;
    if (zBeyondHigh)
        zInTopLeft = inputImages_depth-2;
    }


   
   const int outAddress = output_strideBatch * batchIdx + output_strideDepth * dOut + output_strideHeight * hOut + output_strideWidth * wOut;  // here assume the channel will be calculated later
   const int inTopLeftFrontAddress = inputImages_strideBatch * batchIdx + inputImages_strideDepth*zInTopLeft+ inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
   const int inTopRightFrontAddress = inTopLeftFrontAddress + inputImages_strideWidth;
   const int inBottomLeftFrontAddress = inTopLeftFrontAddress + inputImages_strideHeight;
   const int inBottomRightFrontAddress = inBottomLeftFrontAddress + inputImages_strideWidth;

   const int inTopLeftBehindAddress = inTopLeftFrontAddress + inputImages_strideDepth;
   const int inTopRightBehindAddress = inTopRightFrontAddress + inputImages_strideDepth;
   const int inBottomLeftBehindAddress = inBottomLeftFrontAddress + inputImages_strideDepth;
   const int inBottomRightBehindAddress = inBottomRightFrontAddress + inputImages_strideDepth;

   float v=0;
   float inTopLeftFront=0;
   float inTopRightFront=0;
   float inBottomLeftFront=0;
   float inBottomRightFront=0;

   float inTopLeftBehind=0;
   float inTopRightBedhind=0;
   float inBottomLeftBehind=0;
   float inBottomRightBehind=0;



   // interpolation happens here
   for(int t=0; t<inputImages_channels; t++)
   {

      if (zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh || zBeyondLow || zBeyondHigh))){
      inTopLeftFront = inputImages_data[inTopLeftFrontAddress + t*inputImages_strideChannels];
      inTopRightFront=inputImages_data[inTopRightFrontAddress + t*inputImages_strideChannels];
      inBottomLeftFront=inputImages_data[inBottomLeftFrontAddress + t*inputImages_strideChannels];
      inBottomRightFront=inputImages_data[inBottomRightFrontAddress + t*inputImages_strideChannels];

      inTopLeftBehind=inputImages_data[inTopLeftBehindAddress + t*inputImages_strideChannels];
      inTopRightBedhind=inputImages_data[inTopRightBehindAddress + t*inputImages_strideChannels];
      inBottomLeftBehind=inputImages_data[inBottomLeftBehindAddress + t*inputImages_strideChannels];
      inBottomRightBehind=inputImages_data[inBottomRightBehindAddress + t*inputImages_strideChannels];

      }

      v = xWeightTopLeft      * yWeightTopLeft       *    zWeightTopLeft      *  inTopLeftFront
       + (1 - xWeightTopLeft) * yWeightTopLeft       *    zWeightTopLeft      *  inTopRightFront
       + xWeightTopLeft       * (1 - yWeightTopLeft) *    zWeightTopLeft      *  inBottomLeftFront
       + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) *    zWeightTopLeft      *  inBottomRightFront
       + xWeightTopLeft       * yWeightTopLeft       *    (1-zWeightTopLeft)  *  inTopLeftBehind
       + (1 - xWeightTopLeft) * yWeightTopLeft       *    (1-zWeightTopLeft)  *  inTopRightBedhind
       + xWeightTopLeft       * (1 - yWeightTopLeft) *    (1-zWeightTopLeft)  *  inBottomLeftBehind
       + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) *    (1-zWeightTopLeft)  *  inBottomRightBehind;

       output_data[outAddress + t*output_strideChannels] = v;
    }
  }



 // inputImages,     isb, isc, isd, ish, isw,
 //                                                      gradInputImages, gisb, gisc, gisd, gish, gisw,
 //                                                      grids,           gsb, gsc , gsd, gsh, gsw,
 //                                                      gradGrids,       ggsb, ggsc, ggsd, ggsh, ggsw,
 //                                                      gradOutput,      gosb, gosc, gosd, gosh, gosw,
 //                                                     ic,   id, ih , iw, &count);

template<bool onlyGrid> __global__ void backwardBilinearSampling_3D(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels,
                                          int inputImages_strideDepth, int inputImages_strideHeight, int inputImages_strideWidth,
                                          float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels,
                                          int gradInputImages_strideDepth, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
                                          float* grids_data, int grids_strideBatch, int grids_strideChannels, 
                                          int grids_strideDepth, int grids_strideHeight, int grids_strideWidth,
                                          float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideChannels, 
                                          int gradGrids_strideDepth, int gradGrids_strideHeight, int gradGrids_strideWidth,
                                          float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, 
                                          int gradOutput_strideDepth,int gradOutput_strideHeight, int gradOutput_strideWidth,
                                          int inputImages_channels, int inputImages_depth, int inputImages_height, int inputImages_width,
                                          int outputImages_depth, int outputImages_height, int outputImages_width, int zero_boundary)
{
   const int wOut = blockIdx.z % (outputImages_width);
   const int hOut = blockIdx.y*blockDim.y + threadIdx.y;
   const int dOut = blockIdx.x*blockDim.x + threadIdx.x;

   //const int idInBlock = threadIdx.x + threadIdx.y*blockDim.x;
   const bool withinImageBounds = dOut < outputImages_depth && hOut < outputImages_height; // asume the size of input is the same as the output
   //const bool withinGridBounds = blockIdx.x*blockDim.y + idInBlock < output_width;//.........................
   
   //const int channelIdx = blockDim.z /(inputImages_width);
   const int batchIdx = blockIdx.z /(outputImages_width);
   float zf, yf,xf;
   int grid_address = batchIdx*grids_strideBatch +  dOut*grids_strideDepth + hOut*grids_strideHeight + wOut*grids_strideWidth; // here we use the address of the 0th channel
   float gradzf=0;
   float gradyf=0;
   float gradxf=0;

   if(withinImageBounds)
   {

      //atomicAdd(count,elem);
      xf = grids_data[grid_address];
      yf = grids_data[grid_address + grids_strideChannels];  // address of the 2nd channel
      zf = grids_data[grid_address + 2*grids_strideChannels];

      
      int zInTopLeft, yInTopLeft, xInTopLeft;  // zInTopFrontLeft
      float zWeightTopLeft, yWeightTopLeft, xWeightTopLeft;
      getTopLeftFront(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
      getTopLeftFront(yf, inputImages_height, yInTopLeft, yWeightTopLeft);
      getTopLeftFront(zf, inputImages_depth, zInTopLeft, zWeightTopLeft);
        bool zero_boundary_bool = zero_boundary == 1;






      bool xBeyondLow = xInTopLeft < 0;
       bool yBeyondLow = yInTopLeft < 0;
       bool zBeyondLow = zInTopLeft < 0;
       bool xBeyondHigh = xInTopLeft+1 > inputImages_width-1;
       bool yBeyondHigh = yInTopLeft+1 > inputImages_height-1;
       bool zBeyondHigh = zInTopLeft+1 > inputImages_depth-1;



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
        if (zBeyondLow)
            zInTopLeft = 0;
        if (zBeyondHigh)
            zInTopLeft = inputImages_depth-2;
        }




      
      //const int outAddress = output_strideBatch * batchIdx + output_strideDepth * dOut + output_strideHeight * hOut + output_strideWidth * wOut;  // here assume the channel will be calculated later
      const int inTopLeftFrontAddress = inputImages_strideBatch * batchIdx + inputImages_strideDepth*zInTopLeft+ inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
      const int inTopRightFrontAddress = inTopLeftFrontAddress + inputImages_strideWidth;
      const int inBottomLeftFrontAddress = inTopLeftFrontAddress + inputImages_strideHeight;
      const int inBottomRightFrontAddress = inBottomLeftFrontAddress + inputImages_strideWidth;

      const int inTopLeftBehindAddress = inTopLeftFrontAddress + inputImages_strideDepth;
      const int inTopRightBehindAddress = inTopRightFrontAddress + inputImages_strideDepth;
      const int inBottomLeftBehindAddress = inBottomLeftFrontAddress + inputImages_strideDepth;
      const int inBottomRightBehindAddress = inBottomRightFrontAddress + inputImages_strideDepth;

      const int gradInputImagesTopLeftFrontAddress = gradInputImages_strideBatch * batchIdx + gradInputImages_strideDepth*zInTopLeft+ gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
      const int gradInputImagesTopRightFrontAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideWidth;
      const int gradInputImagesBottomLeftFrontAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideHeight;
      const int gradInputImagesBottomRightFrontAddress = gradInputImagesBottomLeftFrontAddress + gradInputImages_strideWidth;

      const int gradInputImagesTopLeftBehindAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideDepth;
      const int gradInputImagesTopRightBehindAddress = gradInputImagesTopRightFrontAddress + gradInputImages_strideDepth;
      const int gradInputImagesBottomLeftBehindAddress = gradInputImagesBottomLeftFrontAddress + gradInputImages_strideDepth;
      const int gradInputImagesBottomRightBehindAddress = gradInputImagesBottomRightFrontAddress + gradInputImages_strideDepth;

      const int gradOutputAddress = gradOutput_strideBatch * batchIdx + gradOutput_strideDepth * dOut + gradOutput_strideHeight * hOut + gradOutput_strideWidth * wOut;

      float TopLeftFrontDP=0;
      float TopRightFrontDP=0;
      float BottomLeftFrontDP=0;
      float BottomRightFrontDP=0;

      float TopLeftBehindDP=0;
      float TopRightBehindDP=0;
      float BottomLeftBehindDP=0;
      float BottomRightBehindDP=0;

      for(int t=0; t<inputImages_channels; t++)
      {
        int tch = t*gradInputImages_strideChannels;
        float gradOutValue = gradOutput_data[gradOutputAddress + t*gradOutput_strideChannels];
         // bool between_3D(int value, int lowerBound, int upperBound)

         if (zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh || zBeyondLow || zBeyondHigh))){
            float inTopLeftFront = inputImages_data[inTopLeftFrontAddress + tch];
            TopLeftFrontDP += inTopLeftFront * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopLeftFrontAddress + tch], xWeightTopLeft * yWeightTopLeft * zWeightTopLeft * gradOutValue);

            float inTopRightFront = inputImages_data[inTopRightFrontAddress + tch];
            TopRightFrontDP += inTopRightFront * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopRightFrontAddress + tch], (1 - xWeightTopLeft) * yWeightTopLeft * zWeightTopLeft* gradOutValue);

            float inBottomLeftFront = inputImages_data[inBottomLeftFrontAddress + tch];
            BottomLeftFrontDP += inBottomLeftFront * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftFrontAddress + tch], xWeightTopLeft * (1 - yWeightTopLeft) * zWeightTopLeft * gradOutValue);



            float inBottomRightFront = inputImages_data[inBottomRightFrontAddress + tch];
            BottomRightFrontDP += inBottomRightFront * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomRightFrontAddress + tch], (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * zWeightTopLeft *gradOutValue);




            float inTopLeftBehind = inputImages_data[inTopLeftBehindAddress + tch];
            TopLeftBehindDP += inTopLeftBehind * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopLeftBehindAddress + tch], xWeightTopLeft * yWeightTopLeft * (1-zWeightTopLeft) * gradOutValue);

            float inTopRightBehind = inputImages_data[inTopRightBehindAddress + tch];
            TopRightBehindDP += inTopRightBehind * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopRightBehindAddress + tch], (1 - xWeightTopLeft) * yWeightTopLeft * (1-zWeightTopLeft)* gradOutValue);

            float inBottomLeftBehind = inputImages_data[inBottomLeftBehindAddress + tch];
            BottomLeftBehindDP += inBottomLeftBehind * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftBehindAddress + tch], xWeightTopLeft * (1 - yWeightTopLeft) * (1-zWeightTopLeft) * gradOutValue);

            float inBottomRightBehind = inputImages_data[inBottomRightBehindAddress + tch];
            BottomRightBehindDP += inBottomRightBehind * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomRightBehindAddress + tch], (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * (1-zWeightTopLeft)* gradOutValue);
         }

      }





      gradzf= xWeightTopLeft     * yWeightTopLeft       *    (-1)  *  TopLeftFrontDP
          + (1 - xWeightTopLeft) * yWeightTopLeft       *    (-1)  *  TopRightFrontDP
          + xWeightTopLeft       * (1 - yWeightTopLeft) *    (-1)  *  BottomLeftFrontDP
          + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) *    (-1)  *  BottomRightFrontDP
          + xWeightTopLeft       * yWeightTopLeft       *    1  *  TopLeftBehindDP
          + (1 - xWeightTopLeft) * yWeightTopLeft       *    1  *  TopRightBehindDP
          + xWeightTopLeft       * (1 - yWeightTopLeft) *    1  *  BottomLeftBehindDP
          + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) *    1  *  BottomRightBehindDP;


      gradyf= xWeightTopLeft     * (-1)       *    zWeightTopLeft     *  TopLeftFrontDP
          + (1 - xWeightTopLeft) * (-1)       *    zWeightTopLeft     *  TopRightFrontDP
          + xWeightTopLeft       * 1        *    zWeightTopLeft     *  BottomLeftFrontDP
          + (1 - xWeightTopLeft) * 1        *    zWeightTopLeft     *  BottomRightFrontDP
          + xWeightTopLeft       * (-1)       *    (1-zWeightTopLeft) *  TopLeftBehindDP
          + (1 - xWeightTopLeft) * (-1)       *    (1-zWeightTopLeft) *  TopRightBehindDP
          + xWeightTopLeft       * 1        *    (1-zWeightTopLeft) *  BottomLeftBehindDP
          + (1 - xWeightTopLeft) * 1        *    (1-zWeightTopLeft) *  BottomRightBehindDP;

      gradxf= (-1)  * yWeightTopLeft       *    zWeightTopLeft      *  TopLeftFrontDP
          + 1     * yWeightTopLeft       *    zWeightTopLeft      *  TopRightFrontDP
          + (-1)    * (1 - yWeightTopLeft) *    zWeightTopLeft      *  BottomLeftFrontDP
          + 1     * (1 - yWeightTopLeft) *    zWeightTopLeft      *  BottomRightFrontDP
          + (-1)    * yWeightTopLeft       *    (1-zWeightTopLeft)  *  TopLeftBehindDP
          + 1     * yWeightTopLeft       *    (1-zWeightTopLeft)  *  TopRightBehindDP
          + (-1)    * (1 - yWeightTopLeft) *    (1-zWeightTopLeft)  *  BottomLeftBehindDP
          + 1     * (1 - yWeightTopLeft) *    (1-zWeightTopLeft)  *  BottomRightBehindDP;

      gradGrids_data[grid_address] = gradxf * (inputImages_width-1) / 2;   // change to adpat cpu code
      gradGrids_data[grid_address + grids_strideChannels] = gradyf * (inputImages_height-1) / 2;
      gradGrids_data[grid_address + 2*grids_strideChannels] = gradzf * (inputImages_depth-1) / 2; 

    }
       
}



#ifdef __cplusplus
extern "C" {
#endif






int BilinearSamplerBCWHD_updateOutput_cuda_kernel_3D(/*output->size[2]*/int szw,
                                                 /*output->size[1]*/int szc,
                                                 /*output->size[0]*/int szb,
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
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
  // batch channel x y  z
  //  0      1     2 3  4
   //dim3 blocks((output->size[2]+15)/16, output->size[1], output->size[0]);
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
   bilinearSamplingFromGrid_3D <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                    inputImages, isb, isc, isd, ish, isw,
                                                    grids,       gsb, gsc, gsd, gsh, gsw,
                                                    output,      osb, osc, osd, osh, osw,
                                                    ic,  id,  ih, iw, od, oh, ow, zero_boundary);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}







int BilinearSamplerBCWHD_updateGradInput_cuda_kernel_3D(/*gradOutput->size[2]*/int szw, 
                                                    /*gradOutput->size[1]*/int szc,
                                                    /*gradOutput->size[0]*/int szb,
                                                    /*THCudaTensor_size(state, inputImages, 1)*/int ic,
                                                    /*THCudaTensor_size(state, inputImages, 2)*/int iw,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/int ih,
                                                    /*THCudaTensor_size(state, inputImages, 4)*/int id,
                                                    /*THCudaTensor_size(state, inputImages, 2)*/int ow,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/int oh,
                                                    /*THCudaTensor_size(state, inputImages, 4)*/int od,
                                                    /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, int isd, 
                                                    /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh,  int gsd,
                                                    /*THCudaTensor *gradInputImages*/float *gradInputImages, int gisb, int gisc, int gisw, int gish, int gisd,
                                                    /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsw, int ggsh, int ggsd,
                                                    /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosw, int gosh, int gosd,
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

   //dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 blocks((od+bdx-1)/bdx, (oh+bdy-1)/bdy, szw*szb);
   dim3 threads(bdx,bdy);
   //float count =0;
   //int grids_channels=2;
   //printf("ic %d   gsc %d gosc %d\n",ic,gsc,gosc);
   //printf(" gsh %d  gsc %d gosh %d gosc %d  ic %d id %d ih %d\n!!!!",gsh,gsc,gosh,gosc,ic,id,ih);


//                                            float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels,
//                                           int inputImages_strideDepth, int inputImages_strideHeight, int inputImages_strideWidth,
//                                           float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels,
//                                           int gradInputImages_strideDepth, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
//                                           float* grids_data, int grids_strideBatch, int grids_strideChannels, 
//                                           int grids_strideDepth, int grids_strideHeight, int grids_strideWidth,
//                                           float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideChannels, 
//                                           int gradGrids_strideDepth, int gradGrids_strideHeight, int gradGrids_strideWidth,
//                                           float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, 
//                                           int gradOutput_strideDepth,int gradOutput_strideHeight, int gradOutput_strideWidth,
//                                           int inputImages_channels, int inputImages_depth, int inputImages_height, int inputImages_width)
   backwardBilinearSampling_3D <false> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      inputImages,     isb, isc, isd, ish, isw,
                                                      gradInputImages, gisb, gisc, gisd, gish, gisw,
                                                      grids,           gsb, gsc , gsd, gsh, gsw,
                                                      gradGrids,       ggsb, ggsc, ggsd, ggsh, ggsw,
                                                      gradOutput,      gosb, gosc, gosd, gosh, gosw,
                                                      ic,   id, ih , iw, od, oh, ow, zero_boundary);
   //printf("num of count %f",count);
   //printf(" gsh %d  gsc %d gosh %d gosc %d  ic %d id %d ih %d\n!!!!",gsh,gsc,gosh,gosc,ic,id,ih);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}

int BilinearSamplerBCWHD_updateGradInputOnlyGrid_cuda_kernel_3D(/*gradOutput->size[2]*/int szw, 
                                                    /*gradOutput->size[1]*/int szc,
                                                    /*gradOutput->size[0]*/int szb,
                                                    /*THCudaTensor_size(state, inputImages, 1)*/int ic,
                                                    /*THCudaTensor_size(state, inputImages, 2)*/int iw,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/int ih,
                                                    /*THCudaTensor_size(state, inputImages, 4)*/int id,
                                                    /*THCudaTensor_size(state, inputImages, 2)*/int ow,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/int oh,
                                                    /*THCudaTensor_size(state, inputImages, 4)*/int od,
                                                    /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isw, int ish, int isd, 
                                                    /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsw, int gsh,  int gsd,
                                                    /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsw, int ggsh, int ggsd,
                                                    /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosw, int gosh, int gosd,
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream, int zero_boundary)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

   //dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 blocks((od+bdx-1)/bdx, (oh+bdy-1)/bdy, szw*szb);
   dim3 threads(bdx,bdy);
   //float* count;

   //int grids_channels=2;
   //printf("ow %d gsh %d  gsc %d osh %d osc %d\n",ow,gsh,gsc,osh,osc);

   backwardBilinearSampling_3D <true> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      inputImages,     isb, isc, isd, ish, isw,
                                                      0, 0, 0, 0, 0, 0,
                                                      grids,           gsb, gsc , gsd, gsh, gsw,
                                                      gradGrids,       ggsb, ggsc, ggsd, ggsh, ggsw,
                                                      gradOutput,      gosb, gosc, gosd, gosh, gosw,
                                                      ic,   id, ih , iw, od, oh, ow, zero_boundary);



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
