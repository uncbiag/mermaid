#ifndef no_openmp
#include <omp.h>
#endif

#include <TH/TH.h>
#include <stdbool.h>
#include <stdio.h>
#include "my_lib_cuda_1D.h"
#include "my_lib_cuda_2D.h"
#include "my_lib_cuda_3D.h"

#define real float

int BilinearSamplerBXC_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
{
  // *B*atch, *X*-coors, *C*hannel
  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[1];
  int inputImages_C = inputImages->size[2];
  int output_X = output->size[1];
  int output_C = output->size[2];

  int output_strideBatch = output->stride[0];
  int output_stride_X = output->stride[1];
  int output_stride_C = output->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[1];
  int inputImages_stride_C = inputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[1];
  int grids_stride_C = grids->stride[2];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, xOut;

  for(b=0; b < batchsize; b++)
    {
      #pragma omp parallel for
      for(xOut=0; xOut < output_X; xOut++)
	{
	  //read the grid
	  real xf = grids_data[b*grids_strideBatch + xOut*grids_stride_X];

	  // get the weights for interpolation
	  int xLow;
	  real xWeightLow;

	  real xcoord = (xf + 1) * (inputImages_X - 1) / 2; // map it from [-1,1] to [0,1]
	  xLow = floor(xcoord);
	  xWeightLow = 1 - (xcoord - xLow);

	  const int outAddress = output_strideBatch * b + output_stride_X * xOut;
	  const int inLowAddress = inputImages_strideBatch * b + inputImages_stride_X * xLow;
	  const int inHighAddress = inLowAddress + inputImages_stride_X;

	  real v=0;
	  real inLow=0;
	  real inHigh=0;

	  // we are careful with the boundaries
	  bool lowIsIn = xLow >= 0 && xLow <= inputImages_X-1; 
	  bool highIsIn = xLow+1 >= 0 && xLow+1 <= inputImages_X-1;

	  int t;
	  // interpolation happens here
	  for(t=0; t<inputImages_C; t++)
	    {
	      if(lowIsIn) inLow = inputImages_data[inLowAddress + t];
	      if(highIsIn) inHigh = inputImages_data[inHighAddress + t];

	      v = xWeightLow * inLow + (1 - xWeightLow) * inHigh;

	      output_data[outAddress + t] = v;
	    }

	}
    }

  return 1;
}

int BilinearSamplerBCX_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
{
  // *B*atch, *C*hannel, *X*-coors
  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[2];
  int inputImages_C = inputImages->size[1];
  int output_X = output->size[2];
  int output_C = output->size[1];

  int output_strideBatch = output->stride[0];
  int output_stride_X = output->stride[2];
  int output_stride_C = output->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[2];
  int inputImages_stride_C = inputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[2];
  int grids_stride_C = grids->stride[1];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, xOut;

  for(b=0; b < batchsize; b++)
    {
      #pragma omp parallel for
      for(xOut=0; xOut < output_X; xOut++)
	{
	  //read the grid
	  real xf = grids_data[b*grids_strideBatch + xOut*grids_stride_X];

	  // get the weights for interpolation
	  int xLow;
	  real xWeightLow;

	  real xcoord = (xf + 1) * (inputImages_X - 1) / 2; // map it from [-1,1] to [0,1]
	  xLow = floor(xcoord);
	  xWeightLow = 1 - (xcoord - xLow);

	  const int outAddress = output_strideBatch * b + output_stride_X * xOut;
	  const int inLowAddress = inputImages_strideBatch * b + inputImages_stride_X * xLow;
	  const int inHighAddress = inLowAddress + inputImages_stride_X;

	  real v=0;
	  real inLow=0;
	  real inHigh=0;

	  // we are careful with the boundaries
	  bool lowIsIn = xLow >= 0 && xLow <= inputImages_X-1; 
	  bool highIsIn = xLow+1 >= 0 && xLow+1 <= inputImages_X-1;

	  int t;
	  // interpolation happens here
	  for(t=0; t<inputImages_C; t++)
	    {
	      if(lowIsIn) inLow = inputImages_data[inLowAddress + t*inputImages_stride_C];
	      if(highIsIn) inHigh = inputImages_data[inHighAddress + t*inputImages_stride_C];

	      v = xWeightLow * inLow + (1 - xWeightLow) * inHigh;

	      output_data[outAddress + t*output_stride_C] = v;
	    }

	}
    }

  return 1;
}

int BilinearSamplerBXYC_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
{
  // This is actua
  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[1];
  int inputImages_Y = inputImages->size[2];
  int inputImages_C = inputImages->size[3];
  int output_X = output->size[1];
  int output_Y = output->size[2];

  int output_strideBatch = output->stride[0];
  int output_stride_X = output->stride[1];
  int output_stride_Y = output->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[1];
  int inputImages_stride_Y = inputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[1];
  int grids_stride_Y = grids->stride[2];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < output_X; xOut++)
    {
      for(yOut=0; yOut < output_Y; yOut++)
      {
        //read the grid
        real xf = grids_data[b*grids_strideBatch + yOut*grids_stride_Y + xOut*grids_stride_X];
        real yf = grids_data[b*grids_strideBatch + yOut*grids_stride_Y + xOut*grids_stride_X+1];

        // get the weights for interpolation
        int yInLowLow, xInLowLow;
        real yWeightLowLow, xWeightLowLow;

        real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
        xInLowLow = floor(xcoord);
        xWeightLowLow = 1 - (xcoord - xInLowLow);

        real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
        yInLowLow = floor(ycoord);
        yWeightLowLow = 1 - (ycoord - yInLowLow);

        const int outAddress = output_strideBatch * b + output_stride_Y * yOut + output_stride_X * xOut;
        const int inLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Y * yInLowLow + inputImages_stride_X * xInLowLow;
        const int inLowHighAddress = inLowLowAddress + inputImages_stride_Y;
        const int inHighLowAddress = inLowLowAddress + inputImages_stride_X;
        const int inHighHighAddress = inHighLowAddress + inputImages_stride_Y;

        real v=0;
        real inLowLow=0;
        real inLowHigh=0;
        real inHighLow=0;
        real inHighHigh=0;

        // we are careful with the boundaries
        bool lowLowIsIn =  xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;
        bool lowHighIsIn = xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow+1 >= 0 && yInLowLow+1 <= inputImages_Y-1;
        bool highLowIsIn = xInLowLow+1 >= 0 && xInLowLow+1 <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;
        bool highHighIsIn = xInLowLow+1 >= 0 && xInLowLow+1 <= inputImages_X-1 && yInLowLow+1 >= 0 && yInLowLow+1 <= inputImages_Y-1;

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_C; t++)
        {
           if(lowLowIsIn) inLowLow = inputImages_data[inLowLowAddress + t];
           if(lowHighIsIn) inLowHigh = inputImages_data[inLowHighAddress + t];
           if(highLowIsIn) inHighLow = inputImages_data[inHighLowAddress + t];
           if(highHighIsIn) inHighHigh = inputImages_data[inHighHighAddress + t];

           v = xWeightLowLow * yWeightLowLow * inLowLow
             + (1 - xWeightLowLow) * yWeightLowLow * inHighLow
             + xWeightLowLow * (1 - yWeightLowLow) * inLowHigh
             + (1 - xWeightLowLow) * (1 - yWeightLowLow) * inHighHigh;

           output_data[outAddress + t] = v;
        }

      }
    }
  }

  return 1;

}


int BilinearSamplerBCXY_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
{
  // This is actua
  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[2];
  int inputImages_Y = inputImages->size[3];
  int inputImages_C = inputImages->size[1];
  int output_X = output->size[2];
  int output_Y = output->size[3];

  int output_strideBatch = output->stride[0];
  int output_stride_X = output->stride[2];
  int output_stride_Y = output->stride[3];
  int output_stride_C = output->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[2];
  int inputImages_stride_Y = inputImages->stride[3];
  int inputImages_stride_C = inputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_C = grids->stride[1];
  int grids_stride_X = grids->stride[2];
  int grids_stride_Y = grids->stride[3];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < output_X; xOut++)
    {
      for(yOut=0; yOut < output_Y; yOut++)
      {
        //read the grid
        real xf = grids_data[b*grids_strideBatch + yOut*grids_stride_Y + xOut*grids_stride_X];
        real yf = grids_data[b*grids_strideBatch + yOut*grids_stride_Y + xOut*grids_stride_X+grids_stride_C];

        // get the weights for interpolation
        int yInLowLow, xInLowLow;
        real yWeightLowLow, xWeightLowLow;

        real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
        xInLowLow = floor(xcoord);
        xWeightLowLow = 1 - (xcoord - xInLowLow);

        real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
        yInLowLow = floor(ycoord);
        yWeightLowLow = 1 - (ycoord - yInLowLow);

        const int outAddress = output_strideBatch * b + output_stride_Y * yOut + output_stride_X * xOut;
        const int inLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Y * yInLowLow + inputImages_stride_X * xInLowLow;
        const int inLowHighAddress = inLowLowAddress + inputImages_stride_Y;
        const int inHighLowAddress = inLowLowAddress + inputImages_stride_X;
        const int inHighHighAddress = inHighLowAddress + inputImages_stride_Y;

        real v=0;
        real inLowLow=0;
        real inLowHigh=0;
        real inHighLow=0;
        real inHighHigh=0;

        // we are careful with the boundaries
        bool lowLowIsIn =  xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;
        bool lowHighIsIn = xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow+1 >= 0 && yInLowLow+1 <= inputImages_Y-1;
        bool highLowIsIn = xInLowLow+1 >= 0 && xInLowLow+1 <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;
        bool highHighIsIn = xInLowLow+1 >= 0 && xInLowLow+1 <= inputImages_X-1 && yInLowLow+1 >= 0 && yInLowLow+1 <= inputImages_Y-1;

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_C; t++)
        {
           if(lowLowIsIn) inLowLow = inputImages_data[inLowLowAddress + t*inputImages_stride_C];
           if(lowHighIsIn) inLowHigh = inputImages_data[inLowHighAddress + t*inputImages_stride_C];
           if(highLowIsIn) inHighLow = inputImages_data[inHighLowAddress + t*inputImages_stride_C];
           if(highHighIsIn) inHighHigh = inputImages_data[inHighHighAddress + t*inputImages_stride_C];

           v = xWeightLowLow * yWeightLowLow * inLowLow
             + (1 - xWeightLowLow) * yWeightLowLow * inHighLow
             + xWeightLowLow * (1 - yWeightLowLow) * inLowHigh
             + (1 - xWeightLowLow) * (1 - yWeightLowLow) * inHighHigh;

           output_data[outAddress + t*output_stride_C] = v;
        }

      }
    }
  }

  return 1;

}


int BilinearSamplerBXYZC_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
{
  // This is actua
  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[1];
  int inputImages_Y = inputImages->size[2];
  int inputImages_Z = inputImages->size[3];
  int inputImages_C = inputImages->size[4];
  int output_X = output->size[1];
  int output_Y = output->size[2];
  int output_Z = output->size[3];

  int output_strideBatch = output->stride[0];
  int output_stride_X = output->stride[1];
  int output_stride_Y = output->stride[2];
  int output_stride_Z = output->stride[3];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[1];
  int inputImages_stride_Y = inputImages->stride[2];
  int inputImages_stride_Z = inputImages->stride[3];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[1];
  int grids_stride_Y = grids->stride[2];
  int grids_stride_Z = grids->stride[3];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, zOut, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < output_X; xOut++)
      {
	for(yOut=0; yOut < output_Y; yOut++)
	  {
	    for(zOut=0; zOut < output_Z; zOut++)
	      {	
		//read the grid
		real xf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X];
		real yf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X+1];
		real zf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X+2];

		// get the weights for interpolation
		int zInLowLowLow, yInLowLowLow, xInLowLowLow;
		real zWeightLowLowLow, yWeightLowLowLow, xWeightLowLowLow;

		real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
		xInLowLowLow = floor(xcoord);
		xWeightLowLowLow = 1 - (xcoord - xInLowLowLow);

		real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
		yInLowLowLow = floor(ycoord);
		yWeightLowLowLow = 1 - (ycoord - yInLowLowLow);

		real zcoord = (zf + 1) * (inputImages_Z - 1) / 2;
		zInLowLowLow = floor(zcoord);
		zWeightLowLowLow = 1 - (zcoord - zInLowLowLow);

		const int outAddress = output_strideBatch * b + output_stride_Z * zOut + output_stride_Y * yOut + output_stride_X * xOut;
		const int inLowLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Z * zInLowLowLow + inputImages_stride_Y * yInLowLowLow + inputImages_stride_X * xInLowLowLow;
		const int inLowLowHighAddress = inLowLowLowAddress + inputImages_stride_Z;
		const int inLowHighLowAddress = inLowLowLowAddress + inputImages_stride_Y;
		const int inLowHighHighAddress = inLowLowLowAddress + inputImages_stride_Y + inputImages_stride_Z;
		const int inHighLowLowAddress = inLowLowLowAddress + inputImages_stride_X;
		const int inHighLowHighAddress = inLowLowLowAddress + inputImages_stride_X + inputImages_stride_Z;
		const int inHighHighLowAddress = inLowLowLowAddress + inputImages_stride_X + inputImages_stride_Y;
		const int inHighHighHighAddress = inLowLowLowAddress + inputImages_stride_X + inputImages_stride_Y + inputImages_stride_Z;

		real v=0;
		real inLowLowLow=0;
		real inLowLowHigh=0;
		real inLowHighLow=0;
		real inLowHighHigh=0;
		real inHighLowLow=0;
		real inHighLowHigh=0;
		real inHighHighLow=0;
		real inHighHighHigh=0;

		// we are careful with the boundaries
		bool lowLowLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
		bool lowLowHighIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
		bool lowHighLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
		bool lowHighHighIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
		  && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
		bool highLowLowIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
		bool highLowHighIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
		bool highHighLowIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
		  && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
		bool highHighHighIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
		  && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
		  && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;

		int t;
		// interpolation happens here
		for(t=0; t<inputImages_C; t++)
		  {
		    if(lowLowLowIsIn) inLowLowLow = inputImages_data[inLowLowLowAddress + t];
		    if(lowLowHighIsIn) inLowLowHigh = inputImages_data[inLowLowHighAddress + t];
		    if(lowHighLowIsIn) inLowHighLow = inputImages_data[inLowHighLowAddress + t];
		    if(lowHighHighIsIn) inLowHighHigh = inputImages_data[inLowHighHighAddress + t];
		    if(highLowLowIsIn) inHighLowLow = inputImages_data[inHighLowLowAddress + t];
		    if(highLowHighIsIn) inHighLowHigh = inputImages_data[inHighLowHighAddress + t];
		    if(highHighLowIsIn) inHighHighLow = inputImages_data[inHighHighLowAddress + t];
		    if(highHighHighIsIn) inHighHighHigh = inputImages_data[inHighHighHighAddress + t];

		    v = xWeightLowLowLow * yWeightLowLowLow * zWeightLowLowLow * inLowLowLow
		      + xWeightLowLowLow * yWeightLowLowLow * (1-zWeightLowLowLow) * inLowLowHigh
		      + xWeightLowLowLow * (1-yWeightLowLowLow) * zWeightLowLowLow * inLowHighLow
		      + xWeightLowLowLow * (1-yWeightLowLowLow) * (1-zWeightLowLowLow) * inLowHighHigh
		      + (1-xWeightLowLowLow) * yWeightLowLowLow * zWeightLowLowLow * inHighLowLow
		      + (1-xWeightLowLowLow) * yWeightLowLowLow * (1-zWeightLowLowLow) * inHighLowHigh
		      + (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * zWeightLowLowLow * inHighHighLow
		      + (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * (1-zWeightLowLowLow) * inHighHighHigh;

		    output_data[outAddress + t] = v;
		  }

	      }
	  }
      }
  }
  return 1;

}

int BilinearSamplerBCXYZ_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
{
  // This is actua
  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[2];
  int inputImages_Y = inputImages->size[3];
  int inputImages_Z = inputImages->size[4];
  int inputImages_C = inputImages->size[1];
  int output_X = output->size[2];
  int output_Y = output->size[3];
  int output_Z = output->size[4];

  int output_strideBatch = output->stride[0];
  int output_stride_X = output->stride[2];
  int output_stride_Y = output->stride[3];
  int output_stride_Z = output->stride[4];
  int output_stride_C = output->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[2];
  int inputImages_stride_Y = inputImages->stride[3];
  int inputImages_stride_Z = inputImages->stride[4];
  int inputImages_stride_C = inputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[2];
  int grids_stride_Y = grids->stride[3];
  int grids_stride_Z = grids->stride[4];
  int grids_stride_C = grids->stride[1];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, zOut, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < output_X; xOut++)
      {
	for(yOut=0; yOut < output_Y; yOut++)
	  {
	    for(zOut=0; zOut < output_Z; zOut++)
	      {	
		//read the grid
		real xf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X];
		real yf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X+grids_stride_C];
		real zf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X+2*grids_stride_C];

		// get the weights for interpolation
		int zInLowLowLow, yInLowLowLow, xInLowLowLow;
		real zWeightLowLowLow, yWeightLowLowLow, xWeightLowLowLow;

		real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
		xInLowLowLow = floor(xcoord);
		xWeightLowLowLow = 1 - (xcoord - xInLowLowLow);

		real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
		yInLowLowLow = floor(ycoord);
		yWeightLowLowLow = 1 - (ycoord - yInLowLowLow);

		real zcoord = (zf + 1) * (inputImages_Z - 1) / 2;
		zInLowLowLow = floor(zcoord);
		zWeightLowLowLow = 1 - (zcoord - zInLowLowLow);

		const int outAddress = output_strideBatch * b + output_stride_Z * zOut + output_stride_Y * yOut + output_stride_X * xOut;
		const int inLowLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Z * zInLowLowLow + inputImages_stride_Y * yInLowLowLow + inputImages_stride_X * xInLowLowLow;
		const int inLowLowHighAddress = inLowLowLowAddress + inputImages_stride_Z;
		const int inLowHighLowAddress = inLowLowLowAddress + inputImages_stride_Y;
		const int inLowHighHighAddress = inLowLowLowAddress + inputImages_stride_Y + inputImages_stride_Z;
		const int inHighLowLowAddress = inLowLowLowAddress + inputImages_stride_X;
		const int inHighLowHighAddress = inLowLowLowAddress + inputImages_stride_X + inputImages_stride_Z;
		const int inHighHighLowAddress = inLowLowLowAddress + inputImages_stride_X + inputImages_stride_Y;
		const int inHighHighHighAddress = inLowLowLowAddress + inputImages_stride_X + inputImages_stride_Y + inputImages_stride_Z;

		real v=0;
		real inLowLowLow=0;
		real inLowLowHigh=0;
		real inLowHighLow=0;
		real inLowHighHigh=0;
		real inHighLowLow=0;
		real inHighLowHigh=0;
		real inHighHighLow=0;
		real inHighHighHigh=0;

		// we are careful with the boundaries
		bool lowLowLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
		bool lowLowHighIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
		bool lowHighLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
		bool lowHighHighIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
		  && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
		bool highLowLowIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
		bool highLowHighIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
		bool highHighLowIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
		  && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
		bool highHighHighIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
		  && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
		  && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;

		int t;
		// interpolation happens here
		for(t=0; t<inputImages_C; t++)
		  {
		    if(lowLowLowIsIn) inLowLowLow = inputImages_data[inLowLowLowAddress + t*inputImages_stride_C];
		    if(lowLowHighIsIn) inLowLowHigh = inputImages_data[inLowLowHighAddress + t*inputImages_stride_C];
		    if(lowHighLowIsIn) inLowHighLow = inputImages_data[inLowHighLowAddress + t*inputImages_stride_C];
		    if(lowHighHighIsIn) inLowHighHigh = inputImages_data[inLowHighHighAddress + t*inputImages_stride_C];
		    if(highLowLowIsIn) inHighLowLow = inputImages_data[inHighLowLowAddress + t*inputImages_stride_C];
		    if(highLowHighIsIn) inHighLowHigh = inputImages_data[inHighLowHighAddress + t*inputImages_stride_C];
		    if(highHighLowIsIn) inHighHighLow = inputImages_data[inHighHighLowAddress + t*inputImages_stride_C];
		    if(highHighHighIsIn) inHighHighHigh = inputImages_data[inHighHighHighAddress + t*inputImages_stride_C];

		    v = xWeightLowLowLow * yWeightLowLowLow * zWeightLowLowLow * inLowLowLow
		      + xWeightLowLowLow * yWeightLowLowLow * (1-zWeightLowLowLow) * inLowLowHigh
		      + xWeightLowLowLow * (1-yWeightLowLowLow) * zWeightLowLowLow * inLowHighLow
		      + xWeightLowLowLow * (1-yWeightLowLowLow) * (1-zWeightLowLowLow) * inLowHighHigh
		      + (1-xWeightLowLowLow) * yWeightLowLowLow * zWeightLowLowLow * inHighLowLow
		      + (1-xWeightLowLowLow) * yWeightLowLowLow * (1-zWeightLowLowLow) * inHighLowHigh
		      + (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * zWeightLowLowLow * inHighHighLow
		      + (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * (1-zWeightLowLowLow) * inHighHighHigh;

		    output_data[outAddress + t*output_stride_C] = v;
		  }

	      }
	  }
      }
  }
  return 1;

}

int BilinearSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
{

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[1];
  int inputImages_width = inputImages->size[2];
  int output_height = output->size[1];
  int output_width = output->size[2];
  int inputImages_channels = inputImages->size[3];

  int output_strideBatch = output->stride[0];
  int output_strideHeight = output->stride[1];
  int output_strideWidth = output->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];


  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < output_height; yOut++)
    {
      for(xOut=0; xOut < output_width; xOut++)
      {
        //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;

        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);



        const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        real v=0;
        real inTopLeft=0;
        real inTopRight=0;
        real inBottomLeft=0;
        real inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_channels; t++)
        {
           if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
           if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
           if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
           if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];

           v = xWeightTopLeft * yWeightTopLeft * inTopLeft
             + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
             + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
             + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

           output_data[outAddress + t] = v;
        }

      }
    }
  }

  return 1;
}

int BilinearSamplerBXYZC_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim)
{
  switch( ndim )
    {
    case 1: return BilinearSamplerBXC_updateOutput_1D( inputImages, grids, output ); break;
    case 2: return BilinearSamplerBXYC_updateOutput_2D( inputImages, grids, output ); break;
    case 3: return BilinearSamplerBXYZC_updateOutput_3D( inputImages, grids, output ); break;
    default: return -1;
    }
}

int BilinearSamplerBCXYZ_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim, _Bool use_cuda, int ndevice)
{
  if(!use_cuda){
    switch( ndim )
      {
      case 1: return BilinearSamplerBCX_updateOutput_1D( inputImages, grids, output ); break;
      case 2: return BilinearSamplerBCXY_updateOutput_2D( inputImages, grids, output ); break;
      case 3: return BilinearSamplerBCXYZ_updateOutput_3D( inputImages, grids, output ); break;
      default: return -1;
      }
  }
  else{
    switch( ndim ){
      case 1: return BilinearSamplerBCW_updateOutput_cuda_1D( inputImages, grids, output , ndevice); break;
      case 2: return BilinearSamplerBCWH_updateOutput_cuda_2D( inputImages, grids, output , ndevice ); break;
      case 3: return BilinearSamplerBCWHD_updateOutput_cuda_3D( inputImages, grids, output , ndevice ); break;
      default: return -1;
    }


  }
}

int BilinearSamplerBXC_updateGradInput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput )
{
  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[1];
  int gradOutput_X = gradOutput->size[1];
  int inputImages_C = inputImages->size[2];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_stride_X = gradOutput->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[1];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_stride_X = gradInputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[1];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_stride_X = gradGrids->stride[1];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < gradOutput_X; xOut++)
    {
      //read the grid
      real xf = grids_data[b*grids_strideBatch + xOut*grids_stride_X];

      // get the weights for interpolation
      int xInLow;
      real xWeightLow;

      real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
      xInLow = floor(xcoord);
      xWeightLow = 1 - (xcoord - xInLow);

      const int inLowAddress = inputImages_strideBatch * b + inputImages_stride_X * xInLow;
      const int inHighAddress = inLowAddress + inputImages_stride_X;

      const int gradInputImagesLowAddress = gradInputImages_strideBatch * b + gradInputImages_stride_X * xInLow;
      const int gradInputImagesHighAddress = gradInputImagesLowAddress + gradInputImages_stride_X;
      
      const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_stride_X * xOut;
      
      real lowDotProduct = 0;
      real highDotProduct = 0;
      
      real v=0;
      real inLow=0;
      real inHigh=0;
      
      // we are careful with the boundaries
      bool lowIsIn = xInLow >= 0 && xInLow <= inputImages_X-1;
      bool highIsIn = xInLow+1 >= 0 && xInLow+1 <= inputImages_X-1;
      
      int t;
      
      for(t=0; t<inputImages_C; t++)
        {
	  real gradOutValue = gradOutput_data[gradOutputAddress + t];
	  if(lowIsIn)
	    {
              real inLow = inputImages_data[inLowAddress + t];
              lowDotProduct += inLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowAddress + t] += xWeightLow * gradOutValue;
	    }
	  
           if(highIsIn)
           {
              real inHigh = inputImages_data[inHighAddress + t];
              highDotProduct += inHigh * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesHighAddress + t] += (1-xWeightLow) * gradOutValue;
           }

        }

      xf = - lowDotProduct + highDotProduct; // CHECK: CORRECT?

      gradGrids_data[b*gradGrids_strideBatch + xOut*gradGrids_stride_X] = xf * (inputImages_X-1) / 2;
      
    }
  }

  return 1;
}

int BilinearSamplerBCX_updateGradInput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput )
{
  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[2];
  int gradOutput_X = gradOutput->size[2];
  int inputImages_C = inputImages->size[1];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_stride_X = gradOutput->stride[2];
  int gradOutput_stride_C = gradOutput->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[2];
  int inputImages_stride_C = inputImages->stride[1];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_stride_X = gradInputImages->stride[2];
  int gradInputImages_stride_C = gradInputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[2];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_stride_X = gradGrids->stride[2];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < gradOutput_X; xOut++)
    {
      //read the grid
      real xf = grids_data[b*grids_strideBatch + xOut*grids_stride_X];

      // get the weights for interpolation
      int xInLow;
      real xWeightLow;

      real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
      xInLow = floor(xcoord);
      xWeightLow = 1 - (xcoord - xInLow);

      const int inLowAddress = inputImages_strideBatch * b + inputImages_stride_X * xInLow;
      const int inHighAddress = inLowAddress + inputImages_stride_X;

      const int gradInputImagesLowAddress = gradInputImages_strideBatch * b + gradInputImages_stride_X * xInLow;
      const int gradInputImagesHighAddress = gradInputImagesLowAddress + gradInputImages_stride_X;
      
      const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_stride_X * xOut;
      
      real lowDotProduct = 0;
      real highDotProduct = 0;
      
      real v=0;
      real inLow=0;
      real inHigh=0;
      
      // we are careful with the boundaries
      bool lowIsIn = xInLow >= 0 && xInLow <= inputImages_X-1;
      bool highIsIn = xInLow+1 >= 0 && xInLow+1 <= inputImages_X-1;
      
      int t;
      
      for(t=0; t<inputImages_C; t++)
        {
	  real gradOutValue = gradOutput_data[gradOutputAddress + t*gradOutput_stride_C];
	  if(lowIsIn)
	    {
              real inLow = inputImages_data[inLowAddress + t*inputImages_stride_C];
              lowDotProduct += inLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowAddress + t*gradInputImages_stride_C] += xWeightLow * gradOutValue;
	    }
	  
           if(highIsIn)
           {
              real inHigh = inputImages_data[inHighAddress + t*inputImages_stride_C];
              highDotProduct += inHigh * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesHighAddress + t*gradInputImages_stride_C] += (1-xWeightLow) * gradOutValue;
           }

        }

      xf = - lowDotProduct + highDotProduct; // CHECK: CORRECT?

      gradGrids_data[b*gradGrids_strideBatch + xOut*gradGrids_stride_X] = xf * (inputImages_X-1) / 2;
      
    }
  }

  return 1;
}

int BilinearSamplerBXYC_updateGradInput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput )
{
  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[1];
  int inputImages_Y = inputImages->size[2];
  int gradOutput_X = gradOutput->size[1];
  int gradOutput_Y = gradOutput->size[2];
  int inputImages_C = inputImages->size[3];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_stride_X = gradOutput->stride[1];
  int gradOutput_stride_Y = gradOutput->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[1];
  int inputImages_stride_Y = inputImages->stride[2];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_stride_X = gradInputImages->stride[1];
  int gradInputImages_stride_Y = gradInputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[1];
  int grids_stride_Y = grids->stride[2];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_stride_X = gradGrids->stride[1];
  int gradGrids_stride_Y = gradGrids->stride[2];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < gradOutput_X; xOut++)
    {
      for(yOut=0; yOut < gradOutput_Y; yOut++)
      {
        //read the grid
        real xf = grids_data[b*grids_strideBatch + yOut*grids_stride_Y + xOut*grids_stride_X];
        real yf = grids_data[b*grids_strideBatch + yOut*grids_stride_Y + xOut*grids_stride_X + 1];


        // get the weights for interpolation
        int yInLowLow, xInLowLow;
        real yWeightLowLow, xWeightLowLow;

        real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
        xInLowLow = floor(xcoord);
        xWeightLowLow = 1 - (xcoord - xInLowLow);

        real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
        yInLowLow = floor(ycoord);
        yWeightLowLow = 1 - (ycoord - yInLowLow);


        const int inLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Y * yInLowLow + inputImages_stride_X * xInLowLow;
        const int inLowHighAddress = inLowLowAddress + inputImages_stride_Y;
        const int inHighLowAddress = inLowLowAddress + inputImages_stride_X;
        const int inHighHighAddress = inHighLowAddress + inputImages_stride_Y;

        const int gradInputImagesLowLowAddress = gradInputImages_strideBatch * b + gradInputImages_stride_Y * yInLowLow + gradInputImages_stride_X * xInLowLow;
        const int gradInputImagesLowHighAddress = gradInputImagesLowLowAddress + gradInputImages_stride_Y;
        const int gradInputImagesHighLowAddress = gradInputImagesLowLowAddress + gradInputImages_stride_X;
        const int gradInputImagesHighHighAddress = gradInputImagesHighLowAddress + gradInputImages_stride_Y;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_stride_Y * yOut + gradOutput_stride_X * xOut;

        real lowLowDotProduct = 0;
        real lowHighDotProduct = 0;
        real highLowDotProduct = 0;
        real highHighDotProduct = 0;

        real v=0;
        real inLowLow=0;
        real inLowHigh=0;
        real inHighLow=0;
        real inHighHigh=0;

        // we are careful with the boundaries
        bool lowLowIsIn = xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;
        bool lowHighIsIn = xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow+1 >= 0 && yInLowLow+1 <= inputImages_Y-1;
        bool highLowIsIn = xInLowLow+1 >= 0 && xInLowLow+1 <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;
        bool highHighIsIn = xInLowLow+1 >= 0 && xInLowLow+1 <= inputImages_X-1 && yInLowLow+1 >= 0 && yInLowLow+1 <= inputImages_Y-1;

        int t;

        for(t=0; t<inputImages_C; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(lowLowIsIn)
           {
              real inLowLow = inputImages_data[inLowLowAddress + t];
              lowLowDotProduct += inLowLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowAddress + t] += xWeightLowLow * yWeightLowLow * gradOutValue;
           }

           if(lowHighIsIn)
           {
              real inLowHigh = inputImages_data[inLowHighAddress + t];
              lowHighDotProduct += inLowHigh * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighAddress + t] += xWeightLowLow * (1-yWeightLowLow) * gradOutValue; // CHECK: CORRECT?
           }

           if(highLowIsIn)
           {
              real inHighLow = inputImages_data[inHighLowAddress + t];
              highLowDotProduct += inHighLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowAddress + t] += (1-xWeightLowLow) * yWeightLowLow * gradOutValue; // CHECK: CORRECT?
           }

           if(highHighIsIn)
           {
              real inHighHigh = inputImages_data[inHighHighAddress + t];
              highHighDotProduct += inHighHigh * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesHighHighAddress + t] += (1 - xWeightLowLow) * (1 - yWeightLowLow) * gradOutValue;
           }
        }

	xf = - yWeightLowLow * lowLowDotProduct + yWeightLowLow * highLowDotProduct
	  - (1-yWeightLowLow) * lowHighDotProduct + (1-yWeightLowLow) * highHighDotProduct;

        yf = - xWeightLowLow * lowLowDotProduct + xWeightLowLow * lowHighDotProduct
	  - (1-xWeightLowLow) * highLowDotProduct + (1-xWeightLowLow) * highHighDotProduct;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X] = xf * (inputImages_X-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X + 1] = yf * (inputImages_Y-1) / 2;

      }
    }
  }

  return 1;
}

int BilinearSamplerBCXY_updateGradInput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput )
{
  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[2];
  int inputImages_Y = inputImages->size[3];
  int gradOutput_X = gradOutput->size[2];
  int gradOutput_Y = gradOutput->size[3];
  int inputImages_C = inputImages->size[1];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_stride_X = gradOutput->stride[2];
  int gradOutput_stride_Y = gradOutput->stride[3];
  int gradOutput_stride_C = gradOutput->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[2];
  int inputImages_stride_Y = inputImages->stride[3];
  int inputImages_stride_C = inputImages->stride[1];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_stride_X = gradInputImages->stride[2];
  int gradInputImages_stride_Y = gradInputImages->stride[3];
  int gradInputImages_stride_C = gradInputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[2];
  int grids_stride_Y = grids->stride[3];
  int grids_stride_C = grids->stride[1];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_stride_X = gradGrids->stride[2];
  int gradGrids_stride_Y = gradGrids->stride[3];
  int gradGrids_stride_C = gradGrids->stride[1];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < gradOutput_X; xOut++)
    {
      for(yOut=0; yOut < gradOutput_Y; yOut++)
      {
        //read the grid
        real xf = grids_data[b*grids_strideBatch + yOut*grids_stride_Y + xOut*grids_stride_X];
        real yf = grids_data[b*grids_strideBatch + yOut*grids_stride_Y + xOut*grids_stride_X + grids_stride_C];


        // get the weights for interpolation
        int yInLowLow, xInLowLow;
        real yWeightLowLow, xWeightLowLow;

        real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
        xInLowLow = floor(xcoord);
        xWeightLowLow = 1 - (xcoord - xInLowLow);

        real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
        yInLowLow = floor(ycoord);
        yWeightLowLow = 1 - (ycoord - yInLowLow);


        const int inLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Y * yInLowLow + inputImages_stride_X * xInLowLow;
        const int inLowHighAddress = inLowLowAddress + inputImages_stride_Y;
        const int inHighLowAddress = inLowLowAddress + inputImages_stride_X;
        const int inHighHighAddress = inHighLowAddress + inputImages_stride_Y;

        const int gradInputImagesLowLowAddress = gradInputImages_strideBatch * b + gradInputImages_stride_Y * yInLowLow + gradInputImages_stride_X * xInLowLow;
        const int gradInputImagesLowHighAddress = gradInputImagesLowLowAddress + gradInputImages_stride_Y;
        const int gradInputImagesHighLowAddress = gradInputImagesLowLowAddress + gradInputImages_stride_X;
        const int gradInputImagesHighHighAddress = gradInputImagesHighLowAddress + gradInputImages_stride_Y;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_stride_Y * yOut + gradOutput_stride_X * xOut;

        real lowLowDotProduct = 0;
        real lowHighDotProduct = 0;
        real highLowDotProduct = 0;
        real highHighDotProduct = 0;

        real v=0;
        real inLowLow=0;
        real inLowHigh=0;
        real inHighLow=0;
        real inHighHigh=0;

        // we are careful with the boundaries
        bool lowLowIsIn = xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;
        bool lowHighIsIn = xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow+1 >= 0 && yInLowLow+1 <= inputImages_Y-1;
        bool highLowIsIn = xInLowLow+1 >= 0 && xInLowLow+1 <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;
        bool highHighIsIn = xInLowLow+1 >= 0 && xInLowLow+1 <= inputImages_X-1 && yInLowLow+1 >= 0 && yInLowLow+1 <= inputImages_Y-1;

        int t;

        for(t=0; t<inputImages_C; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t*gradOutput_stride_C];
           if(lowLowIsIn)
           {
              real inLowLow = inputImages_data[inLowLowAddress + t*inputImages_stride_C];
              lowLowDotProduct += inLowLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowAddress + t*gradInputImages_stride_C] += xWeightLowLow * yWeightLowLow * gradOutValue;
           }

           if(lowHighIsIn)
           {
              real inLowHigh = inputImages_data[inLowHighAddress + t*inputImages_stride_C];
              lowHighDotProduct += inLowHigh * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighAddress + t*gradInputImages_stride_C] += xWeightLowLow * (1-yWeightLowLow) * gradOutValue; // CHECK: CORRECT?
           }

           if(highLowIsIn)
           {
              real inHighLow = inputImages_data[inHighLowAddress + t*inputImages_stride_C];
              highLowDotProduct += inHighLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowAddress + t*gradInputImages_stride_C] += (1-xWeightLowLow) * yWeightLowLow * gradOutValue; // CHECK: CORRECT?
           }

           if(highHighIsIn)
           {
              real inHighHigh = inputImages_data[inHighHighAddress + t*inputImages_stride_C];
              highHighDotProduct += inHighHigh * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesHighHighAddress + t*gradInputImages_stride_C] += (1 - xWeightLowLow) * (1 - yWeightLowLow) * gradOutValue;
           }
        }

	xf = - yWeightLowLow * lowLowDotProduct + yWeightLowLow * highLowDotProduct
	  - (1-yWeightLowLow) * lowHighDotProduct + (1-yWeightLowLow) * highHighDotProduct;

        yf = - xWeightLowLow * lowLowDotProduct + xWeightLowLow * lowHighDotProduct
	  - (1-xWeightLowLow) * highLowDotProduct + (1-xWeightLowLow) * highHighDotProduct;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X] = xf * (inputImages_X-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X + gradGrids_stride_C] = yf * (inputImages_Y-1) / 2;

      }
    }
  }

  return 1;
}

int BilinearSamplerBXYZC_updateGradInput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput )
{
  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[1];
  int inputImages_Y = inputImages->size[2];
  int inputImages_Z = inputImages->size[3];
  int gradOutput_X = gradOutput->size[1];
  int gradOutput_Y = gradOutput->size[2];
  int gradOutput_Z = gradOutput->size[3];
  int inputImages_C = inputImages->size[4];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_stride_X = gradOutput->stride[1];
  int gradOutput_stride_Y = gradOutput->stride[2];
  int gradOutput_stride_Z = gradOutput->stride[3];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[1];
  int inputImages_stride_Y = inputImages->stride[2];
  int inputImages_stride_Z = inputImages->stride[3];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_stride_X = gradInputImages->stride[1];
  int gradInputImages_stride_Y = gradInputImages->stride[2];
  int gradInputImages_stride_Z = gradInputImages->stride[3];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[1];
  int grids_stride_Y = grids->stride[2];
  int grids_stride_Z = grids->stride[3];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_stride_X = gradGrids->stride[1];
  int gradGrids_stride_Y = gradGrids->stride[2];
  int gradGrids_stride_Z = gradGrids->stride[3];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, zOut, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < gradOutput_X; xOut++)
    {
      for(yOut=0; yOut < gradOutput_Y; yOut++)
      {
	for(zOut=0; zOut < gradOutput_Z; zOut++)
	  {
	    //read the grid
	    real xf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X];
	    real yf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X + 1];
	    real zf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X + 2];

	    // get the weights for interpolation
	    int zInLowLowLow, yInLowLowLow, xInLowLowLow;
	    real zWeightLowLowLow, yWeightLowLowLow, xWeightLowLowLow;
	    
	    real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
	    xInLowLowLow = floor(xcoord);
	    xWeightLowLowLow = 1 - (xcoord - xInLowLowLow);
	    
	    real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
	    yInLowLowLow = floor(ycoord);
	    yWeightLowLowLow = 1 - (ycoord - yInLowLowLow);

	    real zcoord = (zf + 1) * (inputImages_Z - 1) / 2;
	    zInLowLowLow = floor(zcoord);
	    zWeightLowLowLow = 1 - (zcoord - zInLowLowLow);
	    
	    const int inLowLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Z * zInLowLowLow +
	      inputImages_stride_Y * yInLowLowLow + inputImages_stride_X * xInLowLowLow;
	    const int inLowLowHighAddress = inLowLowLowAddress + inputImages_stride_Z;
	    const int inLowHighLowAddress = inLowLowLowAddress + inputImages_stride_Y;
	    const int inLowHighHighAddress = inLowLowLowAddress + inputImages_stride_Y + inputImages_stride_Z;
	    const int inHighLowLowAddress = inLowLowLowAddress + inputImages_stride_X;
	    const int inHighLowHighAddress = inHighLowLowAddress + inputImages_stride_Z;
	    const int inHighHighLowAddress = inHighLowLowAddress + inputImages_stride_Y;
	    const int inHighHighHighAddress = inHighLowLowAddress + inputImages_stride_Y + inputImages_stride_Z;
	    
	    const int gradInputImagesLowLowLowAddress = gradInputImages_strideBatch * b + gradInputImages_stride_Z * zInLowLowLow +
	      gradInputImages_stride_Y * yInLowLowLow + gradInputImages_stride_X * xInLowLowLow;
	    const int gradInputImagesLowLowHighAddress = gradInputImagesLowLowLowAddress + gradInputImages_stride_Z;
	    const int gradInputImagesLowHighLowAddress = gradInputImagesLowLowLowAddress + gradInputImages_stride_Y;
	    const int gradInputImagesLowHighHighAddress = gradInputImagesLowLowLowAddress + gradInputImages_stride_Y + gradInputImages_stride_Z;
	    const int gradInputImagesHighLowLowAddress = gradInputImagesLowLowLowAddress + gradInputImages_stride_X;
	    const int gradInputImagesHighLowHighAddress = gradInputImagesHighLowLowAddress + gradInputImages_stride_Z;
	    const int gradInputImagesHighHighLowAddress = gradInputImagesHighLowLowAddress + gradInputImages_stride_Y;
	    const int gradInputImagesHighHighHighAddress = gradInputImagesHighLowLowAddress + gradInputImages_stride_Y + gradInputImages_stride_Z;
	    
	    const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_stride_Z * zOut + gradOutput_stride_Y * yOut + gradOutput_stride_X * xOut;
	    
	    real lowLowLowDotProduct = 0;
	    real lowLowHighDotProduct = 0;
	    real lowHighLowDotProduct = 0;
	    real lowHighHighDotProduct = 0;
	    real highLowLowDotProduct = 0;
	    real highLowHighDotProduct = 0;
	    real highHighLowDotProduct = 0;
	    real highHighHighDotProduct = 0;
	    
	    real v=0;
	    real inLowLowLow=0;
	    real inLowLowHigh=0;
	    real inLowHighLow=0;
	    real inLowHighHigh=0;
	    real inHighLowLow=0;
	    real inHighLowHigh=0;
	    real inHighHighLow=0;
	    real inHighHighHigh=0;

	    // we are careful with the boundaries
	    bool lowLowLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
	      && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
	      && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
	    bool lowLowHighIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
	      && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
	      && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
	    bool lowHighLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
	      && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
	      && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
	    bool lowHighHighIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
	      && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
	      && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
	    bool highLowLowIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
	      && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
	      && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
	    bool highLowHighIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
	      && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
	      && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
	    bool highHighLowIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
	      && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
	      && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
	    bool highHighHighIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
	      && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
	      && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
	    	    
	    int t;
	    
	    for(t=0; t<inputImages_C; t++)
	      {
		real gradOutValue = gradOutput_data[gradOutputAddress + t];
		if(lowLowLowIsIn)
		  {
		    real inLowLowLow = inputImages_data[inLowLowLowAddress + t];
		    lowLowLowDotProduct += inLowLowLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowLowAddress + t] += xWeightLowLowLow * yWeightLowLowLow * zWeightLowLowLow * gradOutValue;
		  }
		
		if(lowLowHighIsIn)
		  {
		    real inLowLowHigh = inputImages_data[inLowLowHighAddress + t];
		    lowLowHighDotProduct += inLowLowHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowHighAddress + t] += xWeightLowLowLow * yWeightLowLowLow * (1-zWeightLowLowLow) * gradOutValue; // CHECK: CORRECT?
		  }
		
		if(lowHighLowIsIn)
		  {
		    real inLowHighLow = inputImages_data[inLowHighLowAddress + t];
		    lowHighLowDotProduct += inLowHighLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighLowAddress + t] += xWeightLowLowLow * (1-yWeightLowLowLow) * zWeightLowLowLow * gradOutValue; // CHECK: CORRECT?
		  }
		
		if(lowHighHighIsIn)
		  {
		    real inLowHighHigh = inputImages_data[inLowHighHighAddress + t];
		    lowHighHighDotProduct += inLowHighHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighHighAddress + t] += xWeightLowLowLow * (1 - yWeightLowLowLow) * (1-zWeightLowLowLow) * gradOutValue;
		  }

		if(highLowLowIsIn)
		  {
		    real inHighLowLow = inputImages_data[inHighLowLowAddress + t];
		    highLowLowDotProduct += inHighLowLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowLowAddress + t] += (1-xWeightLowLowLow) * yWeightLowLowLow * zWeightLowLowLow * gradOutValue;
		  }
		
		if(highLowHighIsIn)
		  {
		    real inHighLowHigh = inputImages_data[inHighLowHighAddress + t];
		    highLowHighDotProduct += inHighLowHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowHighAddress + t] += (1-xWeightLowLowLow) * yWeightLowLowLow * (1-zWeightLowLowLow) * gradOutValue; // CHECK: CORRECT?
		  }
		
		if(highHighLowIsIn)
		  {
		    real inHighHighLow = inputImages_data[inHighHighLowAddress + t];
		    highHighLowDotProduct += inHighHighLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighHighLowAddress + t] += (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * zWeightLowLowLow * gradOutValue; // CHECK: CORRECT?
		  }
		
		if(highHighHighIsIn)
		  {
		    real inHighHighHigh = inputImages_data[inHighHighHighAddress + t];
		    highHighHighDotProduct += inHighHighHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighHighHighAddress + t] += (1-xWeightLowLowLow) * (1 - yWeightLowLowLow) * (1-zWeightLowLowLow) * gradOutValue;
		  }
	      }


	    // CHECK: CORRECT?

	    xf = - yWeightLowLowLow * zWeightLowLowLow * lowLowLowDotProduct + yWeightLowLowLow * zWeightLowLowLow * highLowLowDotProduct
	      - yWeightLowLowLow * (1-zWeightLowLowLow) * lowLowHighDotProduct + yWeightLowLowLow * (1-zWeightLowLowLow) * highLowHighDotProduct
	      - (1-yWeightLowLowLow) * zWeightLowLowLow * lowHighLowDotProduct + (1-yWeightLowLowLow) * zWeightLowLowLow * highHighLowDotProduct
	      - (1-yWeightLowLowLow) * (1-zWeightLowLowLow) * lowHighHighDotProduct + (1-yWeightLowLowLow) * (1-zWeightLowLowLow) * highHighHighDotProduct;
	    
	    yf = - xWeightLowLowLow * zWeightLowLowLow * lowLowLowDotProduct + xWeightLowLowLow * zWeightLowLowLow * lowHighLowDotProduct
	      - xWeightLowLowLow * (1-zWeightLowLowLow) * lowLowHighDotProduct + xWeightLowLowLow * (1-zWeightLowLowLow) * lowHighHighDotProduct
	      - (1-xWeightLowLowLow) * zWeightLowLowLow * highLowLowDotProduct + (1-xWeightLowLowLow) * zWeightLowLowLow * highHighLowDotProduct
	      - (1-xWeightLowLowLow) * (1-zWeightLowLowLow) * highLowHighDotProduct + (1-xWeightLowLowLow) * (1-zWeightLowLowLow) * highHighHighDotProduct;

	    zf = - xWeightLowLowLow * yWeightLowLowLow * lowLowLowDotProduct + xWeightLowLowLow * yWeightLowLowLow * lowLowHighDotProduct
	      - (1-xWeightLowLowLow) * yWeightLowLowLow * highLowLowDotProduct + (1-xWeightLowLowLow) * yWeightLowLowLow * highLowHighDotProduct
	      - xWeightLowLowLow * (1-yWeightLowLowLow) * lowHighLowDotProduct + xWeightLowLowLow * (1-yWeightLowLowLow) * lowHighHighDotProduct
	      - (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * highHighLowDotProduct + (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * highHighHighDotProduct;
	      
	    gradGrids_data[b*gradGrids_strideBatch + zOut*gradGrids_stride_Z + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X] = xf * (inputImages_X-1) / 2;
	    gradGrids_data[b*gradGrids_strideBatch + zOut*gradGrids_stride_Z + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X + 1] = yf * (inputImages_Y-1) / 2;
	    gradGrids_data[b*gradGrids_strideBatch + zOut*gradGrids_stride_Z + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X + 2] = zf * (inputImages_Z-1) / 2;
	    
	  }
      }
    }
  }

  return 1;
}

int BilinearSamplerBCXYZ_updateGradInput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput )
{
  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_X = inputImages->size[2];
  int inputImages_Y = inputImages->size[3];
  int inputImages_Z = inputImages->size[4];
  int gradOutput_X = gradOutput->size[2];
  int gradOutput_Y = gradOutput->size[3];
  int gradOutput_Z = gradOutput->size[4];
  int inputImages_C = inputImages->size[1];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_stride_X = gradOutput->stride[2];
  int gradOutput_stride_Y = gradOutput->stride[3];
  int gradOutput_stride_Z = gradOutput->stride[4];
  int gradOutput_stride_C = gradOutput->stride[1];
  
  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_stride_X = inputImages->stride[2];
  int inputImages_stride_Y = inputImages->stride[3];
  int inputImages_stride_Z = inputImages->stride[4];
  int inputImages_stride_C = inputImages->stride[1];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_stride_X = gradInputImages->stride[2];
  int gradInputImages_stride_Y = gradInputImages->stride[3];
  int gradInputImages_stride_Z = gradInputImages->stride[4];
  int gradInputImages_stride_C = gradInputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_stride_X = grids->stride[2];
  int grids_stride_Y = grids->stride[3];
  int grids_stride_Z = grids->stride[4];
  int grids_stride_C = grids->stride[1];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_stride_X = gradGrids->stride[2];
  int gradGrids_stride_Y = gradGrids->stride[3];
  int gradGrids_stride_Z = gradGrids->stride[4];
  int gradGrids_stride_C = gradGrids->stride[1];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, zOut, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    #pragma omp parallel for
    for(xOut=0; xOut < gradOutput_X; xOut++)
    {
      for(yOut=0; yOut < gradOutput_Y; yOut++)
      {
	for(zOut=0; zOut < gradOutput_Z; zOut++)
	  {
	    //read the grid
	    real xf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X];
	    real yf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X + grids_stride_C];
	    real zf = grids_data[b*grids_strideBatch + zOut*grids_stride_Z + yOut*grids_stride_Y + xOut*grids_stride_X + 2*grids_stride_C];

	    // get the weights for interpolation
	    int zInLowLowLow, yInLowLowLow, xInLowLowLow;
	    real zWeightLowLowLow, yWeightLowLowLow, xWeightLowLowLow;
	    
	    real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
	    xInLowLowLow = floor(xcoord);
	    xWeightLowLowLow = 1 - (xcoord - xInLowLowLow);
	    
	    real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
	    yInLowLowLow = floor(ycoord);
	    yWeightLowLowLow = 1 - (ycoord - yInLowLowLow);

	    real zcoord = (zf + 1) * (inputImages_Z - 1) / 2;
	    zInLowLowLow = floor(zcoord);
	    zWeightLowLowLow = 1 - (zcoord - zInLowLowLow);
	    
	    const int inLowLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Z * zInLowLowLow +
	      inputImages_stride_Y * yInLowLowLow + inputImages_stride_X * xInLowLowLow;
	    const int inLowLowHighAddress = inLowLowLowAddress + inputImages_stride_Z;
	    const int inLowHighLowAddress = inLowLowLowAddress + inputImages_stride_Y;
	    const int inLowHighHighAddress = inLowLowLowAddress + inputImages_stride_Y + inputImages_stride_Z;
	    const int inHighLowLowAddress = inLowLowLowAddress + inputImages_stride_X;
	    const int inHighLowHighAddress = inHighLowLowAddress + inputImages_stride_Z;
	    const int inHighHighLowAddress = inHighLowLowAddress + inputImages_stride_Y;
	    const int inHighHighHighAddress = inHighLowLowAddress + inputImages_stride_Y + inputImages_stride_Z;
	    
	    const int gradInputImagesLowLowLowAddress = gradInputImages_strideBatch * b + gradInputImages_stride_Z * zInLowLowLow +
	      gradInputImages_stride_Y * yInLowLowLow + gradInputImages_stride_X * xInLowLowLow;
	    const int gradInputImagesLowLowHighAddress = gradInputImagesLowLowLowAddress + gradInputImages_stride_Z;
	    const int gradInputImagesLowHighLowAddress = gradInputImagesLowLowLowAddress + gradInputImages_stride_Y;
	    const int gradInputImagesLowHighHighAddress = gradInputImagesLowLowLowAddress + gradInputImages_stride_Y + gradInputImages_stride_Z;
	    const int gradInputImagesHighLowLowAddress = gradInputImagesLowLowLowAddress + gradInputImages_stride_X;
	    const int gradInputImagesHighLowHighAddress = gradInputImagesHighLowLowAddress + gradInputImages_stride_Z;
	    const int gradInputImagesHighHighLowAddress = gradInputImagesHighLowLowAddress + gradInputImages_stride_Y;
	    const int gradInputImagesHighHighHighAddress = gradInputImagesHighLowLowAddress + gradInputImages_stride_Y + gradInputImages_stride_Z;
	    
	    const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_stride_Z * zOut + gradOutput_stride_Y * yOut + gradOutput_stride_X * xOut;
	    
	    real lowLowLowDotProduct = 0;
	    real lowLowHighDotProduct = 0;
	    real lowHighLowDotProduct = 0;
	    real lowHighHighDotProduct = 0;
	    real highLowLowDotProduct = 0;
	    real highLowHighDotProduct = 0;
	    real highHighLowDotProduct = 0;
	    real highHighHighDotProduct = 0;
	    
	    real v=0;
	    real inLowLowLow=0;
	    real inLowLowHigh=0;
	    real inLowHighLow=0;
	    real inLowHighHigh=0;
	    real inHighLowLow=0;
	    real inHighLowHigh=0;
	    real inHighHighLow=0;
	    real inHighHighHigh=0;

	    // we are careful with the boundaries
	    bool lowLowLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
	      && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
	      && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
	    bool lowLowHighIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
	      && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
	      && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
	    bool lowHighLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
	      && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
	      && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
	    bool lowHighHighIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
	      && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
	      && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
	    bool highLowLowIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
	      && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
	      && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
	    bool highLowHighIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
	      && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
	      && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
	    bool highHighLowIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
	      && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
	      && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;
	    bool highHighHighIsIn = xInLowLowLow+1 >= 0 && xInLowLowLow+1 <= inputImages_X-1
	      && yInLowLowLow+1 >= 0 && yInLowLowLow+1 <= inputImages_Y-1
	      && zInLowLowLow+1 >= 0 && zInLowLowLow+1 <= inputImages_Z-1;
	    	    
	    int t;
	    
	    for(t=0; t<inputImages_C; t++)
	      {
		real gradOutValue = gradOutput_data[gradOutputAddress + t*gradOutput_stride_C];
		if(lowLowLowIsIn)
		  {
		    real inLowLowLow = inputImages_data[inLowLowLowAddress + t*inputImages_stride_C];
		    lowLowLowDotProduct += inLowLowLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowLowAddress + t*gradInputImages_stride_C] += xWeightLowLowLow * yWeightLowLowLow * zWeightLowLowLow * gradOutValue;
		  }
		
		if(lowLowHighIsIn)
		  {
		    real inLowLowHigh = inputImages_data[inLowLowHighAddress + t*inputImages_stride_C];
		    lowLowHighDotProduct += inLowLowHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowHighAddress + t*gradInputImages_stride_C] += xWeightLowLowLow * yWeightLowLowLow * (1-zWeightLowLowLow) * gradOutValue; // CHECK: CORRECT?
		  }
		
		if(lowHighLowIsIn)
		  {
		    real inLowHighLow = inputImages_data[inLowHighLowAddress + t*inputImages_stride_C];
		    lowHighLowDotProduct += inLowHighLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighLowAddress + t*gradInputImages_stride_C] += xWeightLowLowLow * (1-yWeightLowLowLow) * zWeightLowLowLow * gradOutValue; // CHECK: CORRECT?
		  }
		
		if(lowHighHighIsIn)
		  {
		    real inLowHighHigh = inputImages_data[inLowHighHighAddress + t*inputImages_stride_C];
		    lowHighHighDotProduct += inLowHighHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighHighAddress + t*gradInputImages_stride_C] += xWeightLowLowLow * (1 - yWeightLowLowLow) * (1-zWeightLowLowLow) * gradOutValue;
		  }

		if(highLowLowIsIn)
		  {
		    real inHighLowLow = inputImages_data[inHighLowLowAddress + t*inputImages_stride_C];
		    highLowLowDotProduct += inHighLowLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowLowAddress + t*gradInputImages_stride_C] += (1-xWeightLowLowLow) * yWeightLowLowLow * zWeightLowLowLow * gradOutValue;
		  }
		
		if(highLowHighIsIn)
		  {
		    real inHighLowHigh = inputImages_data[inHighLowHighAddress + t*inputImages_stride_C];
		    highLowHighDotProduct += inHighLowHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowHighAddress + t*gradInputImages_stride_C] += (1-xWeightLowLowLow) * yWeightLowLowLow * (1-zWeightLowLowLow) * gradOutValue; // CHECK: CORRECT?
		  }
		
		if(highHighLowIsIn)
		  {
		    real inHighHighLow = inputImages_data[inHighHighLowAddress + t*inputImages_stride_C];
		    highHighLowDotProduct += inHighHighLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighHighLowAddress + t*gradInputImages_stride_C] += (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * zWeightLowLowLow * gradOutValue; // CHECK: CORRECT?
		  }
		
		if(highHighHighIsIn)
		  {
		    real inHighHighHigh = inputImages_data[inHighHighHighAddress + t*inputImages_stride_C];
		    highHighHighDotProduct += inHighHighHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighHighHighAddress + t*gradInputImages_stride_C] += (1-xWeightLowLowLow) * (1 - yWeightLowLowLow) * (1-zWeightLowLowLow) * gradOutValue;
		  }
	      }


	    // CHECK: CORRECT?

	    xf = - yWeightLowLowLow * zWeightLowLowLow * lowLowLowDotProduct + yWeightLowLowLow * zWeightLowLowLow * highLowLowDotProduct
	      - yWeightLowLowLow * (1-zWeightLowLowLow) * lowLowHighDotProduct + yWeightLowLowLow * (1-zWeightLowLowLow) * highLowHighDotProduct
	      - (1-yWeightLowLowLow) * zWeightLowLowLow * lowHighLowDotProduct + (1-yWeightLowLowLow) * zWeightLowLowLow * highHighLowDotProduct
	      - (1-yWeightLowLowLow) * (1-zWeightLowLowLow) * lowHighHighDotProduct + (1-yWeightLowLowLow) * (1-zWeightLowLowLow) * highHighHighDotProduct;
	    
	    yf = - xWeightLowLowLow * zWeightLowLowLow * lowLowLowDotProduct + xWeightLowLowLow * zWeightLowLowLow * lowHighLowDotProduct
	      - xWeightLowLowLow * (1-zWeightLowLowLow) * lowLowHighDotProduct + xWeightLowLowLow * (1-zWeightLowLowLow) * lowHighHighDotProduct
	      - (1-xWeightLowLowLow) * zWeightLowLowLow * highLowLowDotProduct + (1-xWeightLowLowLow) * zWeightLowLowLow * highHighLowDotProduct
	      - (1-xWeightLowLowLow) * (1-zWeightLowLowLow) * highLowHighDotProduct + (1-xWeightLowLowLow) * (1-zWeightLowLowLow) * highHighHighDotProduct;

	    zf = - xWeightLowLowLow * yWeightLowLowLow * lowLowLowDotProduct + xWeightLowLowLow * yWeightLowLowLow * lowLowHighDotProduct
	      - (1-xWeightLowLowLow) * yWeightLowLowLow * highLowLowDotProduct + (1-xWeightLowLowLow) * yWeightLowLowLow * highLowHighDotProduct
	      - xWeightLowLowLow * (1-yWeightLowLowLow) * lowHighLowDotProduct + xWeightLowLowLow * (1-yWeightLowLowLow) * lowHighHighDotProduct
	      - (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * highHighLowDotProduct + (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * highHighHighDotProduct;
	      
	    gradGrids_data[b*gradGrids_strideBatch + zOut*gradGrids_stride_Z + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X] = xf * (inputImages_X-1) / 2;
	    gradGrids_data[b*gradGrids_strideBatch + zOut*gradGrids_stride_Z + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X + gradGrids_stride_C] = yf * (inputImages_Y-1) / 2;
	    gradGrids_data[b*gradGrids_strideBatch + zOut*gradGrids_stride_Z + yOut*gradGrids_stride_Y + xOut*gradGrids_stride_X + 2*gradGrids_stride_C] = zf * (inputImages_Z-1) / 2;
	    
	  }
      }
    }
  }

  return 1;
}


int BilinearSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput )
{
  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[1];
  int inputImages_width = inputImages->size[2];
  int gradOutput_height = gradOutput->size[1];
  int gradOutput_width = gradOutput->size[2];
  int inputImages_channels = inputImages->size[3];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_strideHeight = gradOutput->stride[1];
  int gradOutput_strideWidth = gradOutput->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideHeight = gradInputImages->stride[1];
  int gradInputImages_strideWidth = gradInputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideHeight = gradGrids->stride[1];
  int gradGrids_strideWidth = gradGrids->stride[2];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
        //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;

        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);


        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
        const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

        real topLeftDotProduct = 0;
        real topRightDotProduct = 0;
        real bottomLeftDotProduct = 0;
        real bottomRightDotProduct = 0;

        real v=0;
        real inTopLeft=0;
        real inTopRight=0;
        real inBottomLeft=0;
        real inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;

        for(t=0; t<inputImages_channels; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(topLeftIsIn)
           {
              real inTopLeft = inputImages_data[inTopLeftAddress + t];
              topLeftDotProduct += inTopLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftAddress + t] += xWeightTopLeft * yWeightTopLeft * gradOutValue;
           }

           if(topRightIsIn)
           {
              real inTopRight = inputImages_data[inTopRightAddress + t];
              topRightDotProduct += inTopRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightAddress + t] += (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue;
           }

           if(bottomLeftIsIn)
           {
              real inBottomLeft = inputImages_data[inBottomLeftAddress + t];
              bottomLeftDotProduct += inBottomLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftAddress + t] += xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue;
           }

           if(bottomRightIsIn)
           {
              real inBottomRight = inputImages_data[inBottomRightAddress + t];
              bottomRightDotProduct += inBottomRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightAddress + t] += (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue;
           }
        }

        yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
        xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = yf * (inputImages_height-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 1] = xf * (inputImages_width-1) / 2;

      }
    }
  }

  return 1;
}


int BilinearSamplerBXYZC_updateGradInput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int ndim)
{
  switch( ndim )
    {
    case 1: return BilinearSamplerBXC_updateGradInput_1D( inputImages, grids, gradInputImages, gradGrids, gradOutput ); break;
    case 2: return BilinearSamplerBXYC_updateGradInput_2D( inputImages, grids, gradInputImages, gradGrids, gradOutput ); break;
    case 3: return BilinearSamplerBXYZC_updateGradInput_3D( inputImages, grids, gradInputImages, gradGrids, gradOutput ); break;
    default: return -1;
    }
}

int BilinearSamplerBCXYZ_updateGradInput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int ndim, _Bool use_cuda, int ndevice)
{
  if(!use_cuda){
    switch( ndim ){
      case 1: return BilinearSamplerBCX_updateGradInput_1D( inputImages, grids, gradInputImages, gradGrids, gradOutput ); break;
      case 2: return BilinearSamplerBCXY_updateGradInput_2D( inputImages, grids, gradInputImages, gradGrids, gradOutput ); break;
      case 3: return BilinearSamplerBCXYZ_updateGradInput_3D( inputImages, grids, gradInputImages, gradGrids, gradOutput ); break;
      default: return -1;
      }
  }
  else{
    switch( ndim ){
      case 1: return BilinearSamplerBCW_updateGradInput_cuda_1D( inputImages, grids, gradInputImages, gradGrids, gradOutput, ndevice); break;
      case 2: return BilinearSamplerBCWH_updateGradInput_cuda_2D( inputImages, grids, gradInputImages, gradGrids, gradOutput, ndevice ); break;
      case 3: return BilinearSamplerBCWHD_updateGradInput_cuda_3D( inputImages, grids, gradInputImages, gradGrids, gradOutput, ndevice ); break;
      default: return -1;
      }

  }
}




