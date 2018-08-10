#ifndef no_openmp
#include <omp.h>
#endif

//#include <TH/TH.h>
#include <TH/THTensor.h>
#include <stdbool.h>
#include <stdio.h>

#define real float

int BilinearSamplerBXC_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary)
{
  // *B*atch, *X*-coors, *C*hannel
  //inputImages->(size|stride).(.).
  //THTensor_$1(inputImages,$2)
  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,1);
  int inputImages_C = THFloatTensor_size(inputImages,2);
  int output_X = THFloatTensor_size(output,1);
  int output_C = THFloatTensor_size(output,2);
  bool zero_boundary_bool = zero_boundary == 1;

  int output_strideBatch = THFloatTensor_stride(output,0);
  int output_stride_X = THFloatTensor_stride(output,1);
  int output_stride_C = THFloatTensor_stride(output,2);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,1);
  int inputImages_stride_C = THFloatTensor_stride(inputImages,2);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,1);
  int grids_stride_C = THFloatTensor_stride(grids,2);

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

	  bool xBeyondLow = xLow < 0;
      bool xBeyondHigh = xLow+1 > inputImages_X-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xLow = 0;
            if (xBeyondHigh)
                xLow = inputImages_X-2;
         }



	  const int outAddress = output_strideBatch * b + output_stride_X * xOut;
	  const int inLowAddress = inputImages_strideBatch * b + inputImages_stride_X * xLow;
	  const int inHighAddress = inLowAddress + inputImages_stride_X;

	  real v=0;
	  real inLow=0;
	  real inHigh=0;



	  int t;
	  // interpolation happens here
	  for(t=0; t<inputImages_C; t++)
	    {
            if (!zero_boundary_bool || (! (xBeyondLow || xBeyondHigh ))){

              inLow = inputImages_data[inLowAddress + t];
              inHigh = inputImages_data[inHighAddress + t];
             }

	      v = xWeightLow * inLow + (1 - xWeightLow) * inHigh;

	      output_data[outAddress + t] = v;
	    }

	}
    }

  return 1;
}

int BilinearSamplerBCX_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary)
{
  // *B*atch, *C*hannel, *X*-coors
  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,2);
  int inputImages_C = THFloatTensor_size(inputImages,1);
  int output_X = THFloatTensor_size(output,2);
  int output_C = THFloatTensor_size(output,1);
  bool zero_boundary_bool = zero_boundary == 1;

  int output_strideBatch = THFloatTensor_stride(output,0);
  int output_stride_X = THFloatTensor_stride(output,2);
  int output_stride_C = THFloatTensor_stride(output,1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_C = THFloatTensor_stride(inputImages,1);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,2);
  int grids_stride_C = THFloatTensor_stride(grids,1);

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


	  bool xBeyondLow = xLow < 0;
      bool xBeyondHigh = xLow+1 > inputImages_X-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xLow = 0;
            if (xBeyondHigh)
                xLow = inputImages_X-2;
         }


	  const int outAddress = output_strideBatch * b + output_stride_X * xOut;
	  const int inLowAddress = inputImages_strideBatch * b + inputImages_stride_X * xLow;
	  const int inHighAddress = inLowAddress + inputImages_stride_X;

	  real v=0;
	  real inLow=0;
	  real inHigh=0;


	  int t;
	  // interpolation happens here
	  for(t=0; t<inputImages_C; t++)
	    {

	     if (!zero_boundary_bool || (! (xBeyondLow || xBeyondHigh ))){

	      inLow = inputImages_data[inLowAddress + t*inputImages_stride_C];
	      inHigh = inputImages_data[inHighAddress + t*inputImages_stride_C];

	     }

	      v = xWeightLow * inLow + (1 - xWeightLow) * inHigh;

	      output_data[outAddress + t*output_stride_C] = v;
	    }

	}
    }

  return 1;
}

int BilinearSamplerBXYC_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output , int zero_boundary)
{
  // This is actua
  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,1);
  int inputImages_Y = THFloatTensor_size(inputImages,2);
  int inputImages_C = THFloatTensor_size(inputImages,3);
  int output_X = THFloatTensor_size(output,1);
  int output_Y = THFloatTensor_size(output,2);
  bool zero_boundary_bool = zero_boundary == 1;

  int output_strideBatch = THFloatTensor_stride(output,0);
  int output_stride_X = THFloatTensor_stride(output,1);
  int output_stride_Y = THFloatTensor_stride(output,2);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,1);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,2);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,1);
  int grids_stride_Y = THFloatTensor_stride(grids,2);

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

        bool xBeyondLow = xInLowLow < 0;
        bool yBeyondLow = yInLowLow < 0;
        bool xBeyondHigh = xInLowLow+1 > inputImages_X-1;
        bool yBeyondHigh = yInLowLow+1 > inputImages_Y-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLowLow = 0;
            if (xBeyondHigh)
                xInLowLow = inputImages_X-2;
            if (yBeyondLow)
                yInLowLow = 0;
            if (yBeyondHigh)
                yInLowLow = inputImages_Y-2;
         }




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



        int t;
        // interpolation happens here
        for(t=0; t<inputImages_C; t++)
        {
        // if the first is for non zero condition and the second is for zero condition
         if (!zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh))){
           inLowLow = inputImages_data[inLowLowAddress + t];
           inLowHigh = inputImages_data[inLowHighAddress + t];
           inHighLow = inputImages_data[inHighLowAddress + t];
           inHighHigh = inputImages_data[inHighHighAddress + t];
          }

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


int BilinearSamplerBCXY_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary)
{
  // This is actua
  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,2);
  int inputImages_Y = THFloatTensor_size(inputImages,3);
  int inputImages_C = THFloatTensor_size(inputImages,1);
  int output_X = THFloatTensor_size(output,2);
  int output_Y = THFloatTensor_size(output,3);
  bool zero_boundary_bool = zero_boundary == 1;

  int output_strideBatch = THFloatTensor_stride(output,0);
  int output_stride_X = THFloatTensor_stride(output,2);
  int output_stride_Y = THFloatTensor_stride(output,3);
  int output_stride_C = THFloatTensor_stride(output,1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,3);
  int inputImages_stride_C = THFloatTensor_stride(inputImages,1);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_C = THFloatTensor_stride(grids,1);
  int grids_stride_X = THFloatTensor_stride(grids,2);
  int grids_stride_Y = THFloatTensor_stride(grids,3);








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

        bool xBeyondLow = xInLowLow < 0;
        bool yBeyondLow = yInLowLow < 0;
        bool xBeyondHigh = xInLowLow+1 > inputImages_X-1;
        bool yBeyondHigh = yInLowLow+1 > inputImages_Y-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLowLow = 0;
            if (xBeyondHigh)
                xInLowLow = inputImages_X-2;
            if (yBeyondLow)
                yInLowLow = 0;
            if (yBeyondHigh)
                yInLowLow = inputImages_Y-2;
         }



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



//        if (xcoord == 1.0){
//            inLowLowAddress = inHighLowAddress;
//            inLowHighAddress = inHighHighAddress;
//            }
//        if (ycoord == 1.0){
//            inLowLowAddress = inLowHighAddress;
//            inHighLowAddress = inHighHighAddress;
//            }




        int t;
        // interpolation happens here
        for(t=0; t<inputImages_C; t++)
        {
           // if the first is for non zero condition and the second is for zero condition
         if (!zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh))){
           inLowLow = inputImages_data[inLowLowAddress + t*inputImages_stride_C];
           inLowHigh = inputImages_data[inLowHighAddress + t*inputImages_stride_C];
           inHighLow = inputImages_data[inHighLowAddress + t*inputImages_stride_C];
           inHighHigh = inputImages_data[inHighHighAddress + t*inputImages_stride_C];
          }

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




int BilinearSamplerBCXY_updateOutput_2D_old(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary)
{
  // This is actua
  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,2);
  int inputImages_Y = THFloatTensor_size(inputImages,3);
  int inputImages_C = THFloatTensor_size(inputImages,1);
  int output_X = THFloatTensor_size(output,2);
  int output_Y = THFloatTensor_size(output,3);
  bool zero_boundary_bool = zero_boundary == 1;

  int output_strideBatch = THFloatTensor_stride(output,0);
  int output_stride_X = THFloatTensor_stride(output,2);
  int output_stride_Y = THFloatTensor_stride(output,3);
  int output_stride_C = THFloatTensor_stride(output,1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,3);
  int inputImages_stride_C = THFloatTensor_stride(inputImages,1);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_C = THFloatTensor_stride(grids,1);
  int grids_stride_X = THFloatTensor_stride(grids,2);
  int grids_stride_Y = THFloatTensor_stride(grids,3);

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








int BilinearSamplerBXYZC_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary)
{
  // This is actua
  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,1);
  int inputImages_Y = THFloatTensor_size(inputImages,2);
  int inputImages_Z = THFloatTensor_size(inputImages,3);
  int inputImages_C = THFloatTensor_size(inputImages,4);
  int output_X = THFloatTensor_size(output,1);
  int output_Y = THFloatTensor_size(output,2);
  int output_Z = THFloatTensor_size(output,3);
  bool zero_boundary_bool = zero_boundary == 1;

  int output_strideBatch = THFloatTensor_stride(output,0);
  int output_stride_X = THFloatTensor_stride(output,1);
  int output_stride_Y = THFloatTensor_stride(output,2);
  int output_stride_Z = THFloatTensor_stride(output,3);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,1);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_Z = THFloatTensor_stride(inputImages,3);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,1);
  int grids_stride_Y = THFloatTensor_stride(grids,2);
  int grids_stride_Z = THFloatTensor_stride(grids,3);

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




		bool xBeyondLow = xInLowLowLow < 0;
        bool yBeyondLow = yInLowLowLow < 0;
        bool zBeyondLow = zInLowLowLow < 0;
        bool xBeyondHigh = xInLowLowLow+1 > inputImages_X-1;
        bool yBeyondHigh = yInLowLowLow+1 > inputImages_Y-1;
        bool zBeyondHigh = zInLowLowLow+1 > inputImages_Z-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLowLowLow = 0;
            if (xBeyondHigh)
                xInLowLowLow = inputImages_X-2;
            if (yBeyondLow)
                yInLowLowLow = 0;
            if (yBeyondHigh)
                yInLowLowLow = inputImages_Y-2;
            if (zBeyondLow)
                zInLowLowLow = 0;
            if (zBeyondHigh)
                zInLowLowLow = inputImages_Z-2;
         }






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



		int t;
		// interpolation happens here
		for(t=0; t<inputImages_C; t++)
		  {

		  if (!zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh || zBeyondLow || zBeyondHigh))){
            inLowLowLow = inputImages_data[inLowLowLowAddress + t];
		    inLowLowHigh = inputImages_data[inLowLowHighAddress + t];
		    inLowHighLow = inputImages_data[inLowHighLowAddress + t];
		    inLowHighHigh = inputImages_data[inLowHighHighAddress + t];
		    inHighLowLow = inputImages_data[inHighLowLowAddress + t];
		    inHighLowHigh = inputImages_data[inHighLowHighAddress + t];
		    inHighHighLow = inputImages_data[inHighHighLowAddress + t];
		    inHighHighHigh = inputImages_data[inHighHighHighAddress + t];
          }


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

int BilinearSamplerBCXYZ_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary)
{
  // This is actua
  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,2);
  int inputImages_Y = THFloatTensor_size(inputImages,3);
  int inputImages_Z = THFloatTensor_size(inputImages,4);
  int inputImages_C = THFloatTensor_size(inputImages,1);
  int output_X = THFloatTensor_size(output,2);
  int output_Y = THFloatTensor_size(output,3);
  int output_Z = THFloatTensor_size(output,4);
  bool zero_boundary_bool = zero_boundary == 1;

  int output_strideBatch = THFloatTensor_stride(output,0);
  int output_stride_X = THFloatTensor_stride(output,2);
  int output_stride_Y = THFloatTensor_stride(output,3);
  int output_stride_Z = THFloatTensor_stride(output,4);
  int output_stride_C = THFloatTensor_stride(output,1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,3);
  int inputImages_stride_Z = THFloatTensor_stride(inputImages,4);
  int inputImages_stride_C = THFloatTensor_stride(inputImages,1);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,2);
  int grids_stride_Y = THFloatTensor_stride(grids,3);
  int grids_stride_Z = THFloatTensor_stride(grids,4);
  int grids_stride_C = THFloatTensor_stride(grids,1);

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


		bool xBeyondLow = xInLowLowLow < 0;
        bool yBeyondLow = yInLowLowLow < 0;
        bool zBeyondLow = zInLowLowLow < 0;
        bool xBeyondHigh = xInLowLowLow+1 > inputImages_X-1;
        bool yBeyondHigh = yInLowLowLow+1 > inputImages_Y-1;
        bool zBeyondHigh = zInLowLowLow+1 > inputImages_Z-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLowLowLow = 0;
            if (xBeyondHigh)
                xInLowLowLow = inputImages_X-2;
            if (yBeyondLow)
                yInLowLowLow = 0;
            if (yBeyondHigh)
                yInLowLowLow = inputImages_Y-2;
            if (zBeyondLow)
                zInLowLowLow = 0;
            if (zBeyondHigh)
                zInLowLowLow = inputImages_Z-2;
         }


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



		int t;
		// interpolation happens here
		for(t=0; t<inputImages_C; t++)
		  {
		    if (!zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh || zBeyondLow || zBeyondHigh))){
            inLowLowLow = inputImages_data[inLowLowLowAddress + t*inputImages_stride_C];
		    inLowLowHigh = inputImages_data[inLowLowHighAddress + t*inputImages_stride_C];
		    inLowHighLow = inputImages_data[inLowHighLowAddress + t*inputImages_stride_C];
		    inLowHighHigh = inputImages_data[inLowHighHighAddress + t*inputImages_stride_C];
		    inHighLowLow = inputImages_data[inHighLowLowAddress + t*inputImages_stride_C];
		    inHighLowHigh = inputImages_data[inHighLowHighAddress + t*inputImages_stride_C];
		    inHighHighLow = inputImages_data[inHighHighLowAddress + t*inputImages_stride_C];
		    inHighHighHigh = inputImages_data[inHighHighHighAddress + t*inputImages_stride_C];
          }

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

int BilinearSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int zero_boundary)
{

  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_height = THFloatTensor_size(inputImages,1);
  int inputImages_width = THFloatTensor_size(inputImages,2);
  int output_height = THFloatTensor_size(output,1);
  int output_width = THFloatTensor_size(output,2);
  int inputImages_channels = THFloatTensor_size(inputImages,3);
  bool zero_boundary_bool= zero_boundary == 1;

  int output_strideBatch = THFloatTensor_stride(output,0);
  int output_strideHeight = THFloatTensor_stride(output,1);
  int output_strideWidth = THFloatTensor_stride(output,2);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_strideHeight = THFloatTensor_stride(inputImages,1);
  int inputImages_strideWidth = THFloatTensor_stride(inputImages,2);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_strideHeight = THFloatTensor_stride(grids,1);
  int grids_strideWidth = THFloatTensor_stride(grids,2);


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

int BilinearSamplerBXYZC_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim, int zero_boundary)
{
  switch( ndim )
    {
    case 1: return BilinearSamplerBXC_updateOutput_1D( inputImages, grids, output, zero_boundary); break;
    case 2: return BilinearSamplerBXYC_updateOutput_2D( inputImages, grids, output, zero_boundary); break;
    case 3: return BilinearSamplerBXYZC_updateOutput_3D( inputImages, grids, output, zero_boundary); break;
    default: return -1;
    }
}

int BilinearSamplerBCXYZ_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim, int zero_boundary)
{
  switch( ndim )
    {
    case 1: return BilinearSamplerBCX_updateOutput_1D( inputImages, grids, output, zero_boundary); break;
    case 2: return BilinearSamplerBCXY_updateOutput_2D( inputImages, grids, output, zero_boundary); break;
    case 3: return BilinearSamplerBCXYZ_updateOutput_3D( inputImages, grids, output, zero_boundary); break;
    default: return -1;
    }
}

int BilinearSamplerBXC_updateGradInput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary)
{
  bool onlyGrid=false;
  bool zero_boundary_bool= zero_boundary == 1;

  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,1);
  int gradOutput_X = THFloatTensor_size(gradOutput,1);
  int inputImages_C = THFloatTensor_size(inputImages,2);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput,0);
  int gradOutput_stride_X = THFloatTensor_stride(gradOutput,1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,1);

  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages,0);
  int gradInputImages_stride_X = THFloatTensor_stride(gradInputImages,1);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,1);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids,0);
  int gradGrids_stride_X = THFloatTensor_stride(gradGrids,1);

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



      bool xBeyondLow = xInLow < 0;
      bool xBeyondHigh = xInLow+1 > inputImages_X-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLow = 0;
            if (xBeyondHigh)
                xInLow = inputImages_X-2;
         }





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



      int t;

      for(t=0; t<inputImages_C; t++)
        {
	  real gradOutValue = gradOutput_data[gradOutputAddress + t];
	     if (!zero_boundary_bool || (! (xBeyondLow || xBeyondHigh ))){
              real inLow = inputImages_data[inLowAddress + t];
              lowDotProduct += inLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowAddress + t] += xWeightLow * gradOutValue;

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
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary)
{
  bool onlyGrid=false;
  bool zero_boundary_bool= zero_boundary == 1;

  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,2);
  int gradOutput_X = THFloatTensor_size(gradOutput,2);
  int inputImages_C = THFloatTensor_size(inputImages,1);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput,0);
  int gradOutput_stride_X = THFloatTensor_stride(gradOutput,2);
  int gradOutput_stride_C = THFloatTensor_stride(gradOutput,1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_C = THFloatTensor_stride(inputImages,1);

  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages,0);
  int gradInputImages_stride_X = THFloatTensor_stride(gradInputImages,2);
  int gradInputImages_stride_C = THFloatTensor_stride(gradInputImages,1);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,2);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids,0);
  int gradGrids_stride_X = THFloatTensor_stride(gradGrids,2);

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


      bool xBeyondLow = xInLow < 0;
      bool xBeyondHigh = xInLow+1 > inputImages_X-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLow = 0;
            if (xBeyondHigh)
                xInLow = inputImages_X-2;
         }

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
	     if (!zero_boundary_bool || (! (xBeyondLow || xBeyondHigh ))){
              real inLow = inputImages_data[inLowAddress + t*inputImages_stride_C];
              lowDotProduct += inLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowAddress + t*gradInputImages_stride_C] += xWeightLow * gradOutValue;

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
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary)
{
  bool onlyGrid=false;
  bool zero_boundary_bool= zero_boundary == 1;

  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,1);
  int inputImages_Y = THFloatTensor_size(inputImages,2);
  int gradOutput_X = THFloatTensor_size(gradOutput,1);
  int gradOutput_Y = THFloatTensor_size(gradOutput,2);
  int inputImages_C = THFloatTensor_size(inputImages,3);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput,0);
  int gradOutput_stride_X = THFloatTensor_stride(gradOutput,1);
  int gradOutput_stride_Y = THFloatTensor_stride(gradOutput,2);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,1);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,2);

  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages,0);
  int gradInputImages_stride_X = THFloatTensor_stride(gradInputImages,1);
  int gradInputImages_stride_Y = THFloatTensor_stride(gradInputImages,2);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,1);
  int grids_stride_Y = THFloatTensor_stride(grids,2);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids,0);
  int gradGrids_stride_X = THFloatTensor_stride(gradGrids,1);
  int gradGrids_stride_Y = THFloatTensor_stride(gradGrids,2);

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


        bool xBeyondLow = xInLowLow < 0;
        bool yBeyondLow = yInLowLow < 0;
        bool xBeyondHigh = xInLowLow+1 > inputImages_X-1;
        bool yBeyondHigh = yInLowLow+1 > inputImages_Y-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLowLow = 0;
            if (xBeyondHigh)
                xInLowLow = inputImages_X-2;
            if (yBeyondLow)
                yInLowLow = 0;
            if (yBeyondHigh)
                yInLowLow = inputImages_Y-2;
         }




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



        int t;

        for(t=0; t<inputImages_C; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if (!zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh))){
              real inLowLow = inputImages_data[inLowLowAddress + t];
              lowLowDotProduct += inLowLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowAddress + t] += xWeightLowLow * yWeightLowLow * gradOutValue;

              real inLowHigh = inputImages_data[inLowHighAddress + t];
              lowHighDotProduct += inLowHigh * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighAddress + t] += xWeightLowLow * (1-yWeightLowLow) * gradOutValue; // CHECK: CORRECT?


              real inHighLow = inputImages_data[inHighLowAddress + t];
              highLowDotProduct += inHighLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowAddress + t] += (1-xWeightLowLow) * yWeightLowLow * gradOutValue; // CHECK: CORRECT?

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
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary)
{
  bool onlyGrid=false;
  bool zero_boundary_bool = zero_boundary == 1;

  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,2);
  int inputImages_Y = THFloatTensor_size(inputImages,3);
  int gradOutput_X = THFloatTensor_size(gradOutput,2);
  int gradOutput_Y = THFloatTensor_size(gradOutput,3);
  int inputImages_C = THFloatTensor_size(inputImages,1);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput,0);
  int gradOutput_stride_X = THFloatTensor_stride(gradOutput,2);
  int gradOutput_stride_Y = THFloatTensor_stride(gradOutput,3);
  int gradOutput_stride_C = THFloatTensor_stride(gradOutput,1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,3);
  int inputImages_stride_C = THFloatTensor_stride(inputImages,1);

  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages,0);
  int gradInputImages_stride_X = THFloatTensor_stride(gradInputImages,2);
  int gradInputImages_stride_Y = THFloatTensor_stride(gradInputImages,3);
  int gradInputImages_stride_C = THFloatTensor_stride(gradInputImages,1);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,2);
  int grids_stride_Y = THFloatTensor_stride(grids,3);
  int grids_stride_C = THFloatTensor_stride(grids,1);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids,0);
  int gradGrids_stride_X = THFloatTensor_stride(gradGrids,2);
  int gradGrids_stride_Y = THFloatTensor_stride(gradGrids,3);
  int gradGrids_stride_C = THFloatTensor_stride(gradGrids,1);

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

        bool xBeyondLow = xInLowLow < 0;
        bool yBeyondLow = yInLowLow < 0;
        bool xBeyondHigh = xInLowLow+1 > inputImages_X-1;
        bool yBeyondHigh = yInLowLow+1 > inputImages_Y-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLowLow = 0;
            if (xBeyondHigh)
                xInLowLow = inputImages_X-2;
            if (yBeyondLow)
                yInLowLow = 0;
            if (yBeyondHigh)
                yInLowLow = inputImages_Y-2;
         }



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

        int t;


        for(t=0; t<inputImages_C; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t*gradOutput_stride_C];
           if (!zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh))){
              real inLowLow = inputImages_data[inLowLowAddress + t*inputImages_stride_C];
              lowLowDotProduct += inLowLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowAddress + t*gradInputImages_stride_C] += xWeightLowLow * yWeightLowLow * gradOutValue;

              real inLowHigh = inputImages_data[inLowHighAddress + t*inputImages_stride_C];
              lowHighDotProduct += inLowHigh * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighAddress + t*gradInputImages_stride_C] += xWeightLowLow * (1-yWeightLowLow) * gradOutValue; // CHECK: CORRECT?

              real inHighLow = inputImages_data[inHighLowAddress + t*inputImages_stride_C];
              highLowDotProduct += inHighLow * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowAddress + t*gradInputImages_stride_C] += (1-xWeightLowLow) * yWeightLowLow * gradOutValue; // CHECK: CORRECT?

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
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary)
{
  bool onlyGrid=false;
  bool zero_boundary_bool = zero_boundary == 1;

  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,1);
  int inputImages_Y = THFloatTensor_size(inputImages,2);
  int inputImages_Z = THFloatTensor_size(inputImages,3);
  int gradOutput_X = THFloatTensor_size(gradOutput,1);
  int gradOutput_Y = THFloatTensor_size(gradOutput,2);
  int gradOutput_Z = THFloatTensor_size(gradOutput,3);
  int inputImages_C = THFloatTensor_size(inputImages,4);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput,0);
  int gradOutput_stride_X = THFloatTensor_stride(gradOutput,1);
  int gradOutput_stride_Y = THFloatTensor_stride(gradOutput,2);
  int gradOutput_stride_Z = THFloatTensor_stride(gradOutput,3);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,1);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_Z = THFloatTensor_stride(inputImages,3);

  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages,0);
  int gradInputImages_stride_X = THFloatTensor_stride(gradInputImages,1);
  int gradInputImages_stride_Y = THFloatTensor_stride(gradInputImages,2);
  int gradInputImages_stride_Z = THFloatTensor_stride(gradInputImages,3);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,1);
  int grids_stride_Y = THFloatTensor_stride(grids,2);
  int grids_stride_Z = THFloatTensor_stride(grids,3);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids,0);
  int gradGrids_stride_X = THFloatTensor_stride(gradGrids,1);
  int gradGrids_stride_Y = THFloatTensor_stride(gradGrids,2);
  int gradGrids_stride_Z = THFloatTensor_stride(gradGrids,3);

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



	    bool xBeyondLow = xInLowLowLow < 0;
        bool yBeyondLow = yInLowLowLow < 0;
        bool zBeyondLow = zInLowLowLow < 0;
        bool xBeyondHigh = xInLowLowLow+1 > inputImages_X-1;
        bool yBeyondHigh = yInLowLowLow+1 > inputImages_Y-1;
        bool zBeyondHigh = zInLowLowLow+1 > inputImages_Z-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLowLowLow = 0;
            if (xBeyondHigh)
                xInLowLowLow = inputImages_X-2;
            if (yBeyondLow)
                yInLowLowLow = 0;
            if (yBeyondHigh)
                yInLowLowLow = inputImages_Y-2;
            if (zBeyondLow)
                zInLowLowLow = 0;
            if (zBeyondHigh)
                zInLowLowLow = inputImages_Z-2;
         }







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



	    int t;

	    for(t=0; t<inputImages_C; t++)
	      {
		    real gradOutValue = gradOutput_data[gradOutputAddress + t];
		    if (!zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh || zBeyondLow || zBeyondHigh))){
                real inLowLowLow = inputImages_data[inLowLowLowAddress + t];
                lowLowLowDotProduct += inLowLowLow * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowLowAddress + t] += xWeightLowLowLow * yWeightLowLowLow * zWeightLowLowLow * gradOutValue;

                real inLowLowHigh = inputImages_data[inLowLowHighAddress + t];
                lowLowHighDotProduct += inLowLowHigh * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowHighAddress + t] += xWeightLowLowLow * yWeightLowLowLow * (1-zWeightLowLowLow) * gradOutValue; // CHECK: CORRECT?

                real inLowHighLow = inputImages_data[inLowHighLowAddress + t];
                lowHighLowDotProduct += inLowHighLow * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighLowAddress + t] += xWeightLowLowLow * (1-yWeightLowLowLow) * zWeightLowLowLow * gradOutValue; // CHECK: CORRECT?

                real inLowHighHigh = inputImages_data[inLowHighHighAddress + t];
                lowHighHighDotProduct += inLowHighHigh * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighHighAddress + t] += xWeightLowLowLow * (1 - yWeightLowLowLow) * (1-zWeightLowLowLow) * gradOutValue;

                real inHighLowLow = inputImages_data[inHighLowLowAddress + t];
                highLowLowDotProduct += inHighLowLow * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowLowAddress + t] += (1-xWeightLowLowLow) * yWeightLowLowLow * zWeightLowLowLow * gradOutValue;

                real inHighLowHigh = inputImages_data[inHighLowHighAddress + t];
                highLowHighDotProduct += inHighLowHigh * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowHighAddress + t] += (1-xWeightLowLowLow) * yWeightLowLowLow * (1-zWeightLowLowLow) * gradOutValue; // CHECK: CORRECT?

                real inHighHighLow = inputImages_data[inHighHighLowAddress + t];
                highHighLowDotProduct += inHighHighLow * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesHighHighLowAddress + t] += (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * zWeightLowLowLow * gradOutValue; // CHECK: CORRECT?

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
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary)
{
  bool onlyGrid=false;
  bool zero_boundary_bool = zero_boundary == 1;

  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_X = THFloatTensor_size(inputImages,2);
  int inputImages_Y = THFloatTensor_size(inputImages,3);
  int inputImages_Z = THFloatTensor_size(inputImages,4);
  int gradOutput_X = THFloatTensor_size(gradOutput,2);
  int gradOutput_Y = THFloatTensor_size(gradOutput,3);
  int gradOutput_Z = THFloatTensor_size(gradOutput,4);
  int inputImages_C = THFloatTensor_size(inputImages,1);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput,0);
  int gradOutput_stride_X = THFloatTensor_stride(gradOutput,2);
  int gradOutput_stride_Y = THFloatTensor_stride(gradOutput,3);
  int gradOutput_stride_Z = THFloatTensor_stride(gradOutput,4);
  int gradOutput_stride_C = THFloatTensor_stride(gradOutput,1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_stride_X = THFloatTensor_stride(inputImages,2);
  int inputImages_stride_Y = THFloatTensor_stride(inputImages,3);
  int inputImages_stride_Z = THFloatTensor_stride(inputImages,4);
  int inputImages_stride_C = THFloatTensor_stride(inputImages,1);

  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages,0);
  int gradInputImages_stride_X = THFloatTensor_stride(gradInputImages,2);
  int gradInputImages_stride_Y = THFloatTensor_stride(gradInputImages,3);
  int gradInputImages_stride_Z = THFloatTensor_stride(gradInputImages,4);
  int gradInputImages_stride_C = THFloatTensor_stride(gradInputImages,1);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_stride_X = THFloatTensor_stride(grids,2);
  int grids_stride_Y = THFloatTensor_stride(grids,3);
  int grids_stride_Z = THFloatTensor_stride(grids,4);
  int grids_stride_C = THFloatTensor_stride(grids,1);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids,0);
  int gradGrids_stride_X = THFloatTensor_stride(gradGrids,2);
  int gradGrids_stride_Y = THFloatTensor_stride(gradGrids,3);
  int gradGrids_stride_Z = THFloatTensor_stride(gradGrids,4);
  int gradGrids_stride_C = THFloatTensor_stride(gradGrids,1);

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


	    bool xBeyondLow = xInLowLowLow < 0;
        bool yBeyondLow = yInLowLowLow < 0;
        bool zBeyondLow = zInLowLowLow < 0;
        bool xBeyondHigh = xInLowLowLow+1 > inputImages_X-1;
        bool yBeyondHigh = yInLowLowLow+1 > inputImages_Y-1;
        bool zBeyondHigh = zInLowLowLow+1 > inputImages_Z-1;

        ///////////////  using  non zero border condition

        if (!zero_boundary_bool) {
            if (xBeyondLow)
                xInLowLowLow = 0;
            if (xBeyondHigh)
                xInLowLowLow = inputImages_X-2;
            if (yBeyondLow)
                yInLowLowLow = 0;
            if (yBeyondHigh)
                yInLowLowLow = inputImages_Y-2;
            if (zBeyondLow)
                zInLowLowLow = 0;
            if (zBeyondHigh)
                zInLowLowLow = inputImages_Z-2;
         }




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



	    int t;

       for(t=0; t<inputImages_C; t++)
       {
		    real gradOutValue = gradOutput_data[gradOutputAddress + t*gradOutput_stride_C];
		   if (!zero_boundary_bool || (! (xBeyondLow || yBeyondLow || xBeyondHigh || yBeyondHigh || zBeyondLow || zBeyondHigh))){
		    real inLowLowLow = inputImages_data[inLowLowLowAddress + t*inputImages_stride_C];
		    lowLowLowDotProduct += inLowLowLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowLowAddress + t*gradInputImages_stride_C] += xWeightLowLowLow * yWeightLowLowLow * zWeightLowLowLow * gradOutValue;

		    real inLowLowHigh = inputImages_data[inLowLowHighAddress + t*inputImages_stride_C];
		    lowLowHighDotProduct += inLowLowHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowLowHighAddress + t*gradInputImages_stride_C] += xWeightLowLowLow * yWeightLowLowLow * (1-zWeightLowLowLow) * gradOutValue; // CHECK: CORRECT?

		    real inLowHighLow = inputImages_data[inLowHighLowAddress + t*inputImages_stride_C];
		    lowHighLowDotProduct += inLowHighLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighLowAddress + t*gradInputImages_stride_C] += xWeightLowLowLow * (1-yWeightLowLowLow) * zWeightLowLowLow * gradOutValue; // CHECK: CORRECT?

		    real inLowHighHigh = inputImages_data[inLowHighHighAddress + t*inputImages_stride_C];
		    lowHighHighDotProduct += inLowHighHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesLowHighHighAddress + t*gradInputImages_stride_C] += xWeightLowLowLow * (1 - yWeightLowLowLow) * (1-zWeightLowLowLow) * gradOutValue;

		    real inHighLowLow = inputImages_data[inHighLowLowAddress + t*inputImages_stride_C];
		    highLowLowDotProduct += inHighLowLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowLowAddress + t*gradInputImages_stride_C] += (1-xWeightLowLowLow) * yWeightLowLowLow * zWeightLowLowLow * gradOutValue;

		    real inHighLowHigh = inputImages_data[inHighLowHighAddress + t*inputImages_stride_C];
		    highLowHighDotProduct += inHighLowHigh * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighLowHighAddress + t*gradInputImages_stride_C] += (1-xWeightLowLowLow) * yWeightLowLowLow * (1-zWeightLowLowLow) * gradOutValue; // CHECK: CORRECT?

		    real inHighHighLow = inputImages_data[inHighHighLowAddress + t*inputImages_stride_C];
		    highHighLowDotProduct += inHighHighLow * gradOutValue;
		    if(!onlyGrid) gradInputImages_data[gradInputImagesHighHighLowAddress + t*gradInputImages_stride_C] += (1-xWeightLowLowLow) * (1-yWeightLowLowLow) * zWeightLowLowLow * gradOutValue; // CHECK: CORRECT?

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
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int zero_boundary)
{
  bool onlyGrid=false;
  bool zero_boundary_bool = zero_boundary == 1;

  int batchsize = THFloatTensor_size(inputImages,0);
  int inputImages_height = THFloatTensor_size(inputImages,1);
  int inputImages_width = THFloatTensor_size(inputImages,2);
  int gradOutput_height = THFloatTensor_size(gradOutput,1);
  int gradOutput_width = THFloatTensor_size(gradOutput,2);
  int inputImages_channels = THFloatTensor_size(inputImages,3);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput,0);
  int gradOutput_strideHeight = THFloatTensor_stride(gradOutput,1);
  int gradOutput_strideWidth = THFloatTensor_stride(gradOutput,2);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages,0);
  int inputImages_strideHeight = THFloatTensor_stride(inputImages,1);
  int inputImages_strideWidth = THFloatTensor_stride(inputImages,2);

  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages,0);
  int gradInputImages_strideHeight = THFloatTensor_stride(gradInputImages,1);
  int gradInputImages_strideWidth = THFloatTensor_stride(gradInputImages,2);

  int grids_strideBatch = THFloatTensor_stride(grids,0);
  int grids_strideHeight = THFloatTensor_stride(grids,1);
  int grids_strideWidth = THFloatTensor_stride(grids,2);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids,0);
  int gradGrids_strideHeight = THFloatTensor_stride(gradGrids,1);
  int gradGrids_strideWidth = THFloatTensor_stride(gradGrids,2);

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
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int ndim, int zero_boundary)
{
  switch( ndim )
    {
    case 1: return BilinearSamplerBXC_updateGradInput_1D( inputImages, grids, gradInputImages, gradGrids, gradOutput,zero_boundary ); break;
    case 2: return BilinearSamplerBXYC_updateGradInput_2D( inputImages, grids, gradInputImages, gradGrids, gradOutput,zero_boundary ); break;
    case 3: return BilinearSamplerBXYZC_updateGradInput_3D( inputImages, grids, gradInputImages, gradGrids, gradOutput,zero_boundary ); break;
    default: return -1;
    }
}

int BilinearSamplerBCXYZ_updateGradInput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
					   THFloatTensor *gradGrids, THFloatTensor *gradOutput, int ndim, int zero_boundary)
{
  switch( ndim )
    {
    case 1: return BilinearSamplerBCX_updateGradInput_1D( inputImages, grids, gradInputImages, gradGrids, gradOutput,zero_boundary ); break;
    case 2: return BilinearSamplerBCXY_updateGradInput_2D( inputImages, grids, gradInputImages, gradGrids, gradOutput,zero_boundary ); break;
    case 3: return BilinearSamplerBCXYZ_updateGradInput_3D( inputImages, grids, gradInputImages, gradGrids, gradOutput,zero_boundary ); break;
    default: return -1;
    }
}




