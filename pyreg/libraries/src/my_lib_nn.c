#ifndef no_openmp
#include <omp.h>
#endif

#include <TH/TH.h>
#include <stdbool.h>
#include <stdio.h>

#define real float


int nearestNeighBCX_updateOutput_1D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
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

	  real xcoord = (xf + 1) * (inputImages_X - 1) / 2; // map it from [-1,1] to [0,1]
	  xLow = round(xcoord);

	  const int outAddress = output_strideBatch * b + output_stride_X * xOut;
	  const int inLowAddress = inputImages_strideBatch * b + inputImages_stride_X * xLow;

	  real v=0;
	  real inLow=0;

	  // we are careful with the boundaries
	  bool lowIsIn = xLow >= 0 && xLow <= inputImages_X-1; 

	  int t;
	  // interpolation happens here
	  for(t=0; t<inputImages_C; t++)
	    {
	      if(lowIsIn) inLow = inputImages_data[inLowAddress + t*inputImages_stride_C];

	      v = inLow;

	      output_data[outAddress + t*output_stride_C] = v;
	    }

	}
    }

  return 1;
}



int nearestNeighBCXY_updateOutput_2D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
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

        real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
        xInLowLow = round(xcoord);

        real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
        yInLowLow = round(ycoord);

        const int outAddress = output_strideBatch * b + output_stride_Y * yOut + output_stride_X * xOut;
        const int inLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Y * yInLowLow + inputImages_stride_X * xInLowLow;

        real v=0;
        real inLowLow=0;


        // we are careful with the boundaries
        bool lowLowIsIn =  xInLowLow >= 0 && xInLowLow <= inputImages_X-1 && yInLowLow >= 0 && yInLowLow <= inputImages_Y-1;


        int t;
        // interpolation happens here
        for(t=0; t<inputImages_C; t++)
        {
           if(lowLowIsIn) inLowLow = inputImages_data[inLowLowAddress + t*inputImages_stride_C];


           v = inLowLow;

           output_data[outAddress + t*output_stride_C] = v;
        }

      }
    }
  }

  return 1;

}



int nearestNeighBCXYZ_updateOutput_3D(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output )
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

		real xcoord = (xf + 1) * (inputImages_X - 1) / 2;
		xInLowLowLow = round(xcoord);

		real ycoord = (yf + 1) * (inputImages_Y - 1) / 2;
		yInLowLowLow = round(ycoord);

		real zcoord = (zf + 1) * (inputImages_Z - 1) / 2;
		zInLowLowLow = round(zcoord);

		const int outAddress = output_strideBatch * b + output_stride_Z * zOut + output_stride_Y * yOut + output_stride_X * xOut;
		const int inLowLowLowAddress = inputImages_strideBatch * b + inputImages_stride_Z * zInLowLowLow + inputImages_stride_Y * yInLowLowLow + inputImages_stride_X * xInLowLowLow;


		real v=0;
		real inLowLowLow=0;


		// we are careful with the boundaries
		bool lowLowLowIsIn = xInLowLowLow >= 0 && xInLowLowLow <= inputImages_X-1
		  && yInLowLowLow >= 0 && yInLowLowLow <= inputImages_Y-1
		  && zInLowLowLow >= 0 && zInLowLowLow <= inputImages_Z-1;


		int t;
		// interpolation happens here
		for(t=0; t<inputImages_C; t++)
		  {
		    if(lowLowLowIsIn) inLowLowLow = inputImages_data[inLowLowLowAddress + t*inputImages_stride_C];


		    v =  inLowLowLow;

		    output_data[outAddress + t*output_stride_C] = v;
		  }

	      }
	  }
      }
  }
  return 1;

}





int nearestNeighBCXYZ_updateOutput_ND(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output, int ndim)
{
  switch( ndim )
    {
    case 1: return nearestNeighBCX_updateOutput_1D( inputImages, grids, output ); break;
    case 2: return nearestNeighBCXY_updateOutput_2D( inputImages, grids, output ); break;
    case 3: return nearestNeighBCXYZ_updateOutput_3D( inputImages, grids, output ); break;
    default: return -1;
    }
}


