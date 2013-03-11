#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "ConvolutionHelper.h"

// GPU FUNCTIONS

__global__ void cudaComplexSquare(CUDAArray<float> real, CUDAArray<float> im)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(im.Width>column&&im.Height>row)
	{
		float realValue = real.At(row,column);
		float imValue = im.At(row,column);
		real.SetAt(row,column,realValue*realValue-imValue*imValue);
		im.SetAt(row,column,realValue*imValue*2.0f);
	}
}

__global__ void cudaGetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> real, CUDAArray<float> im)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(im.Width>column&&im.Height>row)
	{
		float realValue = real.At(row,column);
		float imValue = im.At(row,column);
		magnitude.SetAt(row,column,sqrtf(realValue*realValue+imValue*imValue));
	}
}

__global__ void cudaGetPhase(CUDAArray<float> phase, CUDAArray<float> real, CUDAArray<float> im)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(im.Width>column&&im.Height>row)
	{
		float realValue = real.At(row,column);
		float imValue = im.At(row,column);
		phase.SetAt(row,column,atan2f(imValue, realValue));
	}
}

__global__ void cudaNormalizeLS(CUDAArray<float> real, CUDAArray<float> im, CUDAArray<float> i11)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(im.Width>column&&im.Height>row)
	{
		float realValue = real.At(row,column);
		float imValue = im.At(row,column);
		float norm = i11.At(row,column);
		real.SetAt(row,column,realValue/norm);
		im.SetAt(row,column,imValue/norm);
	}
}

__global__ void correctLS1WithLS2(CUDAArray<float> ls1Real, CUDAArray<float> ls1Im,
	CUDAArray<float> ls2Real, CUDAArray<float> ls2Im)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(ls1Real.Width>column&&ls1Real.Height>row)
	{
		float real1 = ls1Real.At(row,column);
		float im1 = ls1Im.At(row,column);
		float real2 = ls2Real.At(row,column);
		float im2 = ls2Im.At(row,column);
		float a1 = atan2(im1, real1);
		float a2 = atan2(im2, real2);
		float multiplier = abs(cos(a1-a2));

		ls1Real.SetAt(row,column,real1*multiplier);
		ls1Im.SetAt(row,column,im1*multiplier);
	}
}

// CPU FUNCTIONS

CUDAArray<float> MakeDifferentialGaussianKernel(float kx, float ky, float c, float sigma)
{
	int size = 2*(int)ceil(sigma*3.0f)+1;
	int center=size/2;
	float* kernel = (float*)malloc(sizeof(float)*size*size);
	float sum=0;
	for(int row=-center; row<=center; row++)
	{
		for(int column=-center; column<=center; column++)
		{
			sum+= kernel[column+center+(row+center)*size] = Gaussian2D(column,row,sigma)*(kx*column+ky*row+c);
		}
	}
	if (abs(sum) >0.00001f)
	for(int row=-center; row<=center; row++)
	{
		for(int column=-center; column<=center; column++)
		{
			kernel[column+center+(row+center)*size]/=sum;
		}
	}

	CUDAArray<float> cudaKernel = CUDAArray<float>(kernel,size,size);

	free(kernel);

	return cudaKernel;
}

void EstimateLS(CUDAArray<float>* real, CUDAArray<float>* im, CUDAArray<float> source, float sigma1, float sigma2)
{
	CUDAArray<float> kernel1 = MakeDifferentialGaussianKernel(-1,0,0,sigma1);
	CUDAArray<float> kernel2 = MakeDifferentialGaussianKernel(0,1,0,sigma1);

	CUDAArray<float> sourceX = CUDAArray<float>(source.Width, source.Height);
	Convolve(sourceX, source, kernel1);

	CUDAArray<float> sourceY = CUDAArray<float>(source.Width, source.Height);
	Convolve(sourceY, source, kernel2);

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(source.Width, defaultThreadCount),
		ceilMod(source.Height, defaultThreadCount));

	cudaComplexSquare<<<gridSize, blockSize>>>(sourceX, sourceY);

	CUDAArray<float> kernel3 = MakeDifferentialGaussianKernel(0,0,1,sigma2);
	CUDAArray<float> kernel4 = MakeDifferentialGaussianKernel(0,0,0,sigma2);
	
	CUDAArray<float> resultReal = CUDAArray<float>(source.Width, source.Height);
	CUDAArray<float> resultIm = CUDAArray<float>(source.Width, source.Height);

	Convolve(resultReal, sourceX ,kernel3);
	Convolve(resultIm, sourceY ,kernel3);

    //ComplexConvolve(resultReal, resultIm, sourceX, sourceY, kernel3, kernel4);

	CUDAArray<float> 
		magnitude = CUDAArray<float>(source.Width, source.Height);

	cudaGetMagnitude<<<gridSize,blockSize>>>(magnitude, sourceX, sourceY);

	CUDAArray<float> I11 = CUDAArray<float>(source.Width, source.Height);

	Convolve(I11, magnitude, kernel3);

	cudaNormalizeLS<<<gridSize,blockSize>>>(resultReal, resultIm, I11);
	//cudaGetMagnitude<<<gridSize,blockSize>>>(magnitude, resultReal, resultIm);
	//SaveArray(magnitude,"C:\\temp\\104_6_mag2.bin");
	*real = resultReal;
	*im = resultIm;

	kernel1.Dispose();
	kernel2.Dispose();
	kernel3.Dispose();
	kernel4.Dispose();

	magnitude.Dispose();

	sourceX.Dispose();
	sourceY.Dispose();
}

void CorrectLS1WithLS2(CUDAArray<float> ls1Real, CUDAArray<float> ls1Im, 
	CUDAArray<float> ls2Real, CUDAArray<float> ls2Im)
{
	CUDAArray<float> ls2RealScaled = Expand(ls2Real, 1.3f, ls1Real.Width, ls1Real.Height);
	CUDAArray<float> ls2ImScaled = Expand(ls2Im, 1.3f, ls1Im.Width, ls1Im.Height);

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(ls1Real.Width, defaultThreadCount),
		ceilMod(ls1Real.Height, defaultThreadCount));

	correctLS1WithLS2<<<gridSize, blockSize>>>(ls1Real, ls1Im, ls2RealScaled, ls2ImScaled);

	ls2RealScaled.Dispose();
	ls2ImScaled.Dispose();
}