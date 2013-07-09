#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ConvolutionHelper.h"

#include <stdio.h>

//Ура! Вперед, к светлому будущему параллельных вычислений!


__global__ void cudaGetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> xGradient, CUDAArray<float> yGradient)
{
	int row = defaultRow();
	int column = defaultColumn();
	float newValue = xGradient.At(row,column)*xGradient.At(row,column) +yGradient.At(row,column)*yGradient.At(row,column);
	newValue = sqrt(newValue);
	magnitude.SetAt(row,column, newValue);
}

void GetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> xGradient, CUDAArray<float> yGradient)
{
		dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(magnitude.Width,defaultThreadCount),
		ceilMod(magnitude.Height,defaultThreadCount));

	cudaGetMagnitude<<<gridSize,blockSize>>>(magnitude, xGradient, yGradient);

	cudaError_t error = cudaDeviceSynchronize();
}

__global__ void cudaGetMask(CUDAArray<float> initialArray, CUDAArray<bool> mask, int blockSize, float average)
{
	float sum = 0;
	for(int i; i<blockSize; i++)
	{
		for(int j; j<blockSize; j++)
		{
			if(defaultRow()*blockSize+j<initialArray.Height&&
				defaultColumn()*blockSize+i<initialArray.Width)
			{
			sum += initialArray.At(defaultRow()*blockSize+j,defaultColumn()*blockSize+i);
			}
		}
	}
	sum = sum/(blockSize*blockSize);
	mask.SetAt(defaultRow(),defaultColumn(),!(sum < average));
}

float GetAverageFromArray(CUDAArray<float> arrayToAverage)
{
	float sum = 0;
	for(int i; i<arrayToAverage.Width; i++)
	{		
		for(int j; j<arrayToAverage.Height; j++)
		{
			sum+= arrayToAverage.At(i,j);
		}
	}
	return sum/(float)(arrayToAverage.Height*arrayToAverage.Width);
}

CUDAArray<float> loadImage(const char* name, bool sourceIsFloat = false)
{
	FILE* f = fopen(name,"rb");
			
	int width;
	int height;
	
	fread(&width,sizeof(int),1,f);
			
	fread(&height,sizeof(int),1,f);
	
	float* ar2 = (float*)malloc(sizeof(float)*width*height);

	if(!sourceIsFloat)
	{
		int* ar = (int*)malloc(sizeof(int)*width*height);
		fread(ar,sizeof(int),width*height,f);
		for(int i=0;i<width*height;i++)
		{
			ar2[i]=ar[i];
		}
		
		free(ar);
	}
	else
	{
		fread(ar2,sizeof(float),width*height,f);
	}
	
	fclose(f);

	CUDAArray<float> sourceImage = CUDAArray<float>(ar2,width,height);

	free(ar2);		

	return sourceImage;
}

void SaveMask()
{
}


  int main(float* img, int xSizeImg, int ySizeImg, int windowSize, float weightConstant, int threshold)
  {
	  //parameters
	  float weightConstan = 0.3; 
	  int windowSize = 12;
	  int threshold = 5;

	  //source image
	  CUDAArray<float> source = loadImage("C:\\temp\\104_6.bin");
	  int xSizeImg = source.Width;		  
	  int ySizeImg = source.Height;

	  // Sobel:	  
	  CUDAArray<float> xGradient = CUDAArray<float>(xSizeImg,ySizeImg);
	  CUDAArray<float> yGradient = CUDAArray<float>(xSizeImg,ySizeImg);

	  float xKernelCPU[3][3] = {{-1,0,1},
							{-2,0,2},
							{-1,0,1}};
	  CUDAArray<float> xKernel = CUDAArray<float>(*xKernelCPU,3,3);
	  
	  float yKernelCPU[3][3] = {{-1,-2,-1},
							{0,0,0},
							{1,2,1}};
	  CUDAArray<float> yKernel = CUDAArray<float>(*yKernelCPU,3,3);
	  
	  Convolve(xGradient, source, xKernel);
	  Convolve(yGradient, source, yKernel);

	  //magnitude of gradient
	  CUDAArray<float> magnitude = CUDAArray<float>(xSizeImg,ySizeImg);


	  xGradient.Dispose();
	  yGradient.Dispose();
	  xKernel.Dispose();
	  yKernel.Dispose();

	  //average magnitude 
	  float averege = GetAverageFromArray(magnitude);

	  //dementions of mask
	  int N = (int)ceil(((double)source.Width) / windowSize);
	  int M = (int)ceil(((double)source.Height) / windowSize);
	  
	  //thread configuration in CUDA
	  	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
		dim3 gridSize =dim3(ceilMod(N,defaultThreadCount),
							ceilMod(M,defaultThreadCount));

		//mask creation
		CUDAArray<bool> mask = CUDAArray<bool>(N,M);
		cudaGetMask<<<gridSize, blockSize>>>(magnitude, mask, WindowSize, average*weightConstant);
		


		return 0;        
  }
