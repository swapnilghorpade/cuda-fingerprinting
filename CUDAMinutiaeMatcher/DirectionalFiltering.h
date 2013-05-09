#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "LinearSymmetry.h"

const float SigmaDirection = 2.0f;

const int KernelSize = 13; //recalculate by 2*(int)ceil(SigmaDirection*3.0f)+1;
const int HalfSize = KernelSize/2; //recalculate by 2*(int)ceil(SigmaDirection*3.0f)+1;

const int DirectionsNumber = 20;

__device__ const int ringInnerRadius = 4;
__device__ const int ringOuterRadius = 6;

__constant__ int constDirectionsX[KernelSize*DirectionsNumber];
__constant__ int constDirectionsY[KernelSize*DirectionsNumber];
__constant__ float constDirectionsKernel[KernelSize];

// GPU FUNCTIONS

__global__ void cudaDirectionalFiltering(CUDAArray<float> result, CUDAArray<float> lsMagnitude, CUDAArray<float> lsPhase, float tau1, float tau2)
{
	// step 1: copy locally all necessary values for annulus calculation
	__shared__ float imageCache[32*32];
	__shared__ float magnitudeCache[32*32];
	__shared__ float kernel[KernelSize];

	int realThreadX = threadIdx.x-HalfSize;
	int realThreadY = threadIdx.y-HalfSize;

	int realBlockSize = blockDim.x-HalfSize*2; // block without apron

	int rowToCopy = blockIdx.y*realBlockSize + realThreadY; 
	int columnToCopy = blockIdx.x*realBlockSize + realThreadX;
	if(threadIdx.y==0&&threadIdx.x<KernelSize)
	{
		kernel[threadIdx.x] = constDirectionsKernel[threadIdx.x];
	}
	if(rowToCopy<0)rowToCopy = 0;
	if(rowToCopy>=lsMagnitude.Height)rowToCopy = lsMagnitude.Height-1;

	
	if(columnToCopy<0)columnToCopy = 0;
	if(columnToCopy>=lsMagnitude.Width)columnToCopy = lsMagnitude.Width-1;

	imageCache[32*threadIdx.y+threadIdx.x] = result.At(rowToCopy, columnToCopy);
	magnitudeCache[32*threadIdx.y+threadIdx.x] = lsMagnitude.At(rowToCopy, columnToCopy);

	__syncthreads();
	// step 2: do the filtering
	if(realThreadY>=0&&realThreadX>=0&&realThreadY<realBlockSize&&realThreadX<realBlockSize)
	{
		rowToCopy = blockIdx.y*realBlockSize + realThreadY; 
		columnToCopy = blockIdx.x*realBlockSize + realThreadX;
		if(rowToCopy<lsMagnitude.Height&&columnToCopy<lsMagnitude.Width)
		{

			if(magnitudeCache[threadIdx.y*32+threadIdx.x] < tau1)
			{
				result.SetAt(rowToCopy,columnToCopy,0.0f);
			}
			else
			{
				// annulus testing
				float sum = 0;
				int area = 0;
				for (int dx = -ringOuterRadius; dx <= ringOuterRadius; dx++)
				{
					if(abs(dx) < ringInnerRadius) continue;
					for (int dy = -ringOuterRadius; dy <= ringOuterRadius; dy++)
					{
						if (abs(dy) < ringInnerRadius)  continue;
						sum += magnitudeCache[(threadIdx.y+dy)*32+threadIdx.x+dx];
						area++;
					}
				}
				if (sum / area < tau2) 
					result.SetAt(rowToCopy,columnToCopy,0.0f);
				else
				{
					float phase = lsPhase.At(rowToCopy, columnToCopy);
					phase = phase / 2.0f - CUDART_PIO2_F;
					if (phase > CUDART_PI_F*39/40) phase -= CUDART_PI_F;
					if (phase < -CUDART_PI_F/40) phase += CUDART_PI_F;
				
					int direction = (int)round(phase / (CUDART_PI_F / 20));
				
					float avg = 0.0f;
				
					for (int i = 0; i < KernelSize; i++)
					{
						int x = constDirectionsX[i + KernelSize*direction];
						int y = constDirectionsY[i + KernelSize*direction];
					
						avg += kernel[i] * imageCache[32*( -y+threadIdx.y)+x+threadIdx.x];
					}
					result.SetAt(rowToCopy, columnToCopy, avg);
				}
			}
		}
	}
}

// CPU FUNCTIONS

void FillDirections()
{
	int* directionsX = (int*)malloc(sizeof(int)*DirectionsNumber*KernelSize);
	int* directionsY = (int*)malloc(sizeof(int)*DirectionsNumber*KernelSize);
	//float* fX = (float*)malloc(sizeof(float)*DirectionsNumber*KernelSize);
	//float* fY = (float*)malloc(sizeof(float)*DirectionsNumber*KernelSize);

	for (int n = 0; n < DirectionsNumber/2; n++)
	{
		float angle = CUDART_PI_F*n/DirectionsNumber;
		
		directionsX[(KernelSize/2) +n*KernelSize] = 0;
		directionsY[(KernelSize/2) +n*KernelSize] = 0;

		float tg = tan(angle);
		
		if (angle <= CUDART_PIO4_F)
		{
			for (int x = 1; x <= KernelSize/2; x++)
			{
				int y = (int)round(tg * x);
				directionsX[(KernelSize/2+x)+n*KernelSize] = x;
				directionsY[(KernelSize/2+x)+n*KernelSize] = y;
				directionsX[(KernelSize/2-x)+n*KernelSize] = -x;
				directionsY[(KernelSize/2-x)+n*KernelSize] = -y;
			}
		}
		else
		{
			for (int y = 1; y <= KernelSize/2; y++)
			{
				int x = (int)round((float)y/tg);

				directionsX[(KernelSize/2+y)+n*KernelSize] = x;
				directionsY[(KernelSize/2+y)+n*KernelSize] = y;
				directionsX[(KernelSize/2-y)+n*KernelSize] = -x;
				directionsY[(KernelSize/2-y)+n*KernelSize] = -y;
			}
		}
	}

	for (int n = DirectionsNumber/2; n < DirectionsNumber; n++)
	{
		for (int i = 0; i < KernelSize; i++)
		{
			int x = directionsX[i + (n - DirectionsNumber/2)*KernelSize];
			int y = directionsY[i + (n - DirectionsNumber/2)*KernelSize];
			
			directionsX[i+ n*KernelSize] = y;
			directionsY[i+ n*KernelSize] = -x;
		}
	}
	
	float* kernel = (float*)malloc(KernelSize*sizeof(float));
	float ksum = 0;
	
	for (int i = 0; i < KernelSize; i++)
	{
		ksum+=kernel[i] = Gaussian1D(i - KernelSize/2, SigmaDirection);
	}
	
	for (int i = 0; i < KernelSize; i++)
	{
		kernel[i] /= ksum;
	}
	
	cudaError_t error;

	error = cudaMemcpyToSymbol(constDirectionsX, directionsX, sizeof(int)*DirectionsNumber*KernelSize);
	error = cudaMemcpyToSymbol(constDirectionsY, directionsY, sizeof(int)*DirectionsNumber*KernelSize);
	error = cudaMemcpyToSymbol(constDirectionsKernel, kernel, sizeof(float)*KernelSize);

	free(kernel);
	//for(int i=0;i<KernelSize*DirectionsNumber;i++)
	//{
	//	fX[i]=directionsX[i];
	//	fY[i]=directionsY[i];
	//}
	//CUDAArray<float> dirX = CUDAArray<float>(fX,KernelSize, DirectionsNumber);
	//CUDAArray<float> dirY = CUDAArray<float>(fY,KernelSize, DirectionsNumber);
	//SaveArray(dirX,"C:\\temp\\dirX.bin");
	//SaveArray(dirY,"C:\\temp\\dirY.bin");
	//kernel = (float*)malloc(KernelSize*sizeof(float));
	free(directionsX);
	free(directionsY);
	//error = cudaMemcpyFromSymbol(kernel, constDirectionsKernel, sizeof(float)*KernelSize);
}

void DirectionFiltering(CUDAArray<float> l, CUDAArray<float> lsReal, CUDAArray<float> lsImaginary, float tau1, float tau2)
{
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(lsReal.Width, defaultThreadCount),
		ceilMod(lsReal.Height, defaultThreadCount));

	cudaError_t error;

	CUDAArray<float> magnitude = CUDAArray<float>(l.Width, l.Height);

	cudaGetMagnitude<<<gridSize, blockSize>>>(magnitude, lsReal, lsImaginary);

	error = cudaDeviceSynchronize();

	CUDAArray<float> phase = CUDAArray<float>(l.Width, l.Height);

	cudaGetPhase<<<gridSize, blockSize>>>(phase, lsReal, lsImaginary);

	error = cudaDeviceSynchronize();

	gridSize = 
		dim3(ceilMod(l.Width, (defaultThreadCount-HalfSize*2)),
		ceilMod(l.Height, (defaultThreadCount-HalfSize*2)));
	//SaveArray(l,"C:\\temp\\l1.bin");
	cudaDirectionalFiltering<<<gridSize, blockSize>>>(l, magnitude, phase, tau1, tau2);
	
	error = cudaDeviceSynchronize();

	magnitude.Dispose();
	phase.Dispose();

	//SaveArray(l,"C:\\temp\\104_6_df_sq.bin");
}