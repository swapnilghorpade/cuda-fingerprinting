#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "LinearSymmetry.h"

const float SigmaDirection = 1.6f;

const int KernelSize = 11; //recalculate by 2*(int)ceil(SigmaDirection*3.0f)+1;
const int HalfSize = KernelSize/2; //recalculate by 2*(int)ceil(SigmaDirection*3.0f)+1;

const int DirectionsNumber = 20;

const int annulusRadius = 2;

__constant__ int constDirectionsX[KernelSize*DirectionsNumber];
__constant__ int constDirectionsY[KernelSize*DirectionsNumber];
__constant__ float constDirectionsKernel[KernelSize];

// GPU FUNCTIONS

__global__ void cudaDirectionalFiltering(CUDAArray<float> result, CUDAArray<float> lsMagnitude, CUDAArray<float> lsPhase, float tau1, float tau2)
{
	// step 1: copy locally all necessary values for annulus calculation
	__shared__ float imageCache[32*32];

	int realThreadX = threadIdx.x-HalfSize;
	int realThreadY = threadIdx.y-HalfSize;

	int realBlockSize = blockDim.x-HalfSize*2; // block without apron

	int rowToCopy = blockIdx.y*realBlockSize + realThreadY; 
	if(rowToCopy<0)rowToCopy = 0;
	if(rowToCopy>lsMagnitude.Height)rowToCopy = lsMagnitude.Height-1;

	int columnToCopy = blockIdx.x*realBlockSize + realThreadX;
	if(columnToCopy<0)columnToCopy = 0;
	if(columnToCopy>lsMagnitude.Height)rowToCopy = lsMagnitude.Width-1;

	imageCache[32*rowToCopy+columnToCopy] = lsMagnitude.At(rowToCopy, columnToCopy);

	__syncthreads();
	// step 2: do the filtering

	rowToCopy = blockIdx.y*realBlockSize + realThreadY; 
	columnToCopy = blockIdx.x*realBlockSize + realThreadX;
	if(rowToCopy>=0&&rowToCopy<lsMagnitude.Height&&
		columnToCopy>=0&&columnToCopy<lsMagnitude.Width)
	{
		if(imageCache[threadIdx.y*32+threadIdx.x] < tau1)
		{
			result.SetAt(rowToCopy,columnToCopy,0.0f);
		}
		else
		{
			// annulus testing
			float sum = 0;
			for (int dx = -annulusRadius; dx <= annulusRadius; dx++)
			{
				for (int dy = -annulusRadius; dy <= annulusRadius; dy++)
				{
					sum += imageCache[(threadIdx.y+dy)*32+threadIdx.x+dx];
				}
			}
			if (sum / (annulusRadius * 2 + 1) * (annulusRadius * 2 + 1) < tau2) 
				result.SetAt(rowToCopy,columnToCopy,0.0f);
			else
			{
				float phase = lsPhase.At(rowToCopy, columnToCopy) / 2 - CUDART_PIO2_F;
				if (phase > CUDART_PI_F*39/40) phase -= CUDART_PI_F;
				if (phase < -CUDART_PI_F/40) phase += CUDART_PI_F;
				
				int direction = (int)round(phase / (CUDART_PI_F / 20));
				
				float avg = 0.0f;
				
				for (int i = 0; i < KernelSize; i++)
				{
					int x = constDirectionsX[i*DirectionsNumber+ direction];
					int y = constDirectionsY[i*DirectionsNumber+ direction];
					
					avg += constDirectionsKernel[i] * imageCache[32*(y+threadIdx.y)+x+threadIdx.x];
				}
				result.SetAt(rowToCopy, columnToCopy, avg);
			}
		}
	}
}

// CPU FUNCTIONS

void FillDirections()
{
	int* directionsX = (int*)malloc(sizeof(int)*DirectionsNumber*KernelSize);
	int* directionsY = (int*)malloc(sizeof(int)*DirectionsNumber*KernelSize);

	for (int n = 0; n < DirectionsNumber/2; n++)
	{
		float angle = CUDART_PI_F*n/DirectionsNumber;
		
		directionsX[(KernelSize/2)*DirectionsNumber + n] = 0;
		directionsY[(KernelSize/2)*DirectionsNumber + n] = 0;

		float tg = tan(angle);
		
		if (angle <= CUDART_PIO4_F)
		{
			for (int x = 1; x <= KernelSize/2; x++)
			{
				int y = (int)round(tg * x);
				directionsX[(KernelSize/2+x)*DirectionsNumber+ n] = x;
				directionsY[(KernelSize/2+x)*DirectionsNumber+ n] = y;
				directionsX[(KernelSize/2-x)*DirectionsNumber+ n] = -x;
				directionsY[(KernelSize/2-x)*DirectionsNumber+ n] = -y;
			}
		}
		else
		{
			for (int y = 1; y <= KernelSize/2; y++)
			{
				int x = (int)round((float)y/tg);

				directionsX[(KernelSize/2+y)*DirectionsNumber+ n] = x;
				directionsY[(KernelSize/2+y)*DirectionsNumber+ n] = y;
				directionsX[(KernelSize/2-y)*DirectionsNumber+ n] = -x;
				directionsY[(KernelSize/2-y)*DirectionsNumber+ n] = -y;
			}
		}
	}

	for (int n = DirectionsNumber/2; n < DirectionsNumber; n++)
	{
		for (int i = 0; i < KernelSize; i++)
		{
			int x = directionsX[i*DirectionsNumber+ n - 10];
			int y = directionsY[i*DirectionsNumber+ n - 10];
			
			directionsX[i*DirectionsNumber+ n - 10] = y;
			directionsY[i*DirectionsNumber+ n - 10] = -x;
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
		dim3(ceilMod(l.Width, defaultThreadCount-HalfSize*2),
		ceilMod(l.Height, defaultThreadCount-HalfSize*2));

	cudaDirectionalFiltering<<<gridSize, blockSize>>>(phase, lsReal, lsImaginary);

	error = cudaDeviceSynchronize();
}