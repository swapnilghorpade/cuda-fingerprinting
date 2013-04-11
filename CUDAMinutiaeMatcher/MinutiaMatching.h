#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "DirectionalFiltering.h"

// GPU FUNCITONS

__constant__ int constX1[32][32];
__constant__ int constY1[32][32];
__constant__ float constAngle1[32][32];
__constant__ float constLength1[32][32];

__device__ __host__ float DetermineLength(int dx, int dy)
{
	return sqrt((float)(dx*dx+dy*dy));
}

__global__ void MatchMinutiae(CUDAArray<int> result, CUDAArray<int> X2, CUDAArray<int> Y2)
{
	__shared__ int x2[32][32];
	__shared__ int y2[32][32];

	__shared__ float length2[32][32];

	__shared__ float angle2[32][32];

	__shared__ unsigned int taskCount;
	__shared__ int tasks[5000];
	__shared__ int totalMax;
	//each shared row corresponds to the fprint centered at its index's minutia
	int dx = X2.At(blockIdx.x,threadIdx.x);
	x2[threadIdx.x][threadIdx.y] = X2.At(blockIdx.x,threadIdx.y)-dx;
	dx = Y2.At(blockIdx.x,threadIdx.x);
	y2[threadIdx.x][threadIdx.y] = Y2.At(blockIdx.x,threadIdx.y)-dx;

	length2[threadIdx.x][threadIdx.y] = DetermineLength(x2[threadIdx.x][threadIdx.y], y2[threadIdx.x][threadIdx.y]);

	angle2[threadIdx.x][threadIdx.y] = atan2((float)-y2[threadIdx.x][threadIdx.y],(float)x2[threadIdx.x][threadIdx.y]);

	if(threadIdx.x==0&&threadIdx.y==0)
	{
		taskCount =0;
		totalMax =0;
	}

	__syncthreads();

	//// now threadidx.x is the row for the 1st, threadidx.y - for second
	for(int i=0;i<32;i++)
	{
		for(int j=0;j<32;j++)
		{
			if (abs(constLength1[threadIdx.x][i] - length2[threadIdx.y][j]) < 3.0f
				&&abs(constAngle1[threadIdx.x][i] - angle2[threadIdx.y][j]) < CUDART_PIO4_F/2)
			{
				unsigned int localIndex = atomicInc(&taskCount ,100500);
				int value = (threadIdx.x<<24)|(i<<16)|(threadIdx.y<<8)|j;
				tasks[localIndex] = value;
			}
		}
	}

	__syncthreads();

	int maxCount =0;

	int limit = taskCount / 1024 + 1;
	int k = -1;
	for(int i=0; i<limit; i++)
	{
		int index = i*1024+threadIdx.x*32+threadIdx.y;
		if(index < taskCount)
		{
			int task = tasks[index];
			int m1From = task>>24;
			int m1To = (task>>16)&8;
			int m2From = (task>>8)&255;
			int m2To = task&255;
			float da = angle2[m2From][m2To] - constAngle1[m1From][m1To];
			float cosine = cos(da);
			float sine = -sin(da);
			int mask = 0;
			int count=0;
			for(int m =0; m<32;m++)
			{
				float xDash = cosine * constX1[m1From][m] - sine * constY1[m1From][m];
				float yDash = sine * constX1[m1From][m] + cosine * constY1[m1From][m];

				int nMax = -1;
				float dMax = 36.0f;

				for(int n=0;n<32;n++)
				{
					float dX = xDash - x2[m2From][n];
					float dY = yDash - y2[m2From][n];
					float d = dX*dX+dY*dY;

					if(d<dMax&&((mask&(1<<n))==0))
					{
						dMax = d;
						nMax = n;
					}
				}

				if(nMax!=-1)
				{
					mask = mask | (1<<nMax);
					count++;
				}

				tasks[index] = count;
			}
		}
	}

	__syncthreads();

	if(threadIdx.x==0&&threadIdx.y==0)
	{
		int absolutemax = 0;
		for(int i=0;i<taskCount;i++)
		{
			if(tasks[i]>absolutemax)
			{
				absolutemax = tasks[i];
			}
		}
			
		result.SetAt(0,blockIdx.x,absolutemax);
	}
}

// CPU FUNCTIONS

void MatchFingers(int* x1, int* y1, CUDAArray<int> x2, CUDAArray<int> y2)
{
	// prepare zee target fingerprint
	cudaError_t error;

	int* localX1 = (int*)malloc(32*32*sizeof(int));
	int* localY1 = (int*)malloc(32*32*sizeof(int));
	float* localLength1 = (float*)malloc(32*32*sizeof(float));
	float* localAngle1 = (float*)malloc(32*32*sizeof(float));

	for(int from = 0;from<32;from++)
	{
		int dx = x1[from];
		int dy = y1[from];

		for(int i=0;i<32;i++)
		{
			localX1[from*32+i] = x1[i]-dx;
			localY1[from*32+i] = y1[i]-dy;
			localLength1[from*32+i] = DetermineLength(x1[i]-dx, y1[i]-dy);
			localAngle1[from*32+i] = atan2((float)-(y1[i]-dy),(float)x1[i]-dx);
		}
	}

	error = cudaMemcpyToSymbol(constX1, localX1, sizeof(int)*32*32);
	error = cudaMemcpyToSymbol(constY1, localY1, sizeof(int)*32*32);
	error = cudaMemcpyToSymbol(constLength1, localLength1, sizeof(float)*32*32);
	error = cudaMemcpyToSymbol(constAngle1, localAngle1, sizeof(float)*32*32);

	int n = 1000;
	CUDAArray<int> result = CUDAArray<int>(n,1);

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(n,1);

	MatchMinutiae<<<gridSize, blockSize>>>(result, x2,y2);

	error = cudaDeviceSynchronize();
	cudaError_t error2 = cudaGetLastError();
	int* res = result.GetData();
	int m =0;
	for(int i=0;i<n;i++)if(res[i]>m)m=res[i];
	result.Dispose();
}