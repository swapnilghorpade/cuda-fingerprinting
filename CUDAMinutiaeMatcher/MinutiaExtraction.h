#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "MinutiaMatching.h"

__device__ const float tauPS = 0.1f;
__device__ const int NeighborhoodSize = 9;
__device__ const float tauLS = 0.9f;

// GPU FUNCTIONS

__global__ void cudaEstimateMeasure(CUDAArray<float> psiReal, CUDAArray<float> psiIm, 
	CUDAArray<float> lsM, CUDAArray<float> psReal, CUDAArray<float> psIm)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(lsM.Width>column&&lsM.Height>row)
	{
		float lsValue = lsM.At(row,column);
		float psImValue = psIm.At(row,column);
		float psRealValue = psReal.At(row,column);
		
		float psMagn = psImValue*psImValue+psRealValue*psRealValue;

		float psiRealValue = psRealValue*(1.0f-lsValue);
		float psiImValue = psImValue*(1.0f-lsValue);

		psiReal.SetAt(row,column,psiRealValue);
		psiIm.SetAt(row,column,psiImValue);
	}
}

__global__ void voteForTheMax(CUDAArray<unsigned int> votes, CUDAArray<float> psim)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(psim.Width>column&&psim.Height>row)
	{
		int maxRow = 0;
		int maxColumn = 0;
		float maxM = tauPS; // a hacky way to get only values above the treshold
	
		for (int dRow = -NeighborhoodSize/2; dRow <= NeighborhoodSize/2; dRow++)
		{
			for (int dColumn = -NeighborhoodSize / 2; dColumn <= NeighborhoodSize / 2; dColumn++)
			{
				int correctRow = row + dRow < 0
					? 0
					: (row + dRow >= psim.Height ? psim.Height - 1 : row + dRow);
				int correctColumn = column + dColumn < 0
				                                     ? 0
			                                         : (column + dColumn >= psim.Width ? psim.Width - 1 : column + dColumn);
				
				float value = psim.At(correctRow, correctColumn);
				if (value > maxM)
				{
					maxM = value;
					maxRow = correctRow;
					maxColumn = correctColumn;
				}
			}
		}
		//todo: check specifically
		if(maxRow>0&&maxColumn>0&&maxRow<votes.Height-1&&maxColumn<votes.Width-1)
			atomicInc(votes.cudaPtr+maxRow*votes.Stride+sizeof(int)*maxColumn,1000);
	}
}

__global__ void calculateMinutiaMetrics(CUDAArray<unsigned int> metrics, CUDAArray<unsigned int> votes, CUDAArray<float> lsm)
{
	int row = defaultRow();
	int column = defaultColumn();

	if(lsm.Width>column&&lsm.Height>row&&votes.At(row,column) > 20)
	{
		float sum = 0;
		int count = 0;

		for (int dRow = -ringOuterRadius; dRow <= ringOuterRadius; dRow++)
		{
			for (int dColumn = -ringOuterRadius; dColumn <= ringOuterRadius; dColumn++)
			{
				if (abs(dRow) < ringInnerRadius && abs(dColumn) < ringInnerRadius) continue;
				count++;
				int correctRow = row + dRow < 0
					? 0
					: (row + dRow >= lsm.Height ? lsm.Height - 1 : row + dRow);
				int correctColumn = column + dColumn < 0
					? 0
					: (column + dColumn >= lsm.Width ? lsm.Width - 1 : column + dColumn);
				sum += lsm.At(correctRow, correctColumn);

			}
		}

		if (sum / count > tauLS)
		{
			atomicInc(metrics.cudaPtr+row*metrics.Stride+sizeof(unsigned int)*column,1000);
		}
	}
}

// CPU FUNCTIONS

void EstimateMeasure(CUDAArray<float> psiReal, CUDAArray<float> psiIm, 
	CUDAArray<float> lsM, CUDAArray<float> psReal, CUDAArray<float> psIm)
{
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(psiReal.Width, defaultThreadCount),
		ceilMod(psiReal.Height, defaultThreadCount));
	cudaEstimateMeasure<<<gridSize,blockSize>>>(psiReal, psiIm, lsM, psReal, psIm);
}

int compare( const void *arg1, const void *arg2 )
{
   return *((float*)arg2)-*((float*)arg1);
}

void ExtractMinutiae(int* xs, int* ys, CUDAArray<float> source)
{
	cudaError error;
	CUDAArray<float> psReal;
	CUDAArray<float> psIm;
	

	CUDAArray<float> lsReal;
	CUDAArray<float> lsIm;
	CUDAArray<float> lsM=CUDAArray<float>(source.Width,source.Height);
	EstimateLS(&lsReal, &lsIm, source, 0.6f, 3.2f);
	

	EstimatePS(&psReal, &psIm, source, 0.6f, 3.2f);
	GetMagnitude(lsM, lsReal, lsIm);
	

	CUDAArray<float> psiReal = CUDAArray<float>(lsM.Width, lsM.Height);
	CUDAArray<float> psiIm = CUDAArray<float>(lsM.Width, lsM.Height);
	CUDAArray<float> psiM = CUDAArray<float>(lsM.Width, lsM.Height);
	EstimateMeasure(psiReal, psiIm,lsM, psReal, psIm);

	GetMagnitude(psiM, psiReal, psiIm);
	
	error = cudaGetLastError();

	CUDAArray<unsigned int> votes = CUDAArray<unsigned int>(lsM.Width, lsM.Height);
	CUDAArray<unsigned int> metrics = CUDAArray<unsigned int>(lsM.Width, lsM.Height);
	cudaMemset2D(votes.cudaPtr, votes.Stride, 0, votes.Width*sizeof(unsigned int), votes.Height);

	cudaMemset2D(metrics.cudaPtr, metrics.Stride, 0, metrics.Width*sizeof(unsigned int), metrics.Height);

	error = cudaGetLastError();

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(psiReal.Width, defaultThreadCount),
		ceilMod(psiReal.Height, defaultThreadCount));
	voteForTheMax<<<gridSize, blockSize>>>(votes, psiM);

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	SaveArray(votes, "C:\\temp\\grid_gpu.bin");
	unsigned int* metricsLocal = metrics.GetData();
	//unsigned int* votesLocal = votes.GetData();
	//for(int i=0;i<votes.Height*votes.Width;i++)
	//{
	//	if(votesLocal[i]!=0)
	//		printf("%d\n",votesLocal[i]);
	//}
	calculateMinutiaMetrics<<<gridSize, blockSize>>>(metrics, votes, lsM);

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	metricsLocal = metrics.GetData();

		//unsigned int* votesLocal = votes.GetData();
	for(int i=0;i<votes.Height*votes.Width;i++)
	{
		if(metricsLocal[i]!=0)
			printf("%d\n",metricsLocal[i]);
	}

	lsM.Dispose();
	lsReal.Dispose();
	lsIm.Dispose();
	psReal.Dispose();
	psIm.Dispose();
	psiReal.Dispose();
	psiIm.Dispose();
	votes.Dispose();
	lsIm.Dispose();
	lsReal.Dispose();
	psIm.Dispose();
	psReal.Dispose();

	float* psimLocal = psiM.GetData();
	psiM.Dispose();
	int count = 0;

	for(int row=1;row<metrics.Height-1;row++)
	{
		for(int column=1;column<metrics.Width-1;column++)
		{
			if(metricsLocal[row*metrics.Width+column])count++;
		}
	}

	int* minutiaeCache = (int*)malloc(count*(2*sizeof(int)+sizeof(float)));
	int cacheCount = 0;

	for(int row=1;row<metrics.Height-1;row++)
	{
		for(int column=1;column<metrics.Width-1;column++)
		{
			if(metricsLocal[row*metrics.Width+column])
			{
				bool valid = true;
				for(int i=0;i<cacheCount;i++)
				{
					int cacheColumn = minutiaeCache[3*i+1];
					int cacheRow = minutiaeCache[3*i+2];
					if((cacheRow - row)*(cacheRow - row)+(cacheColumn - column)*(cacheColumn - column) <30)
					{
						valid = false;
						break;
					}
				}

				if(valid)
				{
					minutiaeCache[cacheCount+1] = column;
					minutiaeCache[cacheCount+2] = row;
					((float*)minutiaeCache)[cacheCount++] = psimLocal[row*metrics.Width+column];
				}
			}
		}
	}

	qsort(minutiaeCache, cacheCount, sizeof(float)+sizeof(int)*2,compare);

	xs = (int*)malloc(sizeof(int)*32);
	ys = (int*)malloc(sizeof(int)*32);

	for(int i=0;i<32;i++)
	{
		xs[i] = minutiaeCache[3*i+1];
		ys[i] = minutiaeCache[3*i+2];
	}

	metrics.Dispose();
	free(metricsLocal);
	free(psimLocal);
	free(minutiaeCache);
}