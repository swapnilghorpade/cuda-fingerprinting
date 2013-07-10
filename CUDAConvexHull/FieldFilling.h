#include "Point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma once
//blockId.x - number of a row
//threadId.x - number of a column
__global__ void Fill(bool *dev_field,Point *dev_Hull,int NHull) {
	int curPoint = blockIdx.x * blockDim.x + threadIdx.x;
	dev_field[curPoint] = true;
	for (int i = NHull-1; (i>0) && (dev_field[curPoint]); i--)
		if ((dev_Hull[i-1].X-dev_Hull[i].X)*((int)(threadIdx.x) - dev_Hull[i].Y) - (dev_Hull[i-1].Y-dev_Hull[i].Y)*((int)(blockIdx.x) - dev_Hull[i].X) < 0)
			dev_field[curPoint] = false;
	if ((dev_Hull[NHull-1].X-dev_Hull[0].X)*((int)(threadIdx.x) - dev_Hull[0].Y) - (dev_Hull[NHull-1].Y-dev_Hull[0].Y)*((int)(blockIdx.x) - dev_Hull[0].X) < 0)
			dev_field[curPoint] = false;
}

void FieldFilling(bool *field,int rows, int columns,int *arr, int N) {
	cudaSetDevice(0);
	Point *Hull = (Point*) malloc (N * sizeof(Point));
	for (int i = 0 ; i < N ; i++)
		Hull[i] = Point(arr[2*i],arr[2*i+1]);
	//Hull = (Point*) arr;
	bool *dev_field;
	cudaMalloc(&dev_field,(rows*columns)*sizeof(bool));
	Point *dev_Hull;
	cudaMalloc(&dev_Hull,N*sizeof(Point));
	cudaMemcpy(dev_Hull,Hull,(size_t)(N * sizeof(Point)), cudaMemcpyHostToDevice);
    Fill<<<rows,columns>>>(dev_field,dev_Hull,N);
	cudaMemcpy(field,dev_field,(size_t)((rows*columns) * sizeof(bool)), cudaMemcpyDeviceToHost);
	cudaFree(dev_field);
	cudaFree(dev_Hull);
	free(Hull);
	cudaDeviceReset();
}
