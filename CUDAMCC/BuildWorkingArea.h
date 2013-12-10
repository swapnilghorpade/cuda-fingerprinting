#include "Point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//---------------------------------------------
Point FirstPoint = Point(0,0);

int compare(const void * a, const void * b) {
    int result = 1;
	Point V1 = *(Point*) a;
	Point V2 = *(Point*) b;
    if ((V1.VectorProduct(V2) < 0) ||
        ((V1.VectorProduct(V2) == 0) &&
		(V1.Distance(FirstPoint) < V2.Distance(FirstPoint))))
        result = -1;
	if (V1.Equals(V2))
        result = 0;
    return result;
}

void Sort(Point *arr, int N) {
	for (int i = 0 ; i < N; i++) {
		if (arr[i].Y > (FirstPoint).Y || 
			arr[i].Y == (FirstPoint).Y && arr[i].X < (FirstPoint).X)
            FirstPoint = arr[i];
    }

	for (int i = 0 ; i < N; i++) arr[i] = arr[i].Subtract(FirstPoint);

	qsort(arr, N, sizeof(Point), compare);
}

void BuildHull(Point *arr, int number, Point *Hull,int *NHull) {

	Sort(arr, number);

	Hull[0] = arr[0];
	Hull[1] = arr[1];
	int top = 1;
	int nextToTop = 0;
    for (int i = 2; i < number; i++)    
	{
		while ((arr[i].Subtract(Hull[nextToTop]).VectorProduct(Hull[top].Subtract(Hull[nextToTop])) <=0) && (!Hull[top].Equals(FirstPoint))) {
			 top--;
			 if (Hull[top].Equals(FirstPoint))
                 nextToTop = top;
             else
                 nextToTop = top-1;
         }
         top++;
		 nextToTop = top-1;
		 Hull[top] = arr[i];
    }
	*NHull = top+1;
}
//----------------------------------------



__global__ void Fill(int *dev_field, Point *dev_Hull, int NHull) {
	int curPoint = blockIdx.x * blockDim.x + threadIdx.x;

	int result = 1;

	for (int i = NHull-1; (i>0) && dev_field[curPoint]; i--)
	{
		if ((dev_Hull[i-1].X-dev_Hull[i].X)*((int)(threadIdx.x) - dev_Hull[i].Y) - (dev_Hull[i-1].Y-dev_Hull[i].Y)*((int)(blockIdx.x) - dev_Hull[i].X) < 0)
			result = 0;
	}
	if ((dev_Hull[NHull-1].X-dev_Hull[0].X)*((int)(threadIdx.x) - dev_Hull[0].Y) - (dev_Hull[NHull-1].Y-dev_Hull[0].Y)*((int)(blockIdx.x) - dev_Hull[0].X) < 0)
			result = 0;

	dev_field[curPoint] = result;
}

void FieldFilling(int *field,int rows, int columns,int *minutiaeXs, int* minutiaeYs, int number) {
	int NHull;
	Point* Hull = (Point*) malloc (number *sizeof(Point));

	Point* minutiae = (Point*) malloc (number *sizeof(Point));
	for(int i=0; i<number; i++)
	{
		minutiae[i].X = minutiaeXs[i];
		minutiae[i].Y = minutiaeYs[i];
	}

	BuildHull(minutiae, number, Hull, &NHull);

	int *dev_field;

	cudaMalloc(&dev_field,(rows*columns)*sizeof(int));

	Point *dev_Hull;

	cudaMalloc(&dev_Hull, number * sizeof(Point));

	cudaMemcpy(dev_Hull,Hull,(size_t)(number * sizeof(Point)), cudaMemcpyHostToDevice);

    Fill<<<rows,columns>>>(dev_field,dev_Hull,NHull);

	cudaMemcpy(field,dev_field,(size_t)((rows*columns) * sizeof(int)), cudaMemcpyDeviceToHost);

	cudaFree(dev_field);

	cudaFree(dev_Hull);

	free(Hull);
}

//----------------------------------------
__device__ int fmax(int a,int b) {
	int result = a;
	if (a < b)
		result = b;
	return result;
}


__device__ int fmin(int a,int b) {
	int result = a;
	if (a > b)
		result = b;
	return result;
}
//----------------------------------------
//blockId.x - number of a row
//threadId.x - number of a column
__global__ void FindArea(int *dev_field,int *dev_NewField,int radius,int rows,int columns) {

	int curPoint = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = fmax(blockIdx.x - radius, 0); (i <= fmin(rows - 1, blockIdx.x + radius)) && (!dev_NewField[curPoint]); i++)
	{
		for (int j = fmax(threadIdx.x - radius, 0); (j <= fmin(columns - 1, threadIdx.x+radius)) && (!dev_NewField[curPoint]); j++) 
		{
			if ((threadIdx.x - j)*(threadIdx.x-j) + (blockIdx.x-i)*(blockIdx.x-i) <= radius * radius)
				if (dev_field[i * blockDim.x + j])
					dev_NewField[curPoint] = 1;
		}
	}
}


void BuildWorkingArea(int *field,int rows,int columns,int radius,int *minutiaeXs, int *minutiaeYs, int number) {
	FieldFilling(field, rows, columns, minutiaeXs, minutiaeYs, number);
	
	int *dev_field;
	int *dev_NewField;
	cudaMalloc(&dev_field,rows*columns*sizeof(int));
	cudaMalloc(&dev_NewField,rows*columns*sizeof(int));
	cudaMemcpy(dev_field,field,(size_t)(rows * columns * sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_NewField,field,(size_t)(rows * columns * sizeof(int)), cudaMemcpyHostToDevice);
	FindArea<<<rows,columns>>>(dev_field,dev_NewField,radius,rows,columns);
	cudaMemcpy(field,dev_NewField,(size_t)(rows * columns * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaFree(dev_field);
	cudaFree(dev_NewField);
}
//----------------------------------------