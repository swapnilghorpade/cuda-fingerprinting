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

__host__ __device__ __inline__ bool OnTheSameSide(float x1, float y1, float x2, float y2, 
            float a, float b, float c)
{
	float result1 = a*x1 + b*y1 + c;
	result1 = result1 / abs(result1);
	float result2 = a*x2 + b*y2 + c;
	result2 = result2 / abs(result2);
	return result1 * result2 > 0;
}

void IntersectionPoint(float a1, float b1, float c1,
	float a2, float b2, float c2, float* x, float* y)
{
	float det = a1*b2 - a2*b1;
	*x = (-b2*c1 + b1*c2)/det;
	*y = (-a1*c2 + a2*c1)/det;
}

__host__ __device__ __inline__ void DetermineLine(int x1, int y1, int x2, int y2, float* a1, float* b1, float* c1)
{
	// determine line coefficients
	*a1 = y1 - y2;
	*b1 = x2 - x1;
	*c1 = x1 * y2 - x2 * y1;

	if (*a1 != 0)
	{
		*c1 /= *a1;
		*b1 /= *a1;
		*a1 = 1.0f;
	}
	else
	{
		*c1 /= *b1;
		*a1 /= *b1;
		*b1 = 1.0f;
	}
}

void DetermineNewPoint(int x1, int y1, int x2, int y2, int x3, int y3, int* x0, int* y0, float radius)
{
	float a1, b1, c1;
	DetermineLine(x1,y1,x2,y2, &a1, &b1, &c1);
	float a2, b2, c2;
	DetermineLine(x2, y2, x3, y3, &a2, &b2,  &c2);

	float c11 = c1 + sqrt(a1 * a1 + b1 * b1) * radius;
	float c12 = c1 - sqrt(a1 * a1 + b1 * b1) * radius;
	float c21 = c2 + sqrt(a2 * a2 + b2 * b2) * radius;
	float c22 = c2 - sqrt(a2 * a2 + b2 * b2) * radius;

	float x, y;
	IntersectionPoint(a1, b1, c11, a2, b2, c21, &x, &y);
	if (!OnTheSameSide(x1, y1, x, y, a2, b2, c2) && !OnTheSameSide(x3, y3, x, y, a1, b1, c1))
	{
		*x0 = (int)x;
		*y0 = (int) y;
		return;
	}

	IntersectionPoint(a1, b1, c12, a2, b2, c21, &x, &y);
	if (!OnTheSameSide(x1, y1, x, y, a2, b2, c2) && !OnTheSameSide(x3, y3, x, y, a1, b1, c1))
	{
		*x0 = (int)x;
		*y0 = (int)y;
		return;
	}

	IntersectionPoint(a1, b1, c11, a2, b2, c22, &x, &y);
	if (!OnTheSameSide(x1, y1, x, y, a2, b2, c2) && !OnTheSameSide(x3, y3, x, y, a1, b1, c1))
	{
		*x0 = (int)x;
		*y0 = (int)y;
		return;
	}

	IntersectionPoint(a1, b1, c12, a2, b2, c22, &x, &y);
	*x0 = (int)x;
	*y0 = (int)y;
}

__global__ void fillConvexHull(CUDAArray<int> field, CUDAArray<int> xs, CUDAArray<int> ys)
{
	int y = defaultRow();
	int x = defaultColumn();

	if(y < field.Height && x < field.Width)
	{
		int result = 1;

		int x1 = xs.At(0,0);
		int y1 = ys.At(0,0);

		int x2 = xs.At(0,1);
		int y2 = ys.At(0,1);

		int x3 = 0;
		int y3 = 0;

		for (int i = 0; i < xs.Width; i++)
		{
			x3 = xs.At(0, (i+2)%3);
			y3 = ys.At(0, (i+2)%3);

			float a, b, c;
			a = y1 - y2;
	b = x2 - x1;
	c = x1 * y2 - x2 * y1;

	if (a != 0)
	{
		c /= a;
		b /= a;
		a = 1.0f;
	}
	else
	{
		c /= b;
		a /= b;
		b = 1.0f;
	}
			if (!OnTheSameSide(x, y, x3, y3, a, b, c))
			{
				result = 0;
				break;
			}
			x1 = x2;
			y1 = y2;

			x2 = x3;
			y2 = y3;
		}

		field.SetAt(y, x, result);
	}
}


int* BuildAreaOfInterest(int rows,int columns,int radius,int *minutiaeXs, int *minutiaeYs, int number) {

	int NHull;
	Point* Hull = (Point*) malloc (number *sizeof(Point));

	Point* minutiae = (Point*) malloc (number *sizeof(Point));
	for(int i=0; i<number; i++)
	{
		minutiae[i].X = minutiaeXs[i];
		minutiae[i].Y = minutiaeYs[i];
	}

	BuildHull(minutiae, number, Hull, &NHull);

	int* xs = (int*)malloc(sizeof(int)*NHull);
	int* ys = (int*)malloc(sizeof(int)*NHull);

	for (int i = 0; i < NHull; i++)
	{
		int x;
		int y;
		DetermineNewPoint(Hull[(i + 2) % 3].X, Hull[(i + 2) % 3].Y, Hull[i].X, Hull[i].Y, Hull[(i + 1) % 3].X,
			Hull[(i + 1)%3].Y, &x, &y, radius);
		xs[i] = x+FirstPoint.X;
		ys[i] = y+FirstPoint.Y;
	}
	
	CUDAArray<int> cudaXs = CUDAArray<int>(xs, NHull, 1);
	cudaError_t error = cudaGetLastError();
	CUDAArray<int> cudaYs = CUDAArray<int>(ys, NHull, 1);
	error = cudaGetLastError();
	CUDAArray<int> cudaField = CUDAArray<int>(columns, rows);
	error = cudaGetLastError();

	dim3 blockSize = dim3(16,16);
	dim3 gridSize = 
		dim3(ceilMod(cudaField.Width,16),
		ceilMod(cudaField.Height,16));

	fillConvexHull<<<gridSize, blockSize>>>(cudaField, cudaXs, cudaYs);

	error = cudaGetLastError();

	int* result = cudaField.GetData();
	//SaveArray(cudaField, "C:\\temp\\field.bin");
	cudaXs.Dispose();
	cudaYs.Dispose();
	cudaField.Dispose();
	
	return result;
}
//----------------------------------------