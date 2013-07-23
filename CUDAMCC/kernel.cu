#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "ConvolutionHelper.h"

#define _USE_MATH_DEFINES
#include <math.h>

#define ceilMod(x, y) (x+y-1)/y
/*
const int R = 70;
const int Ns = 16; // 8
const int Nd = 6;
const double SigmaS = 28 / 3;
const double SigmaD = 2* M_PI / 9;
double deltaS;
double deltaD;
__device__ double deltaD;
const double MuPsi = 0.001;
const int BigSigma = 50;
const double MinVC = 0.75;
const double MinM = 2;
const double MinME = 0.6;
const double SigmaTetta = M_PI / 2;  
const double N = 10;
const double DictionaryCount = 360; // 720, 1440

struct Minutiae
{
	int x;
	int y;
	int numMinutiaeAround;
	float angle;
};

__global__ void cudaMCC (Minutiae* minutiae, CUDAArray<double> integralValues)
{
	//int row = defaultRow(); //
	//int column = defaultColumn(); //



}

__global__ void cudaMakeTableOfIntegrals(double* integralParameters, CUDAArray<double> integralValues, 
	bool* workingArea, double factor, double h)
{
	int column = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	double a = integralParameters[column] - deltaD / 2;
	double integrand = 0;
    double result = 0;

    for (int i = 0; i < N; i++)
    {
		integrand = a + ((2 * i + 1) * h) / 2;
		integrand = exp((-integrand * integrand) / (2 * SigmaD * SigmaD));
		result += h * integrand;    
	}
	
	integralValues.SetAt(1, column, result * factor);
}

double* InitializeIntegralParameters()
{
	double* integralParameters = (double*)malloc(DictionaryCount*sizeof(double));

	 double key = -M_PI;
	 double step = 2 * M_PI / DictionaryCount;

	 for (int i = 0; i < DictionaryCount; i++)
	 {
		 integralParameters[i] = key;
		 key += step;
	 }

	 return integralParameters;
}

void MCCMethod(Minutiae* minutiae, int minutiaeCount, int rows, int columns)
{
	/*
	cudaError_t cudaStatus = cudaSetDevice(0);

	double deltaS = 2 * R / Ns;
	double deltaD = 2 * M_PI / Nd;
	double* integralParameters = InitializeIntegralParameters();
	 
	CUDAArray<double> integralValues = CUDAArray<double>(DictionaryCount, 1);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		printf("integralValues = CUDAArray<double>(DictionaryCount, 1); - ERROR!!!\n");
	}

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(DictionaryCount, defaultThreadCount));
	CUDAArray<double> integralValues = CUDAArray<double>(DictionaryCount, 1);
		 
	cudaMakeTableOfIntegrals<<<gridSize,blockSize>>>(integralParameters, integralValues);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		printf("cudaMakeTableOfIntegrals - ERROR!!!\n");
	}

	double factor = 1 / (SigmaD * sqrt(2 * M_PI));
    double h = deltaD / N;

	//int[, ,] mask = new int[Constants.Ns, Constants.Ns, Constants.Nd];
		 
	gridSize = dim3(ceilMod(minutiaeCount, defaultThreadCount));
	cudaMCC<<<gridSize,blockSize>>>(minutiae, integralValues, factor, h);

    //return Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> response;
	*/ /*
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
*/
void main()
{
	// MinutiaDetectionSpecial.kernel.cu = > Minutiae *minutiae (array of Minutiae struct), int minutiaeCount(length of array)
	// CUDAConvexHull.BuildWorkingArea(int *field,int rows,int columns,int radius,int *IntMinutiae,int NoM);
	// workingArea = WorkingArea.BuildWorkingArea(minutiae, Constants.R, rows, columns);


	// CUDAArray<float> source = loadImage("C:\\temp\\103_4.bin");
	/* float* sourceFloat = source.GetData();

	 int imgWidth = source.Width;
	 int imgHeight = source.Height;
*/

	//MCCMethod(minutiae, minutiaeCount, rows, columns)
}
