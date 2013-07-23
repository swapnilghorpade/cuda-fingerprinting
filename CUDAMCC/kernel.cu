#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "CUDAArray.h";
#include "ConvolutionHelper.h"

#define _USE_MATH_DEFINES
#include <math.h>

#define ceilMod(x, y) (x+y-1)/y

const int R = 70;
const int Ns = 16; // 8
const int Nd = 6;
const double SigmaS = 28 / 3;
const double SigmaD = 2* M_PI / 9;
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

void MCCMethod(Minutiae *minutiae, int minutiaeCount, int rows, int columns)
{
	/*cudaError_t cudaStatus = cudaSetDevice(0);

	double deltaS = 2 * R / Ns;
	double deltaD = 2 * M_PI / Nd;
	double* integralParameters = (double*)malloc(DictionaryCount*sizeof(double));

	for (int i = 0; i < DictionaryCount; i++)
	{
		integralParameters[i] = 
	}

	InitialIntegralValues();

	CUDAArray<double> integralValues = CUDAArray<double>(DictionaryCount, 1);
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(DictionaryCount, defaultThreadCount));

	MakeTableOfIntegrals()*/

}

void main()
{
	// MinutiaDetectionSpecial.kernel.cu = > Minutiae *minutiae (array of Minutiae struct), int minutiaeCount(length of array)
	// CUDAConvexHull.BuildWorkingArea(int *field,int rows,int columns,int radius,int *IntMinutiae,int NoM);
}
