#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionHelper.h"
#include "BuildWorkingArea.h"
#include "FindMinutiae.h"

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
const int MinM = 2;
const double MinME = 0.6;
const double SigmaTetta = M_PI / 2;  
const int N = 10;
const int DictionaryCount = 360; // 720, 1440

__device__ const int cudaR = 70;
__device__ const int cudaNs = 16; // 8
__device__ const int cudaNd = 6;
__device__ const double cudaSigmaS = 28 / 3;
__device__ const double cudaSigmaD = 2* M_PI / 9;
__device__ const double cudaMuPsi = 0.001;
__device__ const int cudaBigSigma = 50;
__device__ const double cudaMinVC = 0.75;
__device__ const int cudaMinM = 2;
__device__ const double cudaMinME = 0.6;
__device__ const double cudaSigmaTetta = M_PI / 2;  
__device__ const int cudaN = 10;
__device__ const int cudaDictionaryCount = 360; // 720, 1440

__host__ __device__ struct Cell
{
	Minutiae minutia;
	int value;
	int mask;
};

__device__ int Psi(double v)
{
    if (v >= cudaMuPsi) 
	{
		return 1;
	}

	return 0;
}

__device__ double GetIntegralParameterIndex(double param, CUDAArray<double> integralParameters)
{
    for (int i = 0; i < cudaDictionaryCount; i++)
    {
        if (abs(integralParameters.At(0, i) - param) <= M_PI / cudaDictionaryCount)
        {
            return i;
        }
    }

	// never use
	return 0;
}

__device__ double GetAngleFromLevel(int k, double deltaD)
{
    return M_PI + (k - 1 / 2) * deltaD;
}

__device__ double NormalizeAngle(double angle)
{
    if (angle < -M_PI)
    {
        return 2 * M_PI + angle;
    }

    if (angle < M_PI)
    {
        return angle;
    }

    return -2 * M_PI + angle;
}

__device__ double GetDirectionalContribution(double mAngle, double mtAngle, int k, CUDAArray<double> integralParameters, CUDAArray<double> integralValues, double deltaD)
{
    double angleFromLevel = NormalizeAngle(GetAngleFromLevel(k, deltaD));
    double differenceAngles = NormalizeAngle(mAngle - mtAngle);
    double param = NormalizeAngle(angleFromLevel - differenceAngles);
	int indexOfResult = GetIntegralParameterIndex(param, integralParameters);

    return integralValues.At(0, indexOfResult);
}

__device__ double GetDistance(int x1, int y1, int x2, int y2)
{
	return sqrt((double)((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

__device__ double GetSpatialContribution(Minutiae m, int currentCellX, int currentCellY)
{
	double distance = GetDistance(m.x, m.y, currentCellX, currentCellY);

	double commonDenom = cudaSigmaS * cudaSigmaS * 2;
    double denominator = cudaSigmaS * sqrt(M_PI * 2);
    double result = exp(-(distance * distance) / commonDenom) / denominator;

    return result;
}

__device__ bool IsEqualMinutiae(Minutiae m1, Minutiae m2)
{
	return m1.x == m2.x && m1.y == m2.y && m1.angle == m2.angle;
}

__device__ void GetCoordinatesInFingerprint(int& pointX, int& pointY, Minutiae m, int i, int j, double deltaS)
{
	double halfNs = (1 + cudaNs) / 2;
	double sinTetta = sin(m.angle);
	double cosTetta = cos(m.angle);
	double iDelta = cosTetta * (i - halfNs) + sinTetta * (j - halfNs);
	double jDelta = -sinTetta * (i - halfNs) + cosTetta * (j - halfNs);
    
	pointX = (int)(m.x + deltaS * iDelta);
	pointY = (int)(m.y + deltaS * jDelta);
}

__device__ int CalculateMaskValue(Minutiae m, int i, int j, int rows, int columns, CUDAArray<int> workingArea, double deltaS)
{
	int pointX = 0; 
	int pointY = 0; 
	
	GetCoordinatesInFingerprint(pointX, pointY, m, i, j, deltaS);
            
	if (pointX < 0 || pointX >= rows ||
		pointY < 0 || pointY >= columns)
	{
		return 0;
	}
            
	return ((GetDistance(m.x, m.y, pointX, pointY) <= R) && (workingArea.At(pointY, pointX) == 1))
		? 1
		: 0;
}

__global__ void GetallNeighbours(Minutiae* minutiae, CUDAArray<int> numOfNeighbours, int numOfMinutiae)
{
	int sum = 0;
	int current = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	for (int i = 0; i < numOfMinutiae; i++)
	{
		if(GetDistance(minutiae[i].x,minutiae[i].y, minutiae[current].x,minutiae[current].y)<=(R+3*cudaSigmaS)
			&& (minutiae[i].x!=minutiae[current].x || minutiae[i].y!=minutiae[current].y))
		{
			sum ++;
		}
	}
	numOfNeighbours.SetAt(0, current, sum);
}

__global__ void cudaMCC (CUDAArray<Minutiae> minutiae, int minutiaeCount, CUDAArray<double> integralParameters, 
	CUDAArray<double> integralValues, int rows, int columns, CUDAArray<int> workingArea, double deltaS, double deltaD, CUDAArray<Cell> arr)
{
	Cell result;
	int row = defaultRow(); // J  < ~80
	int column = defaultColumn(); // I  < 1536 = 16*16*6
	
	int index = row;
	int i = column % ((cudaNs - 1) * (cudaNd - 1));
	int j = (column - i*((cudaNs - 1) * (cudaNd - 1))) % (cudaNd - 1);
	int k = (column - i*((cudaNs - 1) * (cudaNd - 1))) - j * (cudaNd - 1);

	int coordinateX = 0; 
	int coordinateY = 0; 
	double psiParameter = 0;
	double spatialContribution = 0;
	double directionalContribution = 0;

	GetCoordinatesInFingerprint(coordinateX, coordinateY, minutiae.At(0, index), i, j, deltaS);

	if (coordinateX < 0 || coordinateX >= rows ||
		coordinateY < 0 || coordinateY >= columns)
	{
		result.minutia = minutiae.At(0, index);
		result.mask = 0;
		result.value = 0;
		arr.SetAt(row, column, result);

		return;
	}

	for (int i = 0; i < minutiaeCount; i++)
	{
		if (!IsEqualMinutiae(minutiae.At(0, i), minutiae.At(0, index)) && 
			GetDistance((minutiae.At(0, i)).x, (minutiae.At(0, i)).y, coordinateX, coordinateY) <= 3 * cudaSigmaS)
		{
			spatialContribution = GetSpatialContribution(minutiae.At(0, i), coordinateX, coordinateY);
			directionalContribution = GetDirectionalContribution((minutiae.At(0, index)).angle, (minutiae.At(0, i)).angle, k, integralParameters, integralValues, deltaD);
			psiParameter = psiParameter + spatialContribution * directionalContribution;
		}
	}

	int psiValue = Psi(psiParameter); 
	int maskValue = CalculateMaskValue(minutiae.At(0, index), i, j, rows, columns, workingArea, deltaS);
	
	result.minutia = minutiae.At(0, index);
	result.mask = maskValue;
	result.value = psiValue;
	arr.SetAt(row, column, result);

	// check

	//allNeighbours.AddRange(neighbourMinutiae);
	//allNeighbours = MinutiaListDistinct(allNeighbours);
	
	//if (allNeighbours.Count < Constants.MinM && !IsValidMask())
	//{
	//	continue;
	//}

	//response.Add(minutiae[index], new Tuple<int[, ,], int[, ,]>(value, mask));
}

__global__ void cudaMakeTableOfIntegrals(CUDAArray<double> integralParameters, CUDAArray<double> integralValues, double factor, double h, double deltaD)
{
	int column = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	double a = integralParameters.At(0, column) - deltaD / 2;
	double integrand = 0;
    double result = 0;

    for (int i = 0; i < cudaN; i++)
    {
		integrand = a + ((2 * i + 1) * h) / 2;
		integrand = exp((-integrand * integrand) / (2 * cudaSigmaD * cudaSigmaD));
		result += h * integrand;    
	}
	
	integralValues.SetAt(0, column, result * factor);
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

void MCCMethod(Cell* result, Minutiae* minutiae, int minutiaeCount, int rows, int columns, int* workingArea)
{
	cudaError_t cudaStatus = cudaSetDevice(0);

	double deltaS = 2 * R / Ns;
	double deltaD = 2 * M_PI / Nd;
	double* integralParameters = InitializeIntegralParameters();
	 
	CUDAArray<double> cudaIntegralParameters = CUDAArray<double>(integralParameters, DictionaryCount, 1);
	CUDAArray<double> cudaIntegralValues = CUDAArray<double>(DictionaryCount, 1);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		printf("cudaIntegralValues = CUDAArray<double>(DictionaryCount, 1); - ERROR!!!\n");
	}

	double factor = 1 / (SigmaD * sqrt(2 * M_PI));
    double h = deltaD / N;
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(DictionaryCount, defaultThreadCount));
		 
	cudaMakeTableOfIntegrals<<<gridSize,blockSize>>>(cudaIntegralParameters, cudaIntegralValues, factor, h, deltaD);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		printf("cudaMakeTableOfIntegrals - ERROR!!!\n");
	}

	CUDAArray<Minutiae> cudaMinutiae = CUDAArray<Minutiae>(minutiae, minutiaeCount, 1);
	CUDAArray<int> cudaWorkingArea = CUDAArray<int>(workingArea, columns, rows);
	CUDAArray<Cell> cudaResult = CUDAArray<Cell>(result, Ns * Ns * Nd, minutiaeCount); // result

	gridSize = dim3(ceilMod(Ns * Ns * Nd, defaultThreadCount), ceilMod(minutiaeCount, defaultThreadCount));
	
	cudaMCC<<<gridSize,blockSize>>>(cudaMinutiae, minutiaeCount, cudaIntegralParameters, cudaIntegralValues, 
		rows, columns, cudaWorkingArea, deltaS, deltaD, cudaResult);

	cudaResult.GetData(result);

	cudaIntegralParameters.Dispose();
	cudaIntegralValues.Dispose();
	cudaMinutiae.Dispose();
	cudaWorkingArea.Dispose();
	cudaResult.Dispose(); 
}

CUDAArray<float> LoadImageMCC(const char* name, bool sourceIsFloat = false)
{
	FILE* f = fopen(name,"rb");
			
	int width;
	int height;
	
	fread(&width,sizeof(int),1,f);			
	fread(&height,sizeof(int),1,f);
	
	float* ar2 = (float*)malloc(sizeof(float)*width*height);

	if(!sourceIsFloat)
	{
		float* ar = (float*)malloc(sizeof(float)*width*height);
		fread(ar,sizeof(float),width*height,f);
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

CUDAArray<int> LoadImageAsIntMCC(const char* name, bool sourceIsFloat = false)
{
	FILE* f = fopen(name,"rb");
			
	int width;
	int height;
	
	fread(&width,sizeof(int),1,f);			
	fread(&height,sizeof(int),1,f);
	
	int* ar2 = (int*)malloc(sizeof(int)*width*height);

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
		fread(ar2,sizeof(int),width*height,f);
	}
	
	fclose(f);

	CUDAArray<int> sourceImage = CUDAArray<int>(ar2,width,height);

	free(ar2);		

	return sourceImage;
}

void main()
{
	 CUDAArray<int> source = LoadImageAsIntMCC("C:\\Users\\Tanya\\Documents\\Studies\\SummerSch0ol2013\\SVN\\CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking.Tests\\Resources\\104_61globalBinarization150Thinned.phg");
	 int* sourceInt = source.GetData(); ////
	 int width = 256;
	 int height = 364;

	 Minutiae* minutiae = (Minutiae*)malloc(width*height*sizeof(Minutiae));
	 int* minutiaeCounter = (int*)malloc(sizeof(int));
	 int* minutiaeAsIntArr = (int*)malloc(minutiaeCounter[0] * 2 * sizeof(int));
	 int* workingArea = (int*)malloc(width*height*sizeof(int));

	 FindBigMinutiaeCUDA(sourceInt, width, height, minutiae, minutiaeCounter, R);

	 for (int i = 0; i < minutiaeCounter[0] * 2; i++)
	 {
		 if (i % 2 == 0)
		 {
			 minutiaeAsIntArr[i] = minutiae[i / 2].x;
		 }
		 else
		 {
			 minutiaeAsIntArr[i] = minutiae[i / 2].y;
		 }
	 }

	 BuildWorkingArea(workingArea, height, width, R, minutiaeAsIntArr, minutiaeCounter[0]);
	
	 Cell* result = (Cell*)malloc(Ns * Ns * Nd * minutiaeCounter[0] * sizeof(Cell)); 

	 MCCMethod(result, minutiae, minutiaeCounter[0], height, width, workingArea);
	
	 int q = 0;

	 source.Dispose();
	 cudaFree(minutiae);
	 cudaFree(minutiaeCounter);
	 cudaFree(minutiaeAsIntArr);
	 cudaFree(workingArea);
}
