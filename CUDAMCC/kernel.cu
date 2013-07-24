#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionHelper.h"
#include "Point.h"

#define _USE_MATH_DEFINES
#include <math.h>

#define ceilMod(x, y) (x+y-1)/y
/*
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

struct Minutiae
{
	int x;
	int y;
	float angle;
};

struct Cell
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

__device__ double GetIntegralParameterIndex(double param, double* integralParameters)
{
    for (int i = 0; i < cudaDictionaryCount; i++)
    {
        if (abs(integralParameters[i] - param) <= M_PI / cudaDictionaryCount)
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

__device__ double GetDirectionalContribution(double mAngle, double mtAngle, int k, double* integralParameters, CUDAArray<double> integralValues, double deltaD)
{
    double angleFromLevel = NormalizeAngle(GetAngleFromLevel(k, deltaD));
    double differenceAngles = NormalizeAngle(mAngle - mtAngle);
    double param = NormalizeAngle(angleFromLevel - differenceAngles);
	int indexOfResult = GetIntegralParameterIndex(param, integralParameters);

    return integralValues.At(1, indexOfResult);
}

__device__ double GetDistance(int x1, int y1, int x2, int y2)
{
	return sqrt((double)((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

__device__ double GetSpatialContribution(Minutiae m, Point currentCell)
{
	double distance = GetDistance(m.x, m.y, currentCell.X, currentCell.Y);

	double commonDenom = cudaSigmaS * cudaSigmaS;
	commonDenom *= 2;
    double denominator = cudaSigmaS * sqrt(M_PI * 2);
    double result = exp(-(distance * distance) / commonDenom) / denominator;

    return result;
}

__device__ bool IsEqualMinutiae(Minutiae m1, Minutiae m2)
{
	return m1.x == m2.x && m1.y == m2.y && m1.angle == m2.angle;
}

__device__ void GetNeighbourMinutiae(Minutiae* neighbourMinutiae, int& neighbourMinutiaeCount, Minutiae* minutiae, int minutiaeCount, Minutiae minutia, Point currentCell)
{
	for (int i = 0; i < minutiaeCount; i++)
	{
		if (!IsEqualMinutiae(minutiae[i], minutia) && GetDistance(minutiae[i].x, minutiae[i].y, currentCell.X, currentCell.Y) <= 3 * cudaSigmaS)
		{
			neighbourMinutiae[neighbourMinutiaeCount++] = minutiae[i];
		}
	}
}

__device__ void GetCoordinatesInFingerprint(Point point, Minutiae m, int i, int j, double deltaS)
{
	double halfNs = (1 + cudaNs) / 2;
	double sinTetta = sin(m.angle);
	double cosTetta = cos(m.angle);
	double iDelta = cosTetta * (i - halfNs) + sinTetta * (j - halfNs);
	double jDelta = -sinTetta * (i - halfNs) + cosTetta * (j - halfNs);
    
	point.X = (int)(m.x + deltaS * iDelta);
	point.Y = (int)(m.y + deltaS * jDelta);
}

__device__ int CalculateMaskValue(Minutiae m, int i, int j, int rows, int columns, bool* workingArea, double deltaS)
{
	Point point = Point();
	//cudaError_t error = cudaMalloc((void**)&point, sizeof(Point)); /////////
	GetCoordinatesInFingerprint(point, m, i, j, deltaS);
            
	if (point.X < 0 || point.X >= rows ||
		point.Y < 0 || point.Y >= columns)
	{
		return 0;
	}
            
	return ((GetDistance(m.x, m.y, point.X, point.Y) <= R) && workingArea[point.X, point.Y])
		? 1
		: 0;
}

__global__ void cudaMCC (CUDAArray<Cell> result, Minutiae* minutiae, int minutiaeCount, double* integralParameters, CUDAArray<double> integralValues, int rows, int columns, bool* workingArea, double deltaS, double deltaD)
{
	int row = defaultRow(); // J  < ~80
	int column = defaultColumn(); // I  < 1512 = 16*16*6

	int index = column;
	int i = row % ((cudaNs - 1) * (cudaNd - 1));
	int j = (row - i*((cudaNs - 1) * (cudaNd - 1))) % (cudaNd - 1);
	int k = (row - i*((cudaNs - 1) * (cudaNd - 1))) - j * (cudaNd - 1);

	Cell newCell;
	cudaError_t error = cudaMalloc((void**)&newCell, sizeof(Cell)); /////////

	// (0)
	newCell.minutia = minutiae[index];
	// (1)
	newCell.mask = CalculateMaskValue(minutiae[index], i, j, rows, columns, workingArea, deltaS);  //////
	// (2)
	Point currentCoordinate = Point();
	GetCoordinatesInFingerprint(currentCoordinate, minutiae[index], i, j, deltaS);

    Minutiae* neighbourMinutiae = 0;
	error = cudaMalloc((void**)&neighbourMinutiae, minutiaeCount * sizeof(Minutiae));
	int neighbourMinutiaeCount = 0;
	GetNeighbourMinutiae(neighbourMinutiae, neighbourMinutiaeCount, minutiae, minutiaeCount, minutiae[index], currentCoordinate);
	
	double psiParameter = 0;

    for (int counter = 0; counter < neighbourMinutiaeCount; counter++)
    {
        double spatialContribution = GetSpatialContribution(neighbourMinutiae[counter], currentCoordinate);
        double directionalContribution = GetDirectionalContribution(minutiae[index].angle, neighbourMinutiae[counter].angle, k, integralParameters, integralValues, deltaD);
        psiParameter = psiParameter + spatialContribution * directionalContribution;
    }
	
	newCell.value = Psi(psiParameter); 

	//allNeighbours.AddRange(neighbourMinutiae);
	//allNeighbours = MinutiaListDistinct(allNeighbours);
	
	//if (allNeighbours.Count < Constants.MinM && !IsValidMask())
	//{
	//	continue;
	//}

	//response.Add(minutiae[index], new Tuple<int[, ,], int[, ,]>(value, mask));
}

__global__ void cudaMakeTableOfIntegrals(double* integralParameters, CUDAArray<double> integralValues, double factor, double h, double deltaD)
{
	int column = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	double a = integralParameters[column] - deltaD / 2;
	double integrand = 0;
    double result = 0;

    for (int i = 0; i < cudaN; i++)
    {
		integrand = a + ((2 * i + 1) * h) / 2;
		integrand = exp((-integrand * integrand) / (2 * cudaSigmaD * cudaSigmaD));
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

void MCCMethod(Minutiae* minutiae, int minutiaeCount, int rows, int columns, bool* workingArea)
{
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

	double factor = 1 / (SigmaD * sqrt(2 * M_PI));
    double h = deltaD / N;
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(DictionaryCount, defaultThreadCount));
		 
	cudaMakeTableOfIntegrals<<<gridSize,blockSize>>>(integralParameters, integralValues, factor, h, deltaD);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		printf("cudaMakeTableOfIntegrals - ERROR!!!\n");
	}

	Cell* resultData = 0;
	//cudaStatus = cudaMalloc((Cell**)&resultData, Ns * Ns * Nd * minutiaeCount * sizeof(Cell));
	//if (cudaStatus != cudaSuccess) 
	//{
	//	printf("cudaMalloc((Cell**)&resultData, Ns * Ns * Nd * minutiaeCount * sizeof(Cell)); - ERROR!!!\n");
	//}

	CUDAArray<Cell> result = CUDAArray<Cell>(Ns*Ns*Nd, minutiaeCount); // resultData, 
	gridSize = dim3(ceilMod(minutiaeCount, defaultThreadCount));
	cudaMCC<<<gridSize,blockSize>>>(result, minutiae, minutiaeCount, integralParameters, integralValues, rows, columns, workingArea, deltaS, deltaD);

	return result.GetData(resultData);
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


	 /*CUDAArray<float> source = loadImage("C:\\temp\\103_4.bin");
	 float* sourceFloat = source.GetData();

	 int imgWidth = source.Width;
	 int imgHeight = source.Height;*/


	 // int rows, int columns

	//MCCMethod(minutiae, minutiaeCount, rows, columns, workingArea)
}
