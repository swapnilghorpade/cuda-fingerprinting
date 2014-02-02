#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionHelper.h"
#include "BuildWorkingArea.h"
#include "FindMinutiae.h"
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

#define ceilMod(x, y) (x+y-1)/y

const int Omega = 50;
const int R = 70;
const int Ns = 16; // 8
const int Nd = 6;
const float deltaD = M_PI * 2 / Nd;
const float deltaS = (float)R * 2 / Ns;

const float SigmaD = 2 * M_PI / 9;
const float SigmaS = 28.0f / 3;
const int BigSigma = 50; //
const float MinVC = 0.75; //

const int MinM = 2; // 
const float MinME = 0.6; //
const float SigmaTetta = M_PI / 2;  //
const int N = 10;
const int DictionaryCount = 360; // 720, 1440

__device__ const int cudaNs = 16; // 8
__device__ const int cudaNd = 6;
__device__ const float cudaSigmaS = 28.0f / 3;
__device__ const float cudaSigmaD = 2.0f* M_PI / 9;
__device__ const float cudaDeltaD = M_PI * 2 / Nd;
__device__ const float cudaDeltaS = (float)R * 2 / Ns;
__device__ const float cudaMuPsi = 0.01f;
__device__ const int cudaBigSigma = 50; //
__device__ const float cudaMinVC = 0.75; //
__device__ const int cudaMinM = 2; //
__device__ const float cudaMinME = 0.6; //
__device__ const float cudaSigmaTetta = M_PI / 2;  //
__device__ const int cudaN = 10;
__device__ const int cudaDictionaryCount = 360; // 720, 1440


float previousSigmaS = -100500.0f, previousDeltaD = -100500.0f;

float* integrals = NULL;

int amountInCircle = -1;

int* pointsInCuboid = NULL;

__device__ __inline__ float GetAngleFromLevel(int k, float deltaD)
{
	return - M_PI + ((float)k - 0.5f) * deltaD;
}

__device__ float NormalizeAngle(float angle)
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

__host__ __device__ __inline__ float GetSquaredDistance(int x1, int y1, int x2, int y2)
{
	return (float)((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

__device__ __inline__ float GetSpatialContribution(int minutiaX, int minutiaY, int currentCellX, int currentCellY)
{
	float distance = sqrt(GetSquaredDistance(minutiaX, minutiaY, currentCellX, currentCellY));

	float commonDenom = cudaSigmaS * cudaSigmaS * 2;
	float denominator = cudaSigmaS * 2.50662827463f; // sqrt(2*pi)
	float result = exp(-(distance * distance) / commonDenom) / denominator;

	return result;
}

__device__ bool IsEqualMinutiae(Minutiae m1, Minutiae m2)
{
	return m1.x == m2.x && m1.y == m2.y && m1.angle == m2.angle;
}

__global__ void cudaMCC (CUDAArray<unsigned int> result, CUDAArray<unsigned int> validity,
			CUDAArray<int> minutiaeXs, CUDAArray<int> minutiaeYs, CUDAArray<float> minutiaeAngles, int minutiaeCount, 
			CUDAArray<float> integrals, CUDAArray<int> workingArea, CUDAArray<int> circleInclusion, int amountInCircle)
{
	int x = threadIdx.x + 1;
	int y = threadIdx.y + 1;

	int number = circleInclusion.At(threadIdx.y, threadIdx.x);

	if(number != 0) // only for valid cells
	{
		int cellValidity = 1;
		int cellValue = 0;

		// determine reference minutia
		int minutiaX = minutiaeXs.At(0, blockIdx.x);
		int minutiaY = minutiaeYs.At(0, blockIdx.x);
		float minutiaAngle = minutiaeAngles.At(0, blockIdx.x);
		minutiaAngle = NormalizeAngle(minutiaeAngles.At(0, blockIdx.x));

		// determine coordinates in terms of fingerprint
		// TODO: to check thoroughly
		float halfNs = (1.0f + (float)cudaNs) / 2;
		float sinTetta = sin(minutiaAngle);
		float cosTetta = cos(minutiaAngle);
		float iDelta = cosTetta * (x - halfNs) + sinTetta * (y - halfNs);
		float jDelta = -sinTetta * (x - halfNs) + cosTetta * (y - halfNs);
    
		int coordinateX = (int)(minutiaX + cudaDeltaS * iDelta);
		int coordinateY = (int)(minutiaY + cudaDeltaS * jDelta);

		int workerAreaValidity = workingArea.At(coordinateY, coordinateX);

		if (coordinateX < 0 || coordinateX >= workingArea.Width ||
			coordinateY < 0 || coordinateY >= workingArea.Height || workerAreaValidity == 0)
		{
			// cell is invalid - out of working area
			cellValidity = 0;
		}
		else
		{
			float psiValue = 0;
			for (int i = 0; i < minutiaeCount; i++)
			{
				int secondMinutiaX = minutiaeXs.At(0, i);
				int secondMinutiaY = minutiaeYs.At(0, i);
				
				if(i != blockIdx.x && GetSquaredDistance(coordinateX, coordinateY, secondMinutiaX, secondMinutiaY) <= 9.0f * cudaSigmaS * cudaSigmaS)
				{
					float spatialContribution = GetSpatialContribution(secondMinutiaX, secondMinutiaY, coordinateX, coordinateY);

					float angleFromLevel = NormalizeAngle(GetAngleFromLevel(blockIdx.y + 1, cudaDeltaD));
					float secondMinutiaAngle = NormalizeAngle(minutiaeAngles.At(0, i));
					float differenceAngles = NormalizeAngle(minutiaAngle - secondMinutiaAngle);
					float param = NormalizeAngle(angleFromLevel - differenceAngles);

					float step = M_PI * 2 / DictionaryCount;
					param = (param + M_PI) / step;

					int index = (int)roundf(param);
					if(index >= integrals.Width) index = 0;

					float directionalContribution = integrals.At(0, index);

					psiValue = psiValue + spatialContribution * directionalContribution;
				}
			}

			// bit-wise value
			if(psiValue >= cudaMuPsi) 
				cellValue = 1;
		}
		
		// save values at position

		number--;
		int intNumber = (amountInCircle * blockIdx.y + number) / (sizeof(unsigned int) * 8);
		int position = (amountInCircle * blockIdx.y + number) % (sizeof(unsigned int) * 8);
		int templateSize = ceilMod(amountInCircle*gridDim.y, (sizeof(unsigned int) * 8));
		if(cellValue != 0)
		{
			atomicAdd(result.cudaPtr + intNumber + templateSize*blockIdx.x, ((unsigned int)1)<<position);
		}

		if(cellValidity !=0)
		{
			atomicAdd(validity.cudaPtr + intNumber + templateSize*blockIdx.x, ((unsigned int)1)<<position);
		}
	}

	//int row = defaultRow(); // J  < ~80
	//int column = defaultColumn(); // I  < 1536 = 16*16*6
	//
	//if(row<minutiaeCount && column< cudaNs*cudaNs*cudaNd)
	//{

	//int index = row;
	//int i = column % ((cudaNs - 1) * (cudaNd - 1));
	//int j = (column - i*((cudaNs - 1) * (cudaNd - 1))) % (cudaNd - 1);
	//int k = (column - i*((cudaNs - 1) * (cudaNd - 1))) - j * (cudaNd - 1);

	//int coordinateX = 0; 
	//int coordinateY = 0; 
	//float psiParameter = 0;
	//float spatialContribution = 0;
	//float directionalContribution = 0;

	//if (coordinateX < 0 || coordinateX >= rows ||
	//	coordinateY < 0 || coordinateY >= columns)
	//{
	//	result.minutia = minutiae.At(0, index);
	//	result.mask = 0;
	//	result.value = 0;
	//	arr.SetAt(row, column, result);

	//	return;
	//}

	//for (int i = 0; i < minutiaeCount; i++)
	//{
	//	if (!IsEqualMinutiae(minutiae.At(0, i), minutiae.At(0, index)) && 
	//		GetDistance((minutiae.At(0, i)).x, (minutiae.At(0, i)).y, coordinateX, coordinateY) <= 3 * cudaSigmaS)
	//	{
	//		spatialContribution = GetSpatialContribution(minutiae.At(0, i), coordinateX, coordinateY);
	//		directionalContribution = GetDirectionalContribution((minutiae.At(0, index)).angle, (minutiae.At(0, i)).angle, k, integralParameters, integralValues, deltaD);
	//		psiParameter = psiParameter + spatialContribution * directionalContribution;
	//	}
	//}

	//int psiValue = Psi(psiParameter); 
	//int maskValue = CalculateMaskValue(minutiae.At(0, index), i, j, rows, columns, workingArea, deltaS);
	//if(maskValue > 0)
	//{
	//	numOfValidMask.SetAt(0, column, numOfValidMask.At(0,column)+1);
	//}
	//result.minutia = minutiae.At(0, index);
	//result.mask = maskValue;
	//result.value = psiValue;
	//arr.SetAt(row, column, result);
	//}
	// check

	//allNeighbours.AddRange(neighbourMinutiae);
	//allNeighbours = MinutiaListDistinct(allNeighbours);
	
	//if (allNeighbours.Count < Constants.MinM && !IsValidMask())
	//{
	//	continue;
	//}

}

void Init(float sigmaS, float deltaD)
{
	if(sigmaS != previousSigmaS || deltaD != previousDeltaD)
	{
		if(integrals != NULL) free(integrals);
		integrals = (float*)malloc(sizeof(float)*DictionaryCount);

		float step = M_PI * 2 / DictionaryCount;
		int i=0;
		for(float key = -M_PI; i < DictionaryCount; key += step, i++)
			integrals[i] = expf(0.5f / sigmaS / sigmaS) / sigmaS / 2.0f / sqrt(2.0f) * (erff(key + deltaD / 2) - erff(key - deltaD / 2.0f));
	}

	pointsInCuboid = (int*)malloc(sizeof(int)*Ns*Ns);

	amountInCircle = 1;

	for(int i=0; i<Ns; i++)
	{
		float xCoord = deltaS*i+deltaS/2-R;

		for(int j=0; j<Ns; j++)
		{
			float yCoord = deltaS*j+deltaS/2-R;
			pointsInCuboid[i*Ns+j] = (xCoord*xCoord+yCoord*yCoord)<=R*R ? amountInCircle++ : 0; // test for point being in circle
		}
	}

	amountInCircle--;
}

int getTemplateLengthInBits()
{
	return amountInCircle * Nd;
}

int countBits(unsigned int x)
{
	x -= (x >> 1) & (0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F;
	x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
	x = (x & 0x0000FFFF) + (x >> 16);
	return x;
}

void makeMCCTemplate(unsigned char* result, unsigned char* validity, int* templateCount,
	int* minutiaeXs, int* minutiaeYs, float* minutiaeAngles, int minutiaeCount, 
	int rows, int columns)
{
	Init(SigmaD, deltaD);

	int* workingArea = BuildAreaOfInterest(rows, columns, Omega, minutiaeXs, minutiaeYs, minutiaeCount);

	CUDAArray<float> cudaIntegrals = CUDAArray<float>(integrals, DictionaryCount, 1);

	CUDAArray<int> cudaCircleInclusion = CUDAArray<int>(pointsInCuboid ,Ns , Ns);

	unsigned int* preliminaryResult = (unsigned int*)malloc(sizeof(unsigned int)*minutiaeCount*(ceilMod(getTemplateLengthInBits(), (sizeof(unsigned int) * 8))));
	memset(preliminaryResult, 0, sizeof(unsigned int)*minutiaeCount*ceilMod(getTemplateLengthInBits(), (sizeof(unsigned int) * 8)));
	unsigned int* preliminaryValidity = (unsigned int*)malloc(sizeof(unsigned int)*minutiaeCount*(ceilMod(getTemplateLengthInBits(), (sizeof(unsigned int) * 8))));
	memset(preliminaryValidity, 0, sizeof(unsigned int)*minutiaeCount*ceilMod(getTemplateLengthInBits(), (sizeof(unsigned int) * 8)));

	CUDAArray<unsigned int> cudaResult = CUDAArray<unsigned int>(preliminaryResult, minutiaeCount*(ceilMod(getTemplateLengthInBits(), (sizeof(unsigned int) * 8))),  1);
	CUDAArray<unsigned int> cudaValidity = CUDAArray<unsigned int>(preliminaryValidity, minutiaeCount*(ceilMod(getTemplateLengthInBits(), (sizeof(unsigned int) * 8))),  1);

	CUDAArray<int> cudaMinutiaeXs = CUDAArray<int>(minutiaeXs, minutiaeCount, 1);
	CUDAArray<int> cudaMinutiaeYs = CUDAArray<int>(minutiaeYs, minutiaeCount, 1);
	CUDAArray<float> cudaMinutiaeAngles = CUDAArray<float>(minutiaeAngles, minutiaeCount, 1);
	
	CUDAArray<int> cudaWorkingArea = CUDAArray<int>(workingArea, columns, rows);

	dim3 blockSize = dim3(Ns, Ns); // one cylinder layer per block
	dim3 gridSize = dim3(minutiaeCount, Nd); // block per cylinder per layer

	cudaMCC<<<gridSize,blockSize>>>(cudaResult, cudaValidity,
		cudaMinutiaeXs, cudaMinutiaeYs, cudaMinutiaeAngles, minutiaeCount, 
		cudaIntegrals, cudaWorkingArea, cudaCircleInclusion, amountInCircle);
	
	cudaError_t error = cudaGetLastError();

	preliminaryResult = cudaResult.GetData();
	preliminaryValidity = cudaValidity.GetData();

	cudaMinutiaeXs.Dispose();
	cudaMinutiaeYs.Dispose();
	cudaMinutiaeAngles.Dispose();
	cudaWorkingArea.Dispose();
	cudaResult.Dispose();
	cudaValidity.Dispose();
	
	cudaCircleInclusion.Dispose();
	cudaIntegrals.Dispose();

	// make validity check

	unsigned int* uresult = (unsigned int*)result;
	unsigned int* uvalidity = (unsigned int*)validity;
	*templateCount = 0;

	int checkSize = ceilMod(getTemplateLengthInBits(), (sizeof(unsigned int) * 8));
	int targetCount = (int)(getTemplateLengthInBits()*MinVC);
	for(int i = 0; i < minutiaeCount; i++)
	{
		// check neighbors
		int neighbourCount = 0;
		float maxDistance = ((float)R+3.0f*SigmaS)*((float)R+3.0f*SigmaS);
		for(int j = 0; j< minutiaeCount; j++)
		{
			if(i != j && GetSquaredDistance(minutiaeXs[i], minutiaeYs[i], minutiaeXs[j], minutiaeYs[j]) <= maxDistance)
				neighbourCount++;
		}
	
		if(neighbourCount < MinM)continue;
		// check validities
		
		int count = 0;
		for(int j = 0; j<checkSize; j++)
		{
			count += countBits(preliminaryValidity[i*checkSize+j]);
		}

		if(count < targetCount) continue;

		(*templateCount)++;

		for(int j = 0; j<checkSize; j++)
		{
			*(uresult++) = preliminaryResult[i*checkSize+j];
			*(uvalidity++) = preliminaryValidity[i*checkSize+j];
		}
	}
}

void LoadImage(int* sourceInt, const char* name, bool sourceIsFloat = false)
{
	FILE* f = fopen(name,"rb");
			
	int width;
	int height;
	
	fread(&width,sizeof(int),1,f);			
	fread(&height,sizeof(int),1,f);
	
	if(!sourceIsFloat)
	{
		int* ar = (int*)malloc(sizeof(int)*width*height);
		fread(ar,sizeof(int),width*height,f);
		for(int i=0;i<width*height;i++)
		{
			sourceInt[i]=ar[i];
		}
		
		free(ar);
	}
	else
	{
		fread(sourceInt,sizeof(int),width*height,f);
	}
	
	fclose(f);
}

void main()
{
	int width = 256;
	int height = 364;

	

	//FindBigMinutiaeCUDA(sourceInt, width, height, minutiae, minutiaeCounter, 5);

	int* minutiaeXs = (int*)malloc(sizeof(int) * 3);
	int* minutiaeYs = (int*)malloc(sizeof(int) * 3);
	float* minutiaeAngles = (float*)malloc(sizeof(float) * 3);

	minutiaeXs[0] =  100;
	minutiaeYs[0] =  100;
	minutiaeAngles[0] = 0;

	minutiaeXs[1] =  150;
	minutiaeYs[1] =  100;
	minutiaeAngles[1] = M_PI/6;

	minutiaeXs[2] =  125;
	minutiaeYs[2] =  150;
	minutiaeAngles[2] = M_PI/2;

	int* result = (int*)malloc(sizeof(int) * 40 * 3);
	int* validity = (int*)malloc(sizeof(int) * 40 * 3);
	int templateCount = 0;
	
	makeMCCTemplate((unsigned char*) result, (unsigned char*) validity, &templateCount,
	minutiaeXs, minutiaeYs, minutiaeAngles, 3, height, width);
	clock_t time = clock();
	makeMCCTemplate((unsigned char*) result, (unsigned char*) validity, &templateCount,
	minutiaeXs, minutiaeYs, minutiaeAngles, 3, height, width);
	time = clock() - time;
	//free(sourceInt);
	free(minutiaeXs);
	free(minutiaeYs);
	free(minutiaeAngles);
}
