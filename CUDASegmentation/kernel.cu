#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionHelper.h"
#include <stdio.h>

extern "C"{

__declspec(dllexport) void CUDASegmentator(float* img, int imgWidth, int imgHeight, float weightConstant, int windowSize, int* mask, int maskWidth, int maskHight);
__declspec(dllexport) void PostProcessing(int* mask, int maskX, int maskY, int threshold);
}

#define ceilMod(x, y) (x+y-1)/y

struct Point
	{
		int X;
		int Y;
		Point* Next;
	} ;

struct AreaStruct
	{
		int AreaSize;
		Point* Points;
		AreaStruct* Next;
	} ;

struct AreasList
{
	AreaStruct* First;
};

bool IsNearBorder(Point* points, int xBorder, int yBorder)
{
	Point* current = points;

	while(current != 0)
    {
		if (current->X == 0 || 
			current->Y == 0 ||
			current->X == xBorder - 1 || 
			current->Y == yBorder - 1)
		{
			return true;
		}
		current = current->Next;
	}
	return false;
}

AreaStruct* FindAreaWithPoint(AreaStruct* areas, int i, int j)
{	
	AreaStruct* currentArea = areas;
    Point* currentPoint = 0;
	
	while(currentArea != 0)
	{
		currentPoint = currentArea->Points;

		while(currentPoint != 0)
		{
			if(currentPoint->X == i && currentPoint->Y == j)
			{
				return currentArea;
			}
			currentPoint = currentPoint->Next;
		}
		currentArea = currentArea->Next;
	}
	return 0;
}

Point* findLastPoint(Point* points)
{
	Point* lastPoint = points;

	while(lastPoint->Next != 0)
	{
		lastPoint = lastPoint->Next;
	} 

	return lastPoint;
}

void MergeAreas(AreasList* areas, int i, int j)
{
	AreaStruct* firstArea = FindAreaWithPoint(areas->First, i - 1, j);
	AreaStruct* secondArea = FindAreaWithPoint(areas->First, i, j - 1);
                       
	if (firstArea != secondArea)
	{
		Point* lastPoint = findLastPoint(firstArea->Points);
		lastPoint->Next = secondArea->Points;
		firstArea->AreaSize += secondArea->AreaSize;
	}
    	
	Point* newPoint = (Point*)malloc(sizeof(Point));	
	newPoint->X = i;
	newPoint->Y = j;
	newPoint->Next = 0;
	Point* lastPoint = findLastPoint(firstArea->Points);
	
	lastPoint->Next = newPoint;
	firstArea->AreaSize +=1;

	if(firstArea != secondArea)
	{
		//remove secondArea
		AreaStruct* prevArea = areas->First;
		if(prevArea == secondArea)
		{

			areas->First = prevArea->Next;	
		}
		else
		{
			while(prevArea->Next != secondArea)
			{
				prevArea = prevArea->Next;
			}

			prevArea->Next = secondArea->Next;
		}

		free(secondArea);
	}
}

void AddPointToArea(AreasList* areas, int iSearch, int jSearch, int i, int j)
{
	AreaStruct* areaToAddPoint = FindAreaWithPoint(areas->First, iSearch, jSearch);
	
	Point* lastPoint = findLastPoint(areaToAddPoint->Points);
	
	Point* newPoint = (Point*)malloc(sizeof(Point));	
	newPoint->X = i;
	newPoint->Y = j;
	newPoint->Next = 0;
	lastPoint->Next = newPoint;	
	areaToAddPoint->AreaSize += 1;
}

AreasList* GenerateAreas(int* mask, int maskX, int maskY, bool isBlack)
{
	AreasList* areas = 0; 

	for (int i = 0; i < maskY; i++)
    {
		for (int j = 0; j < maskX; j++)
        {
			if (mask[i*maskX + j] && isBlack || !mask[i*maskX + j] && !isBlack)
            {
				continue;
            }
			
			if(isBlack)
			{
				if (j - 1 >= 0 && i - 1 >= 0 && !mask[i*maskX + j - 1] && !mask[(i - 1)*maskX + j])
				{
					MergeAreas(areas, i, j);
					continue;
				}
				if (j - 1 >= 0 && !mask[i*maskX + j - 1] && (i - 1 < 0 || i - 1 >= 0 && mask[(i - 1) * maskX + j]))
				{
					AddPointToArea(areas, i, j-1, i, j);
					continue;
				}
				if (i - 1 >= 0 && !mask[(i - 1) * maskX + j] && (j - 1 < 0 || j - 1 >= 0 && mask[i * maskX + j - 1]))
				{
					AddPointToArea(areas, i-1, j, i, j);
					continue;
				}
			}
			else
			{
				if (j - 1 >= 0 && i - 1 >= 0 && mask[i * maskX + j - 1] && mask[(i - 1) * maskX + j])
				{					
					MergeAreas(areas, i, j);
					continue;
				}
				if (j - 1 >= 0 && mask[i * maskX + j - 1] && (i - 1 < 0 || i - 1 >= 0 && !mask[(i - 1) * maskX + j]))
				{
					AddPointToArea(areas, i, j - 1, i, j);
					continue;
				}
				if (i - 1 >= 0 && mask[(i - 1) * maskX + j] && (j - 1 < 0 || j - 1 >= 0 && !mask[i * maskX + j - 1]))
				{
					AddPointToArea(areas, i - 1, j, i, j);
					continue;
				}
			}

			//create new area
			Point* newPoint = (Point*)malloc(sizeof(Point));
			newPoint->X = i;
			newPoint->Y = j;
			newPoint->Next = 0;

			AreaStruct* newArea = (AreaStruct*)malloc(sizeof(AreaStruct));
			newArea->AreaSize = 1;
			newArea->Points = newPoint;
			newArea->Next = 0;

			
			
			if(areas == 0)
			{
				areas =(AreasList*)malloc(sizeof(AreasList));
				areas->First = newArea;
			}
			else
			{
				AreaStruct* currentArea = areas->First;
				while(currentArea->Next != 0)
				{
					currentArea = currentArea->Next;
				}
				currentArea->Next = newArea;			
			}			
		}
	}
	return areas;
}

int* FillAreas(AreasList* areas, int* mask, int maskX, int maskY, int threshold, bool isBlack)
{
	for(int i = 0; i < maskY; i++)
	{
		for(int j = 0; j < maskX; j++)
		{
			if(mask[i * maskX + j] && isBlack || !mask[i * maskX + j] && !isBlack)
			{
				continue;		
			}

			AreaStruct* currentArea = FindAreaWithPoint(areas->First, i, j); 

			if( isBlack && ((currentArea->AreaSize) < threshold) && !IsNearBorder(currentArea->Points, maskX, maskY)
				|| !isBlack &&((currentArea->AreaSize) < threshold))
			{
				mask[i * maskX + j]= mask[i * maskX + j] ? 0 : 1;
			}
		}
	}
	return mask;
}

void PostProcessing(int* mask, int maskX, int maskY, int threshold)
{
	AreasList* blackAreas = GenerateAreas(mask, maskX, maskY, true);
	mask = FillAreas(blackAreas, mask, maskX, maskY, threshold, true);
	AreasList* imageAreas = GenerateAreas(mask, maskX, maskY, false);
	mask = FillAreas(imageAreas, mask, maskX, maskY, threshold, false);
}

__global__ void cudaGetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> xGradient, CUDAArray<float> yGradient)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(row<magnitude.Height&&column<magnitude.Width)
	{
		float newValue = xGradient.At(row,column)*xGradient.At(row,column) +yGradient.At(row,column)*yGradient.At(row,column);
		newValue = sqrt(newValue);
		magnitude.SetAt(row,column, newValue);
	}
}

void GetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> xGradient, CUDAArray<float> yGradient)
{
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(magnitude.Width,defaultThreadCount),
		ceilMod(magnitude.Height,defaultThreadCount));

	cudaGetMagnitude<<<gridSize,blockSize>>>(magnitude, xGradient, yGradient);
	cudaError_t error = cudaDeviceSynchronize();
}

__global__ void cudaGetMask(CUDAArray<float> initialArray, CUDAArray<int> mask, int blockSize, float average)
{
	if(defaultRow()<mask.Height&&defaultColumn()<mask.Width)
	{
		float sum = 0.0f;

		int rowOffset = defaultRow() * blockSize;
		int columnOffset = defaultColumn() * blockSize;

		for(int i = 0; i < blockSize; i++)
		{
			if(columnOffset + i < initialArray.Width)
			{
				for(int j = 0; j < blockSize; j++)
				{
					if(rowOffset + j < initialArray.Height)
					{
						sum += initialArray.At(rowOffset + j, columnOffset + i);
					}		
				}
			}			
		}

		float avg = sum/(blockSize*blockSize);
		int result = (int)(!(avg < average));
		mask.SetAt(defaultRow(),defaultColumn(),result);
	}
}

float GetAverageFromArray(CUDAArray<float> arrayToAverage)
{
	float sum = 0;
	float* ar = arrayToAverage.GetData();
	for(int i; i<arrayToAverage.Width*arrayToAverage.Height; i++)
	{
		sum += ar[i];
	}
	free(ar);

	return sum/(arrayToAverage.Height*arrayToAverage.Width);	
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

void SaveMask(int* mask,int width, int height, const char* name)
{
	FILE* f = fopen(name,"wb");
	
	char* ar = (char*)malloc(sizeof(char)*(width*2+2)*height);
	int k = 0;

	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			ar[k++] = mask[j + i * width] ? 49 : 48;
			ar[k++] = ' ';
		}

		ar[k++] = 10;
		ar[k++] = 13;
	}

	fwrite(ar, sizeof(char), (width*2+2)*height,f);
	fclose(f);
}

void CUDASegmentator(float* img, int imgWidth, int imgHeight, float weightConstant, int windowSize, int* mask, int maskWidth, int maskHight)
  {	 
	  cudaError_t cudaStatus = cudaSetDevice(0);
	  
	  CUDAArray<float> source = CUDAArray<float>(img, imgWidth, imgHeight);
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("CUDAArray<float> source = loadImage(...) - ERROR!!!\n");
	  }

	  // Sobel :	  
	  CUDAArray<float> xGradient = CUDAArray<float>(imgWidth,imgHeight);
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("Create xGradient[] - ERROR!!!\n");
	  }

	  CUDAArray<float> yGradient = CUDAArray<float>(imgWidth,imgHeight);

	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("Create yGradient[] - ERROR!!!\n");
	  }
	 
	  float xKernelCPU[3][3] = {{-1,0,1},
							{-2,0,2},
							{-1,0,1}};

	  CUDAArray<float> xKernel = CUDAArray<float>((float*)&xKernelCPU,3,3);	  
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("Create xKernel[] - ERROR!!!\n");
	  }

	  float yKernelCPU[3][3] = {{-1,-2,-1},
							{0,0,0},
							{1,2,1}};

	  CUDAArray<float> yKernel = CUDAArray<float>((float*)&yKernelCPU,3,3);	  
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("Create yKernel[] - ERROR!!!\n");
	  }

	  Convolve(xGradient, source, xKernel);
	  Convolve(yGradient, source, yKernel);
	  
	  //magnitude of gradient
	  CUDAArray<float> magnitude = CUDAArray<float>(imgWidth,imgHeight);
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("Create magnitude[] - ERROR!!!\n");
	  }

	  GetMagnitude(magnitude, xGradient, yGradient);
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("cudaGetMask - ERROR!!!\n");
	  }

	  xGradient.Dispose();
	  yGradient.Dispose();
	  xKernel.Dispose();
	  yKernel.Dispose();

	  //average magnitude 
	  float average = GetAverageFromArray(magnitude);

	  dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
		
	  dim3 gridSize = dim3(ceilMod(maskWidth, defaultThreadCount), ceilMod(maskHight, defaultThreadCount));

		//mask creation
	  CUDAArray<int> CUDAmask = CUDAArray<int>(mask, maskWidth, maskHight);
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("create Mask - ERROR!!!\n");
	  }
	  
	  cudaGetMask<<<gridSize,blockSize>>>(magnitude, CUDAmask, windowSize, average*weightConstant);
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("cudaGetMask - ERROR!!!\n");
	  }

	  cudaStatus = cudaDeviceSynchronize();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("Synchronize - ERROR!!!\n");
	  }

	  magnitude.Dispose();

	 CUDAmask.GetData(mask);

	 CUDAmask.Dispose();
	 cudaDeviceReset();
}

//void main()
//{
//	 CUDAArray<float> source = loadImage("C:\\temp\\103_4.bin");
//	 float* sourceFloat = source.GetData();
//
//	 int imgWidth = source.Width;
//	 int imgHeight = source.Height;
//
//	 float weightConstant = 0.3; 
//	 int windowSize = 12;
//	 int threshold = 5;
//
//	 int maskX = (int)ceil(((double)imgWidth) / windowSize);
//	 int maskY = (int)ceil(((double)imgHeight) / windowSize);
//
//	 int* mask = 0;
//	 mask = (int*)malloc(maskX*maskY*sizeof(int));
//
//	 CUDASegmentator(sourceFloat, imgWidth, imgHeight, weightConstant, windowSize, mask, maskX, maskY);
//	 SaveMask(mask, maskX, maskY, "C:\\temp\\maskCUDASegmentator.txt");
//
//	 PostProcessing(mask, maskX, maskY, threshold);
//	 SaveMask(mask,maskX, maskY, "C:\\temp\\maskPostProcessing.txt");
//
//	 free(mask);
//}
