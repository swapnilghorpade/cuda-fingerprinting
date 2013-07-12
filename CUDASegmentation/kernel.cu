#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionHelper.h"
#include <stdio.h>

extern "C"{

__declspec(dllexport) void CUDASegmentator(int* img, int imgWidth, int imgHeight, float weightConstant, int windowSize, bool* mask, int maskWidth, int maskHight);
__declspec(dllexport) void PostProcessing(bool* mask, int maskX, int maskY, int threshold);

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
		int AreaNumber;
		int AreaSize;
		Point* Points;
		AreaStruct* Next;
	} ;


bool IsNearBorder(Point* points, int xBorder, int yBorder)
{
	Point* current = points;
	while(current!=0)
    {
		if (current->X == 0 || 
			current->Y == 0 ||
			current->X == xBorder-1 || 
			current->Y == yBorder-1)
		{
			return true;
		}
	}

	return false;
}

AreaStruct* FindAreaWithPoint(AreaStruct* areas, int i, int j)
{
	AreaStruct* result = 0;
	AreaStruct* currentArea = areas;
	Point* currentPoint = currentArea->Points;
	while(currentArea!=0)
	{
		while(currentPoint!=0)
		{
			if(currentPoint->X ==i && currentPoint->Y ==j)
			{
				result = currentArea;
				break;
			}
			currentPoint = currentPoint->Next;
		}
		if(result !=0)
		{
			break;
		}
		currentArea = currentArea->Next;
	}
	return result;
}

Point* findLastPoint(Point* points)
{
	Point* lastPoint =0;
	lastPoint = points;
	while(lastPoint->Next!=0)
	{
		lastPoint = lastPoint->Next;
	}
	return lastPoint;
}

void MergeAreas(AreaStruct* areas, int maskX, int areasSize, int i, int j)
{
	AreaStruct* firstArea = FindAreaWithPoint(areas, i-1, j);
	AreaStruct* secondArea = FindAreaWithPoint(areas, i, j-1);
                       
	if (firstArea != secondArea)
	{
		Point* lastPoint = findLastPoint(firstArea->Points);
		lastPoint->Next = secondArea->Points;
	}
    	
	Point* newPoint = (Point*) malloc (sizeof(Point));	
	newPoint->X = i;
	newPoint->Y = j;
	newPoint->Next =0;
	Point* lastPoint = findLastPoint(firstArea->Points);
	
	lastPoint->Next = newPoint;

	//remove secondArea
	AreaStruct* prevArea = areas;
	if(areas == secondArea)
	{
		areas = secondArea->Next;
	}
	else
	{
		while(prevArea->Next != secondArea)
		{
			prevArea= prevArea->Next;
		}
		prevArea->Next = secondArea->Next;
	}
	free(secondArea);
}

bool IsLeftImageTopBlack(int i, int j, bool topValue, bool leftValue, bool isBlack) 
{
	return (j - 1 >= 0 && (topValue || isBlack) && !(topValue && isBlack) &&					//top block is black 
           (i - 1 >= 0 && (leftValue || !isBlack) && !(leftValue && !isBlack)) || i - 1 < 0);	//left block is not black or not exist
}

bool IsLeftBlackTopImage(int i, int j, bool topValue, bool leftValue, bool isBlack) 
{
	return (i - 1 >= 0 && (leftValue || isBlack) && !(leftValue && isBlack) &&					//left block is black
           (j - 1 >= 0 && (topValue || !isBlack) && !(topValue && !isBlack)) || j - 1 < 0);	    //top block is not black or not exist
}

bool IsLeftBlackTopBlack(int i, int j, bool topValue, bool leftValue, bool isBlack)
{
	return (j - 1 >= 0 && (topValue || isBlack) && !(topValue && isBlack) &&					//top block is black 
            i - 1 >= 0 && (leftValue || isBlack) && !(leftValue && isBlack));					//left block is black
}

void fillArea(AreaStruct* areas, int iSearch, int jSearch, int i, int j)
{
	AreaStruct* areaToAddPoint = FindAreaWithPoint(areas,iSearch,jSearch);
	
	Point* lastPoint = findLastPoint(areaToAddPoint->Points);
	
	Point* newPoint = (Point*) malloc (sizeof(Point));	
	newPoint->X = i;
	newPoint->Y = j;
	newPoint->Next =0;

	lastPoint->Next = newPoint;	
}

AreaStruct* GenerateAreas(bool* mask, int maskX, int maskY, bool isBlack)
{
	int areaIndex = 0;
	bool isLeftImageTopBlack = false, isLeftBlackTopImage = false, isLeftBlackTopBlack = false;

	AreaStruct* areas =0; 

	for (int i = 0; i < maskX; i++)
    {
		for (int j = 0; j < maskY; j++)
        {
			if (!mask[i, j] && isBlack || mask[i, j] && !isBlack)
            {
				continue;
            }
			
			isLeftBlackTopImage = IsLeftBlackTopImage(i, j, mask[i, j - 1], mask[i - 1, j], isBlack);
			isLeftImageTopBlack = IsLeftImageTopBlack(i, j, mask[i, j - 1], mask[i - 1, j], isBlack);
			isLeftBlackTopBlack = IsLeftBlackTopBlack(i, j, mask[i, j - 1], mask[i - 1, j], isBlack);

			if (isLeftBlackTopBlack)
            {
				MergeAreas(areas, maskX, i, j, areaIndex);
				areaIndex--;
				continue;
            }
			if (isLeftBlackTopImage || isLeftImageTopBlack)
            {
				
				if (isLeftBlackTopImage)
				{

					fillArea(areas,i-1, j, i, j);
				}
				else
				{
					fillArea(areas, i, j-1, i, j);
				}
			   continue;
            }
			
			//create new area

			Point* newPoint = (Point*) malloc (sizeof(Point));
			newPoint->X = i;
			newPoint->Y = j;
			newPoint->Next =0;

			AreaStruct* newArea = (AreaStruct*) malloc (sizeof(AreaStruct));
			newArea->AreaNumber = areaIndex;
			newArea->AreaSize =1;
			newArea->Points = newPoint;
			newArea->Next = 0;

			AreaStruct* currentArea = areas;
			while(currentArea!=0&&currentArea->Next!=0)
			{
				if(currentArea ==0)
				{
					currentArea = newArea;
				}
				else
				{
					currentArea->Next = newArea;
				}
			}

			areaIndex++;
		}
	}
	return areas;
}

bool* FillAreas(AreaStruct* areas, bool* mask, int maskX, int maskY, int threshold, bool isBlack)
{
	for(int i = 0; i < maskX; i++)
	{
		for(int j = 0; j < maskY; j++)
		{
			if(mask[i, j] && isBlack || !mask[i, j] && !isBlack)
			{
				break;		
			}

			AreaStruct* currentArea = FindAreaWithPoint(areas, i,j); 

			if( isBlack && ((currentArea->AreaSize) < threshold) && !IsNearBorder(currentArea->Points, maskX, maskY)
				|| !isBlack &&((currentArea->AreaSize) < threshold))
			{
				mask[i, j]= !mask[i, j];
			}
		}
	}
	return mask;
}

void PostProcessing(bool* mask, int maskX, int maskY, int threshold)
{
	AreaStruct* blackAreas = GenerateAreas(mask, maskX, maskY, true);
	mask = FillAreas(blackAreas, mask, maskX, maskY, threshold, true);
	AreaStruct* imageAreas = GenerateAreas(mask, maskX, maskY, false);
	mask = FillAreas(imageAreas, mask, maskX, maskY, threshold, false);
}

__global__ void cudaGetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> xGradient, CUDAArray<float> yGradient)
{
	int row = defaultRow();
	int column = defaultColumn();
	float newValue = xGradient.At(row,column)*xGradient.At(row,column) +yGradient.At(row,column)*yGradient.At(row,column);
	newValue = sqrt(newValue);
	magnitude.SetAt(row,column, newValue);
}

void GetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> xGradient, CUDAArray<float> yGradient)
{
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(magnitude.Width,defaultThreadCount),
		ceilMod(magnitude.Height,defaultThreadCount));

	cudaGetMagnitude<<<gridSize,blockSize>>>(magnitude, xGradient, yGradient);
	cudaError_t error = cudaDeviceSynchronize();
}

__global__ void cudaGetMask(CUDAArray<float> initialArray, CUDAArray<bool> mask, int blockSize, float average)
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
		bool result = !(avg < average);
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

void SaveMask(bool* mask,int width, int height, const char* name)
{
	FILE* f = fopen(name,"wb");
	
	char* ar = (char*)malloc(sizeof(char)*(width*2+2)*height);
	int k =0;
	for(int i =0; i<height; i++)
	{
		for(int j =0; j<width; j++)
		{

			ar[k++] = mask[j+i*width]?49:48;
			ar[k++] = ' ';
		}
		ar[k++] = 10;
		ar[k++] = 13;
	}
	//fprintf(
	fwrite(ar, sizeof(char), (width*2+2)*height,f);
	fclose(f);
}


  int main() 
/*void CUDASegmentator(int* img, int imgWidth, int imgHeight, float weightConstant, int windowSize, bool* mask, int maskWidth, int maskHight);*/
  {
	  //parameters
	  float weightConstant = 0.3; 
	  int windowSize = 12;
	  int threshold = 5;

	  int count = 100500;
	  
	  cudaError_t cudaStatus = cudaGetDeviceCount(&count);

	  cudaStatus = cudaSetDevice(0);
	  
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("CUDAArray<float> source = loadImage(...) - ERROR!!!\n");
	  }
	  //source image
	  CUDAArray<float> source = loadImage("C:\\temp\\104_6.bin");
	// CUDAArray<int> source = CUDAArray<int>(img, imgwidth, imgHeight);
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("CUDAArray<float> source = loadImage(...) - ERROR!!!\n");
	  }

	  // imgWidth, imgHeight
	  int xSizeImg = source.Width;		  
	  int ySizeImg = source.Height;

	  // Sobel:	  
	  CUDAArray<float> xGradient = CUDAArray<float>(xSizeImg,ySizeImg);
	  //SaveArray(xGradient,"C:\\temp\\xGradientEmpty.bin");
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("Create xGradient[] - ERROR!!!\n");
	  }

	  CUDAArray<float> yGradient = CUDAArray<float>(xSizeImg,ySizeImg);

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
	  SaveArray(xGradient,"C:\\temp\\xGradient.bin");
	  SaveArray(yGradient,"C:\\temp\\yGradient.bin");

	  //magnitude of gradient
	  CUDAArray<float> magnitude = CUDAArray<float>(xSizeImg,ySizeImg);
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

	  SaveArray(magnitude,"C:\\temp\\magnitude.bin");
	  xGradient.Dispose();
	  yGradient.Dispose();
	  xKernel.Dispose();
	  yKernel.Dispose();

	  //average magnitude 
	  float average = GetAverageFromArray(magnitude);

	  //dementions of mask
	  int N = (int)ceil(((double)source.Width) / windowSize);
	  int M = (int)ceil(((double)source.Height) / windowSize);
	  
	  //thread configuration in CUDA
	  	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
		dim3 gridSize =dim3(ceilMod(N,defaultThreadCount),
							ceilMod(M,defaultThreadCount));

		//mask creation
		CUDAArray<bool> CUDAmask = CUDAArray<bool>(N,M);
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

	  bool* mask = CUDAmask.GetData();
	 // *maskWidth = (int)(CUDAmask.Width);
	 // *maskHight = (int)CUDAmask.Height;

	  PostProcessing(mask, N, M, threshold);

		//save mask
	  SaveMask(mask, (int)(CUDAmask.Width), (int)(CUDAmask.Height), "C:\\temp\\mask.txt");
		
	  CUDAmask.Dispose();
	  cudaDeviceReset(); 
	  return 0;
}


