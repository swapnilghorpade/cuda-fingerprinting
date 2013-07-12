#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionHelper.h"
#include <stdio.h>

#define ceilMod(x, y) (x+y-1)/y
/*
typedef struct 
	{
		int X;
		int Y;
		Point* Next;
	} Point;

typedef struct
	{
		int AreaNumber;
		int AreaSize;
		Point* Points;
		AreaStruct* Next;
	} AreaStruct;
  		
bool IsNearBorder(Point* points, int size, int xBorder, int yBorder)
{
	for (int i = 0; i < size; i++)
    {
		if (points[i].X == 0 || 
			points[i].Y == 0 ||
			points[i].X == xBorder || 
			points[i].Y == yBorder)
		{
			return true;
		}
	}

	return false;
}

void AddPointToArea(AreaStruct* areas, int areaNumber, Point newPoint)
{
	AreaStruct area = areas[areaNumber];

	area.Points[area.AreaSize] = newPoint;
	area.AreaSize++;
	areas[areaNumber] = area;
}

void MergeAreas(CUDAArray<AreaStruct> cudaAreas, int maskX, int areasSize, int i, int j, int areaIndex)
{
	int areaNumberi = 0;
	int areaNumberj = 0;
	AreaStruct* areas = cudaAreas.GetData();
	AreaStruct area = areas[j*maskX + i, 1];

	for (int k = 0; k < areasSize; k++)
	{
		if (area.Points[k].X == i && area.Points[k].Y == j-1)
		{
			areaNumberj = k;
		}
		if (area.Points[k].X == i-1 && area.Points[k].Y == j)
		{
			areaNumberi = k;
		}
	}
                       
	if (areaNumberi != areaNumberj)
	{
		for (int k = 0; k < areas[areaNumberj].AreaSize; k++)
		{
			AddPointToArea(areas, areaNumberi, areas[areaNumberj].Points[k]); 
		}

		for (int k = areaNumberj + 1; k < areaIndex; k++)
		{
			areas[k-1] = areas[k];
		}
	}
    
	Point p = {i,j};
	AddPointToArea(areas, areaNumberi, p); 
	cudaAreas = CUDAArray<AreaStruct>(areas, cudaAreas.Width + 1, cudaAreas.Height);
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

//__global__ void fillArea(CUDAArray<AreaStruct> areas, int areasIndex, int maskX, int maskY, int iSearch, int jSearch, int i, int j, bool isFirst)
//{
//	int columnX = threadIdx.y*blockDim.x + threadIdx.x; 
//
//	if (columnX >= areasIndex)
//	{
//		return;
//	}
//    
//	AreaStruct area;
//	AreaStruct toSetArea;
//
//	cudaError_t cudaStatus = cudaMalloc(&area, sizeof(AreaStruct));
//	if (cudaStatus != cudaSuccess) 
//	{
//		printf("cudaMalloc(&area, sizeof(AreaStruct)); - ERROR!!!\n");
//	}
//
//	cudaStatus = cudaMalloc(&toSetArea, sizeof(AreaStruct));
//	if (cudaStatus != cudaSuccess) 
//	{
//		printf("cudaMalloc(&toSetArea, sizeof(AreaStruct)); - ERROR!!!\n");
//	}
//
//	cudaStatus = cudaMemcpy(area, areas.At(columnX, 1);, sizeof(AreaStruct), cudaMemcpyDeviceToDevice);
//	if (cudaStatus != cudaSuccess) 
//	{
//		printf("cudaMemcpy(area, areas.At(columnX, 1);, sizeof(AreaStruct), cudaMemcpyDeviceToDevice); - ERROR!!!\n");
//	}
//
//	//////////////////////////
//
//	for (int i = 0; i < area.AreaSize; i++)
//	{
//		if (area.Points[i].X == iSearch && area.Points[i].Y == jSearch)
//		{
//			// 
//			area.Points[area.AreaSize + 1].X = i;
//			area.Points[area.AreaSize + 1].Y = j;
//			area.AreaNumber++;
//			toSetArea = areas.At(columnX*maskX + rowY, 1);
//			toSetArea.Points = area.Points;
//			areas.SetAt(columnX*maskX + rowY, 1, toSetArea);
//			return;
//		}
//	}
//}

CUDAArray<AreaStruct> InitializeAreas(CUDAArray<AreaStruct> cudaAreas, int areasSize)
{
	AreaStruct* areas = cudaAreas.GetData();
	AreaStruct area;
	Point* points;

	cudaError_t cudaStatus = cudaMalloc(&area, sizeof(AreaStruct));
	if (cudaStatus != cudaSuccess) 		
	{
		printf("cudaMalloc(&area, sizeof(AreaStruct)); - ERROR!!!\n");
	}

	cudaStatus = cudaMalloc(&points, sizeof(Point) * areaSize);
	if (cudaStatus != cudaSuccess) 		
	{
		printf("cudaMalloc(&point, sizeof(Point) * areaSize); - ERROR!!!\n");
	}

	area = {-1, 0, points};

	for (int i = 0; i < areasSize; i++)
	{
		cudaStatus = cudaMemcpy(areas[i], area, sizeof(AreaStruct), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) 		
		{
			printf("cudaMemcpy(areas[i], area, sizeof(AreaStruct), cudaMemcpyHostToDevice); - ERROR!!!\n");
		}
	}
}

AreaStruct* GenerateAreas(bool* mask, int maskX, int maskY, bool isBlack)
{
	//int areasSize = maskX * maskY + 1;
	int areaIndex = 0;
	bool isLeftImageTopBlack = false, isLeftBlackTopImage = false, isLeftBlackTopBlack = false;
	//bool* mask = cudaMask.GetData();
	//CUDAArray<AreaStruct> cudaAreas = CUDAArray<AreaStruct>(areasSize, 1);

	//InitializeAreas(cudaAreas, areasSize);

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
				MergeAreas(cudaAreas, maskX, areasSize, i, j, areaIndex);
				areaIndex--;
				continue;
            }
			//cudaError_t cudaStatus;
			if (isLeftBlackTopImage || isLeftImageTopBlack)
            {
				//dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
				//dim3 gridSize = dim3(ceilMod(areaIndex,defaultThreadCount));

				if (isLeftBlackTopImage)
				{

					fillArea(cudaAreas, areaIndex, maskX, maskY, i-1, j, i, j);
				}
				else
				{
					fillArea(cudaAreas, areaIndex, maskX, maskY, i, j-1, i, j);
				}
				//cudaStatus = cudaDeviceSynchronize();
				//cudaStatus = cudaGetLastError();




				for (int i = 0; i < area.AreaSize; i++)
//	{
//		if (area.Points[i].X == iSearch && area.Points[i].Y == jSearch)
//		{
//			// 
//			area.Points[area.AreaSize + 1].X = i;
//			area.Points[area.AreaSize + 1].Y = j;
//			area.AreaNumber++;
//			toSetArea = areas.At(columnX*maskX + rowY, 1);
//			toSetArea.Points = area.Points;
//			areas.SetAt(columnX*maskX + rowY, 1, toSetArea);
//			return;
//		}
//	}




                continue;
            }
			

			Point* initialPoints;
			 cudaStatus = cudaMalloc(&initialPoints, sizeof(Point) * areasSize);
				cudaStatus = cudaDeviceSynchronize();
				cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) 
			{
				printf("cudaMalloc(&initialPoints, sizeof(Point) * areasSize); - ERROR!!!\n");
			}

			Point newPoint = {i, j};
			cudaStatus = cudaMemcpy(initialPoints, &newPoint, sizeof(Point), cudaMemcpyHostToDevice);
			//initialPoints[0] =  newPoint;

			AreaStruct newArea = {areaIndex, 1, initialPoints};
			AreaStruct*	areas = cudaAreas.GetData();
			areas[areaIndex++] = newArea;

			cudaAreas.Dispose();
			cudaAreas = CUDAArray<AreaStruct>(areas, areasSize, 1);
			cudaFree(initialPoints);
			cudaFree(areas);
		}
	}
	

	return cudaAreas;
} 

//__global__ void changeColor(CUDAArray<bool> mask, CUDAArray<Point> toRestore, int toRestoreCounter)
//{
//	// coordinates of points in dev_toRestores
//	int columnX = blockIdx.x*blockIdx.y*blockDim.x+threadIdx.y*blockDim.x + threadIdx.x; 
//
//	if (columnX >= toRestoreCounter)
//	{
//		return;
//	}
//
//	Point point = toRestore.At(columnX, 1);
//	
//	mask.SetAt(point.X, point.Y, !(mask.At(point.X, point.Y)));
//}

CUDAArray<bool> FillAreas(CUDAArray<AreaStruct> cudaAreas, CUDAArray<bool> cudaMask, int maskX, int maskY, int threshold)
{
	int maskSize = maskX*maskY + 1;
	int toRestoreCounter = 0;
	int newRestorePoints = 0;
	cudaError_t cudaStatus;
	AreaStruct* areas = cudaAreas.GetData();
	Point* toRestore;

	cudaStatus = cudaMalloc(&toRestore, sizeof(Point) * maskSize);

	if (cudaStatus != cudaSuccess) 
	{
		printf("cudaMalloc(&toRestore, sizeof(Point) * maskSize); - ERROR!!!\n");
	}
		
	for(int i = 0; i < maskSize; i++)
	{
		newRestorePoints = 0;

		if (areas[i].AreaSize < threshold && 
			!IsNearBorder(areas[i].Points, areas[i].AreaSize, maskX, maskY))
        {
			while(newRestorePoints <= areas[i].AreaSize)
			{
				toRestore[toRestoreCounter] = areas[i].Points[newRestorePoints]; 
				toRestoreCounter++;
			}
		}
	}

	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(toRestoreCounter, defaultThreadCount));
	CUDAArray<Point> cudaToRestore = CUDAArray<Point>(toRestore, maskSize, 1);

	changeColor<<<gridSize, blockSize>>>(cudaMask, cudaToRestore, toRestoreCounter);
	cudaToRestore.Dispose();
	cudaFree(toRestore);

	return cudaMask;
}

void PostProcessing(bool* mask, int maskX, int maskY, int threshold)
{
	AreaStruct* blackAreas = GenerateAreas(mask, maskX, maskY, true);
	cudaMask = FillAreas(blackAreas, mask, maskX, maskY, threshold);
	AreaStruct* imageAreas = GenerateAreas(mask, maskX, maskY, false);
	cudaMask = FillAreas(imageAreas, mask, maskX, maskY, threshold);
}
*/
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
	dim3 gridSize = 
		dim3(ceilMod(magnitude.Width,defaultThreadCount),
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

  int main() /*char* pathToImg, float weightConstant, int windowSize, int threshold,//parmeters
	  bool* mask,int* maskWidth, int* maskHight) //result*/
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
	  //CUDAArray<float> source = loadImage(pathToImg);
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("CUDAArray<float> source = loadImage(...) - ERROR!!!\n");
	  }

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

	 // PostProcessing(mask, N, M, threshold);

		//save mask
	  SaveMask(mask, (int)(CUDAmask.Width), (int)(CUDAmask.Height), "C:\\temp\\mask.txt");
		
	  CUDAmask.Dispose();
	  cudaDeviceReset(); 
	  return 0;
}


