#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionHelper.h"
#include <stdio.h>

#define ceilMod(x, y) (x+y-1)/y

//typedef struct 
//	{
//		int X;
//		int Y;
//	} Point;
//
//typedef struct
//	{
//		int AreaNumber;
//		int AreaSize;
//		Point* Points;
//	} AreaStruct;
//  			
//bool IsNearBorder(Point* points, int size, int xBorder, int yBorder)
//{
//	for (int i = 0; i < size; i++)
//    {
//		if (points[i].X == 0 || 
//			points[i].Y == 0 ||
//			points[i].X == xBorder || 
//			points[i].Y == yBorder)
//		{
//			return true;
//		}
//	}
//
//	return false;
//}
//
//void AddPointToArea(CUDAArray<AreaStruct> areas, int areaNumber, int pointNumber, Point newPoint)
//{
//	AreaStruct area = areas.At(areaNumber, 1);
//
//	area.Points[area.AreaSize] = newPoint;
//	area.AreaSize++;
//	areas.SetAt(areaNumber, 1, area);
//}
//
//void MergeAreas(CUDAArray<AreaStruct> areas, AreaStruct initialValue, int maskX, int areasSize, int i, int j, int areaIndex)
//{
//	int areaNumberi = 0;
//	int areaNumberj = 0;
//	AreaStruct area = areas.At(j*maskX + i, 1);
//
//	for (int k = 0; k < areasSize; k++)
//	{
//		if (area.Points[k].X == i && area.Points[k].Y == j-1)
//		{
//			areaNumberj = k;
//		}
//		if (area.Points[k].X == i-1 && area.Points[k].Y == j)
//		{
//			areaNumberi = k;
//		}
//	}
//                       
//	if (areaNumberi != areaNumberj)
//	{
//		for (int k = 0; k < (areas.At(areaNumberj, 1)).AreaSize; k++)
//		{
//			AddPointToArea(areas, areaNumberi, 
//				(areas.At(areaNumberi, 1)).AreaSize, 
//				(areas.At(areaNumberj, 1)).Points[k]
//			);
//		}
//
//		areas.SetAt(areaNumberj, 1, initialValue);
//
//		for (int k = areaNumberj + 1; k < areaIndex; k++)
//		{
//			areas.SetAt(k-1, 1, areas.At(k, 1));
//		}
//	}
//    
//	Point p = {i,j};
//	AddPointToArea(areas, areaNumberi, (areas.At(areaNumberi, 1)).AreaSize, p);
//}
//
//bool IsLeftImageTopBlack(int i, int j, bool topValue, bool leftValue, bool isBlack) 
//{
//	return (j - 1 >= 0 && (topValue || isBlack) && !(topValue && isBlack) &&					//top block is black 
//           (i - 1 >= 0 && (leftValue || !isBlack) && !(leftValue && !isBlack)) || i - 1 < 0);	//left block is not black or not exist
//}
//
//bool IsLeftBlackTopImage(int i, int j, bool topValue, bool leftValue, bool isBlack) 
//{
//	return (i - 1 >= 0 && (leftValue || isBlack) && !(leftValue && isBlack) &&					//left block is black
//           (j - 1 >= 0 && (topValue || !isBlack) && !(topValue && !isBlack)) || j - 1 < 0);	    //top block is not black or not exist
//}
//
//bool IsLeftBlackTopBlack(int i, int j, bool topValue, bool leftValue, bool isBlack)
//{
//	return (j - 1 >= 0 && (topValue || isBlack) && !(topValue && isBlack) &&					//top block is black 
//            i - 1 >= 0 && (leftValue || isBlack) && !(leftValue && isBlack));					//left block is black
//}
//
//AreaStruct MakeInitialValue(CUDAArray<AreaStruct> areas, CUDAArray<Point> initialPoints, int areasSize)
//{
//	Point point = {-1,-1};
//	cudaError_t cudaStatus;
//
//	for (int i = 0; i < areasSize; i++)
//	{
//		initialPoints.SetAt(i, 1, point);
//	}
//
//	AreaStruct initialValue = {-1, 0, initialPoints.GetData()};
//	
//	for (int i = 0; i < areasSize; i++)
//	{
//		areas.SetAt(i, 1, initialValue);
//	}
//
//	return initialValue;
//}
//
//__global__ void fillArea(CUDAArray<AreaStruct> areas, int areasSize, int maskX, int iSearch, int jSearch, int i, int j, bool isFirst)
//{
//    int columnX = defaultColumn();
//	int rowY = defaultRow();
//	AreaStruct area = areas.At(rowY*maskX + columnX, 1);
//	AreaStruct toSetArea;
//
//	if (isFirst)
//	{
//		iSearch = i-1;
//		jSearch = j;
//	}
//	
//	for (int i = 0; i < areasSize; i++)
//	{
//		if (area.Points[i].X == iSearch && area.Points[i].Y == jSearch)
//		{
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
//
//CUDAArray<AreaStruct> GenerateAreas(CUDAArray<bool> mask, int maskX, int maskY, bool isBlack)
//{
//	int areasSize = maskX * maskY + 1;
//	int areaIndex = 0;
//	bool isLeftImageTopBlack, isLeftBlackTopImage, isLeftBlackTopBlack;
//	CUDAArray<AreaStruct> areas = CUDAArray<AreaStruct>(areasSize, 1);
//	CUDAArray<Point> initialPoints = CUDAArray<Point>(areasSize, 1);
//	AreaStruct initialValue = MakeInitialValue(areas, initialPoints, areasSize);
//	cudaError_t cudaStatus;
//
//	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
//	dim3 gridSize = dim3(ceilMod(maskX,defaultThreadCount), ceilMod(maskY,defaultThreadCount));
//	
//	for (int i = 0; i < maskX; i++)
//    {
//		for (int j = 0; j < maskY; j++)
//        {
//			if (!mask.At(i, j) && isBlack || mask.At(i, j) && !isBlack)
//            {
//				continue;
//            }
//			
//			isLeftBlackTopImage = IsLeftBlackTopImage(i, j, mask.At(i, j - 1), mask.At(i - 1, j), isBlack);
//			isLeftImageTopBlack = IsLeftImageTopBlack(i, j, mask.At(i, j - 1), mask.At(i - 1, j), isBlack);
//			isLeftBlackTopBlack = IsLeftBlackTopBlack(i, j, mask.At(i, j - 1), mask.At(i - 1, j), isBlack);
//
//			if (isLeftBlackTopBlack)
//            {
//				MergeAreas(areas, initialValue, maskX, areasSize, i, j, areaIndex);
//				areaIndex--;
//
//				continue;
//            }
//
//			if (isLeftBlackTopImage || isLeftImageTopBlack)
//            {
//				if (isLeftBlackTopImage)
//				{
//					fillArea<<<gridSize, blockSize>>>(areas, areasSize, maskX, i-1, j, i, j, true);
//				}
//				else
//				{
//					fillArea<<<gridSize, blockSize>>>(areas, areasSize, maskX, i, j-1, i, j, false);
//				}
//
//                continue;
//            }
//
//			Point newPoint = {i, j};
//			Point emptyPoint = {0, 0};
//			initialPoints.SetAt(0,1, newPoint);
//			AreaStruct newArea = {areaIndex, 1, initialPoints.GetData()};
//			areas.SetAt(areaIndex++, 1, newArea);
//			initialPoints.SetAt(0,1, emptyPoint);
//		}
//	}
//
//	initialPoints.Dispose();
//	
//	return areas;
//} 
//
//__global__ void changeColor(CUDAArray<bool> mask, CUDAArray<Point> toRestore, int toRestoreCounter)
//{
//	// coordinates of points in dev_toRestores
//	int columnX = blockIdx.x*blockIdx.y*blockDim.x+threadIdx.y*blockDim.x + threadIdx.x;  
//	Point point = toRestore.At(columnX, 1);
//	
//	mask.SetAt(point.X, point.Y, !(mask.At(point.X, point.Y)));
//}
//
//CUDAArray<bool> FillAreas(CUDAArray<AreaStruct> areas, CUDAArray<bool> mask, int maskX, int maskY, int threshold)
//{
//	int maskSize = maskX*maskY + 1;
//	int toRestoreCounter = 0;
//	int newRestorePoints = 0;
//	cudaError_t cudaStatus;
//	CUDAArray<Point> toRestore = CUDAArray<Point>(maskSize, 1);
//		
//	for(int i = 0; i < maskSize; i++)
//	{
//		newRestorePoints = 0;
//
//		if ((areas.At(i,1)).AreaSize < threshold && 
//			!IsNearBorder((areas.At(i,1)).Points, (areas.At(i,1)).AreaSize, maskX, maskY))
//        {
//			while(newRestorePoints <= (areas.At(i,1)).AreaSize)
//			{
//				toRestore.SetAt(toRestoreCounter, 1, (areas.At(i, 1)).Points[newRestorePoints]); 
//				toRestoreCounter++;
//			}
//		}
//	}
//
//	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
//	dim3 gridSize = dim3(ceilMod(toRestoreCounter, defaultThreadCount));
//
//	changeColor<<<gridSize, blockSize>>>(mask, toRestore, toRestoreCounter);
//	toRestore.Dispose();
//
//	return mask;
//}
//
//CUDAArray<bool> PostProcessing(CUDAArray<bool> mask, int maskX, int maskY, int threshold)
//{
//	CUDAArray<AreaStruct> blackAreas = GenerateAreas(mask, maskX, maskY, true);
//	mask = FillAreas(blackAreas, mask, maskX, maskY, threshold);
//	CUDAArray<AreaStruct> imageAreas = GenerateAreas(mask, maskX, maskY, false);
//	mask = FillAreas(imageAreas, mask, maskX, maskY, threshold);
//
//	return mask;
//}

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

void SaveMask(CUDAArray<bool> mask, const char* name)
{
	FILE* f = fopen(name,"wb");
	//fclose(f);
	bool* maskOnCPU = mask.GetData();
	char* ar = (char*)malloc(sizeof(char)*(mask.Width*2+2)*mask.Height);
	int k =0;
	for(int i =0; i<mask.Height; i++)
	{
		for(int j =0; j<mask.Width; j++)
		{

			ar[k++] = maskOnCPU[j+i*mask.Width]?49:48;
			ar[k++] = ' ';
		}
		ar[k++] = 10;
		ar[k++] = 13;
	}
	//fprintf(
	fwrite(ar, sizeof(char), (mask.Width*2+2)*mask.Height,f);
	fclose(f);
}

  int main()
  {
	  
	  //parameters
	  float weightConstant = 0.3; 
	  int windowSize = 12;
	  int threshold = 5;

	  int count = 100500;
	  
	  cudaError_t cudaStatus = cudaGetDeviceCount(&count);

	  cudaStatus =cudaSetDevice(0);

	  
	  cudaStatus = cudaGetLastError();
	  if (cudaStatus != cudaSuccess) 
	  {
		printf("CUDAArray<float> source = loadImage(...) - ERROR!!!\n");
	  }
	  //source image
	  CUDAArray<float> source = loadImage("C:\\temp\\104_6.bin");

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
		CUDAArray<bool> mask = CUDAArray<bool>(N,M);
		 if (cudaStatus != cudaSuccess) 
	  {
		printf("create Mask - ERROR!!!\n");
	  }
		cudaGetMask<<<gridSize,blockSize>>>(magnitude, mask, windowSize, average*weightConstant);
	
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

	  //mask = PostProcessing(mask, N, M, threshold);

		//save mask
		SaveMask(mask, "C:\\temp\\mask.txt");
		mask.Dispose();
		cudaDeviceReset();

		return 0;
}


