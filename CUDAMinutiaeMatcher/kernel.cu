#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "MinutiaExtraction.h"
#include <time.h>
extern "C"{

__declspec(dllexport) int main();

}

const float tau1 = 0.1f;
const float tau2 = 0.3f;

CUDAArray<float> EnhanceImage(CUDAArray<float> sourceImage)
{
	CUDAArray<float> g1 = Reduce(sourceImage,1.7f);
	CUDAArray<float> g2 = Reduce(g1,1.21f);
	CUDAArray<float> g3 = Reduce(g2,1.3f);
	CUDAArray<float> g4 = Reduce(g3,1.3f);

	CUDAArray<float> p3 = Expand(g4, 1.3f, g3.Width, g3.Height);
	CUDAArray<float> p2 = Expand(g3, 1.3f, g2.Width, g2.Height);
	CUDAArray<float> p1 = Expand(g2, 1.21f,g1.Width, g1.Height);
	
	SubtractArray(g3,p3);
	EnhanceContrast(g3);
	SubtractArray(g2,p2);
	EnhanceContrast(g2);
	SubtractArray(g1,p1);
	EnhanceContrast(g1);

	CUDAArray<float> ls1Real;
	CUDAArray<float> ls1Im;
	EstimateLS(&ls1Real, &ls1Im, g1, 0.6f, 3.2f);
	
	CUDAArray<float> ls2Real;
	CUDAArray<float> ls2Im;
	EstimateLS(&ls2Real, &ls2Im, g2, 0.6f, 3.2f);

	CUDAArray<float> ls3Real;
	CUDAArray<float> ls3Im;
	EstimateLS(&ls3Real, &ls3Im, g3, 0.6f, 3.2f);

	CorrectLS1WithLS2(ls1Real, ls1Im, ls2Real, ls2Im);
	
	DirectionFiltering(g1, ls1Real, ls1Im, tau1, tau2);
	
	DirectionFiltering(g2, ls2Real, ls2Im, tau1, tau2);
	
	DirectionFiltering(g3, ls3Real, ls3Im, tau1, tau2);

	CUDAArray<float> el3 = Expand(g3, 1.3f, g2.Width, g2.Height);
	AddArray(el3,g2);
	CUDAArray<float> el2 = Expand(el3, 1.21f,g1.Width, g1.Height);
	el3.Dispose();
	AddArray(el2,g1);
	CUDAArray<float> enhanced = Expand(el2, 1.7f,sourceImage.Width, sourceImage.Height);
	el2.Dispose();
	EnhanceContrast(enhanced);

	FixValues(enhanced);

	g1.Dispose();
	g2.Dispose();
	g3.Dispose();
	g4.Dispose();
	p1.Dispose();
	p2.Dispose();
	p3.Dispose();
	return enhanced;
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

int main()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

	char* buf = (char*)malloc(sizeof(char)*50);
	float time = 0;
	//for(int  i=1;i<=110;i++)
	//{
	//	for(int j=1;j<=8;j++)
	//	{
	//		sprintf(buf,"C:\\temp\\bin\\%d_%d.bin",i,j);

	//		cudaDeviceReset();
	//		FillDirections();

	//		CUDAArray<float> sourceImage = loadImage(buf);

	//		clock_t t1 = clock();

	//		CUDAArray<float> enhanced = EnhanceImage(sourceImage);

	//		clock_t t2 = clock();
	//		
	//		float dt = (float)(t2-t1) / CLOCKS_PER_SEC;
	//		time+=dt;
	//		sprintf(buf,"C:\\temp\\enh_bin\\%d_%d.bin",i,j);

	//		SaveArray(enhanced, buf);
	//		sourceImage.Dispose();
	//		enhanced.Dispose();
	//	}
	//}

	//time /= 880;

	// minutia extraction

	time = 0;

	for(int  i=1;i<=110;i++)
	{
		for(int j=1;j<=8;j++)
		{
			cudaDeviceReset();

			sprintf(buf,"C:\\temp\\enh_bin\\%d_%d.bin",i,j);

			CUDAArray<float> sourceImage = loadImage(buf,true);
				
			clock_t t1 = clock();

			int* xs;
			int* ys;

			ExtractMinutiae(&xs, &ys, sourceImage);

			time+= (float)(clock()-t1) / CLOCKS_PER_SEC;

			sprintf(buf,"C:\\temp\\min\\%d_%d.min",i,j);
			FILE* f = fopen(buf,"wb");
			int y = 32;
			fwrite(&y,sizeof(int),1,f);
			for(int i=0;i<32;i++)
			{
				int result = fwrite(xs+i,sizeof(int),1,f);
				result = fwrite(ys+i,sizeof(int),1,f);
			}
			fclose(f);
			free(xs);
			free(ys);

			sourceImage.Dispose();
		}
	}

	time /= 880;

	//// minutia matching
	
	int* dBaseX = (int*)malloc(sizeof(int)*32*880);
	int* dBaseY = (int*)malloc(sizeof(int)*32*880);
	
	int ptrX = 0, ptrY = 0;

	for(int  i=1;i<=110;i++)
	{
		for(int j=1;j<=8;j++)
		{
			sprintf(buf,"C:\\temp\\min\\%d_%d.min",i,j);

			FILE* f = fopen(buf,"rb");

			int amount = 0;

			fread(&amount, sizeof(int), 1, f);
			
			for(int n = 0; n< amount; n++)
			{
				fread(dBaseX+ptrX++,sizeof(int), 1,f);
				fread(dBaseY+ptrY++,sizeof(int), 1,f);
			}

			fclose(f);
		}
	}

	int* same = (int*)malloc(sizeof(int)*33);
	memset(same,0,33*sizeof(int));

	int* different = (int*)malloc(sizeof(int)*33);
	memset(different,0,33*sizeof(int));

	time = 0;

	int totalMatches = 0;

	for(int i=0; i<879; i++)
	{
		cudaDeviceReset();

		int matches = 879-i;

		if(matches%100==0)printf("DBASE size is %d\n",matches);

		totalMatches+=matches;

		CUDAArray<int> cudaBaseX = CUDAArray<int>(dBaseX+32*(1+i),32,879-i);
		CUDAArray<int> cudaBaseY = CUDAArray<int>(dBaseY+32*(1+i),32,879-i);

		clock_t t1 = clock();

		CUDAArray<int> result = MatchFingers(dBaseX+32*i,dBaseY+32*i, cudaBaseX, cudaBaseY);

		int* resultLocal = result.GetData();

		time+= clock()-t1;

		for(int j=0;j<result.Width*result.Height;j++)
		{
			if(i/8 == (i+1+j)/8)same[resultLocal[j]]++;
			else different[resultLocal[j]]++;
		}

		cudaBaseX.Dispose();
		cudaBaseY.Dispose();
		result.Dispose();
		free(resultLocal);
	}

	time /= CLOCKS_PER_SEC;
	time /= totalMatches;

	free(dBaseX);
	free(dBaseY);

	FILE* f1 = fopen("C:\\temp\\ZeeBigResult.bin","wb");
	
	fwrite(same, sizeof(int), 33, f1);
	fwrite(different, sizeof(int), 33, f1);

	fclose(f1);

	cudaDeviceReset();
	free(buf);
    return 0;
}