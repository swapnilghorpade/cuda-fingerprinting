
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "ConvolutionHelper.h"

extern "C"{

__declspec(dllexport) int main();

}

int main()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

	FILE* f = fopen("C:\\temp\\104_6.bin","rb");

	int width;
	int height;

	fread(&width,sizeof(int),1,f);
	fread(&height,sizeof(int),1,f);

	int* ar = (int*)malloc(sizeof(int)*width*height);
	float* ar2 = (float*)malloc(sizeof(float)*width*height);

	fread(ar,sizeof(int),width*height,f);
	for(int i=0;i<width*height;i++)
	{
		ar2[i]=ar[i];
	}
	fclose(f);
	CUDAArray<float> sourceImage = CUDAArray<float>(ar2,width,height);

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

	float* arTest = g1.GetData();
	f = fopen("C:\\temp\\104_6_l1.bin","wb");
	fwrite(&g1.Width,sizeof(int),1,f);
	fwrite(&g1.Height,sizeof(int),1,f);
	for(int i=0;i<g1.Width*p1.Height;i++)
	{
		float value = (float)arTest[i];
		int result = fwrite(&value,sizeof(float),1,f);
		result++;
	}
	fclose(f);
	sourceImage.Dispose();
	g1.Dispose();
	g2.Dispose();
	g3.Dispose();
	g4.Dispose();
	p1.Dispose();
	p2.Dispose();
	p3.Dispose();
	free(ar);
	free(ar2);
	free(arTest);
	cudaDeviceReset();
    return 0;
}