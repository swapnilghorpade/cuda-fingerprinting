
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ConvolutionHelper.h"
//#include "CUDAArray.h"
#include <stdio.h>

//Ура! Вперед, к светлому будущему параллельных вычислений!

/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/

  int main(float* img, int xSizeImg, int ySizeImg, int windowSize, double weightConstant, int threshold)
  {
	  // Sobel:
	  CUDAArray<float> source = CUDAArray<float>(img,xSizeImg,ySizeImg);

	  CUDAArray<float> xGradient = CUDAArray<float>(xSizeImg,ySizeImg);
	  CUDAArray<float> xGradient = CUDAArray<float>(xSizeImg,ySizeImg);

	  float xKernelCPU[3][3] = {{-1,0,1},
							{-2,0,2},
							{-1,0,1}};
	  CUDAArray<float> xKernel = CUDAArray<float>(&xKernelCPU,3,3);
	  
	  float yKernel[3][3] = {{-1,-2,-1},
							{0,0,0},
							{1,2,1}};

	  cudaConvolve(xGradient, source, xFilter);

	//int* xGradients = (int*)malloc((xSizeImg* ySizeImg +1)*sizeof(int)); 
      //      int* yGradients = (int*)malloc((xSizeImg* ySizeImg +1)*sizeof(int));

            //xGradients = OrientationFieldGenerator.GenerateXGradients(img.Select2D(a => (int)a));
            //yGradients = OrientationFieldGenerator.GenerateYGradients(img.Select2D(a => (int)a));
         /*   double[,] magnitudes = xGradients.Select2D((value, x, y) => Math.Sqrt(xGradients[x, y] * xGradients[x, y] + yGradients[x, y] * yGradients[x, y]));
            double averege = KernelHelper.Average(magnitudes);
            double[,] window = new double[windowSize, windowSize];

            int N = (int)Math.Ceiling(((double)img.GetLength(0)) / windowSize);
            int M = (int)Math.Ceiling(((double)img.GetLength(1)) / windowSize);
            bool[,] mask = new bool[N, M];

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    window = window.Select2D((x, y, value) =>
                    {
                        if (i * windowSize + x >= magnitudes.GetLength(0)
                           || j * windowSize + y >= magnitudes.GetLength(1))
                        {
                            return 0;
                        }

                        return magnitudes[(int)(i * windowSize + x), j * windowSize + y];
                    });

                    if (KernelHelper.Average(window) < averege * weightConst * weight[i, j])
                    {
                        mask[i, j] = false;
                    }
                    else
                    {
                        mask[i, j] = true;
                    }
                }
            }

            return PostProcessing(mask, threshold);
			*/
			return 0;
        }
/*
        private static Tuple<int, int> FindAveragePoint(bool[,] mask, double[,] img, int windowSize)
        {
            int xLength = img.GetLength(0);
            int yLength = img.GetLength(1);
            int xAveragePoint = 0;
            int yAveragePoint = 0;
            int xBlock = 0;
            int yBlock = 0;

            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < yLength; j++)
                {
                    xBlock = (int)(((double)i) / windowSize);
                    yBlock = (int)(((double)j) / windowSize);

                    if (mask[xBlock, yBlock])
                    {
                        xAveragePoint += i;
                        yAveragePoint += j;
                    }
                }
            }

            return new Tuple<int, int>(xAveragePoint / (xLength * yLength), yAveragePoint / (xLength * yLength));
        }

        private static double[,] ColorImage(double[,] img, bool[,] mask, int windowSize)
        {
            img = img.Select2D((value, x, y) =>
                {
                    int xBlock = (int)(((double)x) / windowSize);
                    int yBlock = (int)(((double)y) / windowSize);
                    return mask[xBlock, yBlock] ? img[x, y] : 0;
                });

            return img;
        }

        private static bool[,] PostProcessing(bool[,] mask, int threshold)
        {
            var blackAreas = GenerateBlackAreas(mask);
            var toRestore = new List<Tuple<int, int>>();


            foreach (var blackArea in blackAreas)
            {
                if (blackArea.Value.Count < threshold && !IsNearBorder(blackArea.Value, mask.GetLength(0), mask.GetLength(1)))
                {
                    toRestore.AddRange(blackArea.Value);
                }
            }

            var newMask = ChangeColor(toRestore, mask);
            var imageAreas = GenerateImageAreas(newMask);
            toRestore.Clear();


            foreach (var imageArea in imageAreas)
            {
                if (imageArea.Value.Count < threshold && !IsNearBorder(imageArea.Value, mask.GetLength(0), mask.GetLength(1)))
                {
                    toRestore.AddRange(imageArea.Value);
                }
            }

            return ChangeColor(toRestore, newMask);
        }

        private static Dictionary<int, List<Tuple<int, int>>> GenerateBlackAreas(bool[,] mask)
        {
            Dictionary<int, List<Tuple<int, int>>> areas = new Dictionary<int, List<Tuple<int, int>>>();

            int areaIndex = 0;

            for (int i = 0; i < mask.GetLength(0); i++)
            {
                for (int j = 0; j < mask.GetLength(1); j++)
                {
                    if (mask[i, j])
                    {
                        continue;
                    }
                    if (i - 1 >= 0 && !mask[i - 1, j]                   //left block is black
                        && (j - 1 >= 0 && mask[i, j - 1] || j - 1 < 0)) //top block is not black or not exist
                    {
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i - 1, j)))
                            {
                                area.Value.Add((new Tuple<int, int>(i, j)));
                            }
                        }
                        continue;
                    }
                    if (j - 1 >= 0 && !mask[i, j - 1]                   //top block is black 
                        && (i - 1 >= 0 && mask[i - 1, j] || i - 1 < 0)) //left block is not black or not exist
                    {
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i, j - 1)))
                            {
                                area.Value.Add((new Tuple<int, int>(i, j)));
                            }
                        }
                        continue;

                    }
                    if (j - 1 >= 0 && !mask[i, j - 1]        //top block is black 
                        && i - 1 >= 0 && !mask[i - 1, j])    //left block is black
                    {
                        int areaNumberi = 0;
                        int areaNumberj = 0;
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i, j - 1)))
                            {
                                areaNumberj = area.Key;
                            }
                            if (area.Value.Contains(new Tuple<int, int>(i - 1, j)))
                            {
                                areaNumberi = area.Key;
                            }
                        }

                        if (areaNumberi != areaNumberj)
                        {
                            areas[areaNumberi].AddRange(areas[areaNumberj]);
                            areas[areaNumberi] = new List<Tuple<int, int>>(areas[areaNumberi].Distinct());
                            areas.Remove(areaNumberj);
                        }

                        areas[areaNumberi].Add(new Tuple<int, int>(i, j));
                        continue;
                    }
                    areas.Add(areaIndex, new List<Tuple<int, int>>());
                    areas[areaIndex].Add(new Tuple<int, int>(i, j));
                    areaIndex++;
                }

            }
            return areas;
        }

        private static Dictionary<int, List<Tuple<int, int>>> GenerateImageAreas(bool[,] mask)
        {
            var areas = new Dictionary<int, List<Tuple<int, int>>>();
            int areaIndex = 0;

            for (int i = 0; i < mask.GetLength(0); i++)
            {
                for (int j = 0; j < mask.GetLength(1); j++)
                {
                    if (!mask[i, j])
                    {
                        continue;
                    }

                    if (i - 1 >= 0 && mask[i - 1, j] && (j - 1 >= 0 && !mask[i, j - 1] || j - 1 < 0))
                    {
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i - 1, j)))
                            {
                                area.Value.Add((new Tuple<int, int>(i, j)));
                            }
                        }
                        continue;
                    }

                    if (j - 1 >= 0 && mask[i, j - 1] && (i - 1 >= 0 && !mask[i - 1, j] || i - 1 < 0))
                    {
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i, j - 1)))
                            {
                                area.Value.Add((new Tuple<int, int>(i, j)));
                            }
                        }
                        continue;

                    }

                    if (j - 1 >= 0 && mask[i, j - 1] && i - 1 >= 0 && mask[i - 1, j])
                    {
                        int areaNumberi = 0;
                        int areaNumberj = 0;

                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i, j - 1)))
                            {
                                areaNumberj = area.Key;
                            }
                            if (area.Value.Contains(new Tuple<int, int>(i - 1, j)))
                            {
                                areaNumberi = area.Key;
                            }
                        }

                        if (areaNumberi != areaNumberj)
                        {
                            areas[areaNumberi].AddRange(areas[areaNumberj]);
                            areas[areaNumberi] = new List<Tuple<int, int>>(areas[areaNumberi].Distinct());
                            areas.Remove(areaNumberj);
                        }
                        areas[areaNumberi].Add(new Tuple<int, int>(i, j));
                        continue;
                    }

                    areas.Add(areaIndex, new List<Tuple<int, int>>());
                    areas[areaIndex].Add(new Tuple<int, int>(i, j));
                    areaIndex++;
                }
            }

            return areas;
        }

        private static bool IsNearBorder(List<Tuple<int, int>> areas, int xBorder, int yBorder)
        {
            return areas.FindAll(area => area.Item1 == 0 ||
                                         area.Item2 == 0 ||
                                         area.Item1 == xBorder ||
                                         area.Item2 == yBorder
                                 ).Any();
        }

        private static bool[,] ChangeColor(List<Tuple<int, int>> areas, bool[,] mask)
        {
            foreach (var area in areas)
            {
                mask[area.Item1, area.Item2] = !mask[area.Item1, area.Item2];
            }

            return mask;
        }
		*/
   