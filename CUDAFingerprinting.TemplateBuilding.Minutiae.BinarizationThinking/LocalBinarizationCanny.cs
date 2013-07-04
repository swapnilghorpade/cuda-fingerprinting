using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking
{
    public static class LocalBinarizationCanny
    {
        public static double[,] Smoothing(double[,] img, double sigma)
        {
            var kernel = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, sigma),
                                                   KernelHelper.GetKernelSizeForGaussianSigma(sigma));
            double [,] data = ConvolutionHelper.Convolve(img, kernel);

            return data;
        }

        public static double[,] Sobel(double[,] img)
        {

            double [,] sobData = new double[img.GetLength(0),img.GetLength(1)];

            double[,] gX = new double[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            double[,] gY = new double[,] { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

            for (int i = 1; i < img.GetLength(0) - 1; i++)
                for (int j = 1; j < img.GetLength(1) - 1; j++)
                {
                    double newX = 0, newY = 0;

                    for (int h = 0; h < 3; h++)
                    {
                        for (int w = 0; w < 3; w++)
                        {
                            double curr = img[i + h - 1,j + w - 1];
                            newX += gX[h, w] * curr;
                            newY += gY[h, w] * curr;
                        }
                    }

                    if (newX * newX + newY * newY > 128 * 128)
                    {
                        sobData[i, j] = 255;
                    }
                    else
                    {
                        sobData[i, j] = 0;
                    }

                }
            return sobData;
        }
    }
}
