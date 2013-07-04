using System;
using System.IO;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.Common.Segmentation
{
    public static class Segmentator
    {
        public static double[,] Segmetator(double[,] img, int windowSize, double weight)
        {
            int[,] xGradients = OrientationFieldGenerator.GenerateXGradients(img.Select2D(a => (int)a));
            int[,] yGradients = OrientationFieldGenerator.GenerateYGradients(img.Select2D(a => (int)a));
            double[,] magnitudes = xGradients.Select2D((value, x, y) => Math.Sqrt(xGradients[x, y] * xGradients[x, y] + yGradients[x, y] * yGradients[x, y]));
            double averege = KernelHelper.Average(magnitudes);
            double[,] window = new double[windowSize, windowSize];
            double[,] result = new double[img.GetLength(0), img.GetLength(1)];

            for (int i = 0; i < magnitudes.GetLength(0); i += windowSize)
            {
                for (int j = 0; j < magnitudes.GetLength(1); j += windowSize)
                {
                    window = window.Select2D((x, y, value) =>
                    {
                        if (i + x >= magnitudes.GetLength(0)
                           || j + y >= magnitudes.GetLength(1))
                        {
                            return 0;
                        }

                        return magnitudes[(int)(i + x), j + y];
                    });

                    if (KernelHelper.Average(window) < averege*weight)
                    {
                        result = result.Select2D((value, x, y) =>
                            {
                                if (x < i + windowSize && x >= i &&
                                    y < j + windowSize && y >= j)
                                {
                                    return 0;
                                }

                                return result[x, y];
                            });
                    }
                    else
                    {
                        result = result.Select2D((value, x, y) =>
                        {
                            if (x < i + windowSize && x >= i &&
                                y < j + windowSize && y >= j)
                            {
                                return img[x,y];
                            }

                            return result[x, y];
                        });
                    }
                }
            }

            return result;
        }
    }
}