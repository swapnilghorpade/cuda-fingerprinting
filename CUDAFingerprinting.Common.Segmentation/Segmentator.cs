using System;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.Common.Segmentation
{
    public static class Segmentator
    {
        public static double[,] Segmetator(double[,] img, int windowRadius)
        {
            int[,] xGradients = OrientationFieldGenerator.GenerateXGradients(img.Select2D(a => (int)a));
            int[,] yGradients = OrientationFieldGenerator.GenerateYGradients(img.Select2D(a => (int)a));
            double[,] magnitudes = xGradients.Select2D((x, y, value) => Math.Sqrt(xGradients[x, y] * xGradients[x, y] + yGradients[x, y] * yGradients[x, y]));
            double averege = KernelHelper.Average(magnitudes);
            double[,] window = new double[windowRadius * 2 + 1, windowRadius * 2 + 1];
            double[,] result = new double[img.GetLength(0), img.GetLength(1)];

            for (int i = 0; i < magnitudes.GetLength(0); i++)
            {
                for (int j = 0; j < magnitudes.GetLength(1); j++)
                {
                    window.Select2D((x, y, value) =>
                    {
                        if (i - windowRadius + x < 0
                            || i - windowRadius + x >= magnitudes.GetLength(0)
                            || j - windowRadius + y < 0
                            || j - windowRadius + y >= magnitudes.GetLength(1))
                        {
                            return 0;
                        }

                        return magnitudes[(int)(i - windowRadius + x), j - windowRadius + y];
                    });

                    result[i, j] = KernelHelper.Average(magnitudes) < averege ? 255 : img[i, j];
                }
            }

            return result;
        }
    }
}