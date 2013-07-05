using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor
{
    public static class OrientationFieldGenerator
    {
        // Half of the frame's size
        private const int W = 8;
        // Sigma of Gaussian's blur
        private const double sigma = 0.6;

        public static double[,] GenerateLeastSquareEstimate(int[,] image)
        {
            int maxY = image.GetLength(0);
            int maxX = image.GetLength(1);
            var result = new double[maxY, maxX];
            var gradX = GradientHelper.GenerateXGradient(image);
            var gradY = GradientHelper.GenerateYGradient(image);

            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    double vx = 0;
                    double vy = 0;

                    for (int dy = -W; dy <= W; dy++)
                    {
                        for (int dx = -W; dx <= W; dx++)
                        {
                            int y = i + dy;
                            int x = j + dx;

                            if (!(x < 0 || y < 0 || x >= maxX || y >= maxY))
                            {
                                vx += 2 * gradX[y, x] * gradY[y, x];
                                vy += Math.Pow(gradX[y, x], 2) - Math.Pow(gradY[y, x], 2);
                            }
                        }
                    }
                    result[i, j] = 0.5 * Math.Atan2(vx, vy);
                }
            }
            return result;
        }

        private static int[,] getGaussianKernel(double sigma)
        {
            int size = 1 + 2 * (int)Math.Ceiling(3 * sigma);
            var result = new int[size, size];


            return result;
        }
    }
}
