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
        private const double sigma = 5;

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

        private static double[,] GenerateGaussianKernel(double sigma)
        {
            int size = 1 + 2 * (int)Math.Ceiling(3 * sigma);
            var result = new double[size, size];
            int center = size / 2;
            Func<int, int, double> gaussian = (x, y) => 1 / (2 * Math.PI * sigma * sigma) * Math.Exp(-(x * x + y * y) / (2 * sigma * sigma));

            for (int i = -center; i <= center; i++)
            {
                for (int j = -center; j <= center; j++)
                {
                    result[i + center, j + center] = gaussian(i, j);
                }
            }
            return result;
        }

        public static double[,] GenerateBlur(double[,] smth)
        {
            int maxY = smth.GetLength(0);
            int maxX = smth.GetLength(1);
            var result = new double[maxY, maxX];
            var gKernel = GenerateGaussianKernel(sigma);
            int size = gKernel.GetLength(0) / 2;

            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    for (int dy = -size; dy <= size; dy++)
                    {
                        for (int dx = -size; dx <= size; dx++)
                        {
                            int x = j + dx;
                            int y = i + dy;
                            double value = 0;
                            if (x < 0 || x >= maxX || y < 0 || y >= maxY)
                            {
                                if ((x < 0 && y < 0) || (x >= maxX && y >= maxY) || (x >= maxX && y < 0) || (x < 0 && y >= maxY))
                                    value = smth[y - 2 * dy, x - 2 * dx];
                                else if (x < 0 || x >= maxX)
                                    value = smth[y, x - 2 * dx];
                                else
                                    value = smth[y - 2 * dy, x];
                            }
                            else
                                value = smth[y, x];
                            result[i, j] += value * gKernel[dy + size, dx + size];
                        }
                    }
                }
            }
            return result;
        }
    }
}
