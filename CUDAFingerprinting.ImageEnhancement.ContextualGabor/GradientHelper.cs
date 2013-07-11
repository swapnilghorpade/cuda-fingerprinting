using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor
{
    public static class GradientHelper
    {
        public static int[,] GenerateXGradient(int[,] image)
        {
            int[,] sobelFilter = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            int maxY = image.GetLength(0);
            int maxX = image.GetLength(1);
            int[,] gradient = new int[maxY, maxX];
            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {

                    for (int dy = -1; dy < 2; dy++)
                    {
                        for (int dx = -1; dx < 2; dx++)
                        {
                            int x = j + dx;
                            int y = i + dy;
                            int color = 0;
                            if (x < 0 || x >= maxX || y < 0 || y >= maxY)
                                color = image[i, j];
                            else 
                                color = image[y, x];
                            gradient[i, j] += color * sobelFilter[dy + 1, dx + 1];
                        }
                    }

                }
            }
            return gradient;
        }

        public static int[,] GenerateYGradient(int[,] image)
        {
            int[,] sobelFilter = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
            int maxY = image.GetLength(0);
            int maxX = image.GetLength(1);
            int[,] gradient = new int[maxY, maxX];
            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {

                    for (int dy = -1; dy < 2; dy++)
                    {
                        for (int dx = -1; dx < 2; dx++)
                        {
                            int x = j + dx;
                            int y = i + dy;
                            int color = 0;
                            if (x < 0 || x >= maxX || y < 0 || y >= maxY)
                                color = image[i, j];
                            else
                                color = image[y, x];
                            gradient[i, j] += color * sobelFilter[dy + 1, dx + 1];
                        }
                    }

                }
            }
            return gradient;
        }

        public static int[,] GenerateGradient(int[,] image)
        {
            var gradX = GenerateXGradient(image);
            var gradY = GenerateYGradient(image);
            int[,] result = new int[image.GetLength(0), image.GetLength(1)];
            for (int i = 0; i < image.GetLength(0); i++)
            {
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    result[i, j] = Convert.ToInt32(Math.Sqrt(gradX[i, j] * gradX[i, j] + gradY[i, j] * gradY[i, j]));
                }
            }
            return result;
        }
    }
}
