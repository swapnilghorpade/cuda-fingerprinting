using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.ImageEnhancement.EightGabor
{
    public class GaborGenerator
    {
        // Average distance between ridges
        public const int AverageDistance = 5;
        public const int GaborFilterSize = 11;
        private const double sigma = 4.0;
        private const double frequency = 5; // 60?


        public static double[,] GenerateGaborFilter(int[,] image, double filterOrientation)
        {
            int maxY = image.GetLength(0);
            int maxX = image.GetLength(1);
            var result = new double[maxY, maxX];

            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    result[i, j] = 0;

                    for (int dy = -GaborFilterSize / 2; dy <= GaborFilterSize / 2; dy++)
                    {
                        for (int dx = -GaborFilterSize / 2; dx <= GaborFilterSize / 2; dx++)
                        {
                            int value = 0;
                            int x = j + dx;
                            int y = i + dy;
                            if (x < 0 || x >= maxX || y < 0 || y >= maxY)
                                value = 0;
                            else
                                value = image[y, x];

                            double RotatedX = dx * Math.Cos(filterOrientation) + dy * Math.Sin(filterOrientation);
                            double RotatedY = - dx * Math.Sin(filterOrientation) + dy * Math.Cos(filterOrientation);
                            double power = - 0.5 * ((RotatedX * RotatedX) / (sigma * sigma) + (RotatedY * RotatedY) / (sigma * sigma));
                            double cosArg = 2 * Math.PI * RotatedX * frequency;
                            result[i, j] += Math.Exp(power) * Math.Cos(cosArg) * value;
                        }
                    }
                }
            }       
            return result;
        }
    }
}
