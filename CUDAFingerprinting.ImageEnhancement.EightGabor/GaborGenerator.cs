using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.ImageEnhancement.EightGabor
{
    public class GaborGenerator
    {  
        private const int GaborFilterSize = 11;
        private const double sigma = 2;
        private const double frequency = 0.15;

        public static double[,] GenerateGaborFilter(int[,] image, double filterOrientation)
        {
            int maxY = image.GetLength(0);
            int maxX = image.GetLength(1);
            var lro = ContextualGabor.OrientationFieldGenerator.GenerateLocalRidgeOrientation(image);
            var result = new double[maxY, maxX];

            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {   
                    result[i, j] = 0;
                    filterOrientation = lro[i, j];
                    int w = GaborFilterSize / 2;

                    for (int dy = -w; dy <= w; dy++)
                    {
                        for (int dx = -w; dx <= w; dx++)
                        {
                            int value = 0;
                            int y = i + dy;
                            int x = j + dx;

                            if (x >= 0 && x < maxX && y >= 0 && y < maxY)
                            {
                                value = image[y, x];
                                double rotatedX = dx * Math.Cos(filterOrientation) + dy * Math.Sin(filterOrientation);
                                double rotatedY = - dx * Math.Sin(filterOrientation) + dy * Math.Cos(filterOrientation);
                                double power = - 0.5 * ((rotatedX * rotatedX + rotatedY * rotatedY) / (sigma * sigma));
                                double cosArg = 2 * Math.PI * rotatedX * frequency;
                                double gabor = Math.Exp(power) * Math.Cos(cosArg);
                                result[i, j] += gabor * value;
                            }
                        }
                    }
                }
            }       
            return result;
        }
    }
}
