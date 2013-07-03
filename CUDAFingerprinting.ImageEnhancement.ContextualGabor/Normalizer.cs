using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor
{
    public static class Normalizer
    {
        public static void Normalize(int expectedMean, int expectedVar, double[,] image)
        {
            double mean = Mean(image);
            double variance = Variance(image);
            for (int i = 0; i < image.GetUpperBound(0); ++i)
            {
                for (int j = 0; j < image.GetUpperBound(1); ++j)
                {
                    if (image[i, j] > mean)
                        image[i, j] = expectedMean + Math.Sqrt(expectedMean * Math.Pow(image[i, j] - mean, 2) / variance);
                    else
                        image[i, j] = expectedMean - Math.Sqrt(expectedMean * Math.Pow(image[i, j] - mean, 2) / variance);
                }
            }
        }

        public static double Mean(double[,] image)
        {
            double summ = 0;
            for (int i = 0; i < image.GetUpperBound(0); ++i)
            {
                for (int j = 0; j < image.GetUpperBound(1); ++j)
                {
                    summ += image[i, j];
                }
            }
            return summ / (image.GetUpperBound(0) * image.GetUpperBound(1));
        }

        public static double Variance(double[,] image)
        {
            double summ = 0;
            double mean = Mean(image);
            for (int i = 0; i < image.GetUpperBound(0); ++i)
            {
                for (int j = 0; j < image.GetUpperBound(1); ++j)
                {
                    summ += Math.Pow(image[i, j] - mean, 2);
                }
            }
            return summ / (image.GetUpperBound(0) * image.GetUpperBound(1));
        }
    }
}
