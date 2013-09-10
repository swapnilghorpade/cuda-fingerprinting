using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.ImageEnhancement.EightGabor
{
        public static class Normalizer
    {
        public static void Normalize(int expectedMean, int expectedVar, int[,] image)
        {
            double mean = Mean(image);
            double variance = Variance(image);
            for (int i = 0; i <= image.GetUpperBound(0); ++i)
            {
                for (int j = 0; j <= image.GetUpperBound(1); ++j)
                {
                    var tmp = Convert.ToInt32(Math.Sqrt(expectedVar * Math.Pow(image[i, j] - mean, 2) / variance));
                    if (image[i, j] > mean)
                        image[i, j] = expectedMean + tmp > 255 ? 255 : expectedMean + tmp; 
                    else
                        image[i, j] = expectedMean - tmp < 0 ? 0 : expectedMean - tmp; 

                }
            }
        }

        public static void LinearNormalize(int newMin, int newMax, int[,] img)
        {
            int min = 255;
            int max = 0;
            foreach (var i in img)
            {
                if (i > max)
                    max = i;
                if (i < min)
                    min = i;
            }

            if (max == min)
                return;

            for (int i = 0; i < img.GetLength(0); i++)
            {
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    var tmp = (img[i, j] - min) * (newMax - newMin) / (max - min) + newMin;
                    img[i, j] = tmp > 255 ? 255 : (tmp < 0 ? 0 : tmp);
                }
            }
        }

        public static double Mean(int[,] image)
        {
            double summ = 0;
            for (int i = 0; i <= image.GetUpperBound(0); ++i)
            {
                for (int j = 0; j < image.GetUpperBound(1); ++j)
                {
                    summ += image[i, j];
                }
            }
            return summ / (image.GetLength(0) * image.GetLength(1));
        }

        public static double Variance(int[,] image)
        {
            double summ = 0;
            double mean = Mean(image);
            for (int i = 0; i <= image.GetUpperBound(0); ++i)
            {
                for (int j = 0; j <= image.GetUpperBound(1); ++j)
                {
                    summ += Math.Pow(image[i, j] - mean, 2);
                }
            }
            return summ / (image.GetLength(0) * image.GetLength(1));
        }
    }
}

