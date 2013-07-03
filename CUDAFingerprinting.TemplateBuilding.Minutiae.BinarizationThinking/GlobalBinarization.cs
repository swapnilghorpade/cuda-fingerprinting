using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;
//using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking
{
    public static class GlobalBinarization
    {
        public static double[,] Binarization(double [,] img, double border)
        {
            for (int i = 0; i < img.GetLength(0); i++)
            {
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    img[i, j] = img[i, j] > border ? 255 : 0;
                }
            }
            return img;
        }
    }
}
