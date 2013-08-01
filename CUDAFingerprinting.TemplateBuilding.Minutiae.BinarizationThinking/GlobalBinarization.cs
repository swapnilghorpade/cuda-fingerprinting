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
            return img.Select2D(x => x > border ? 255.0 : 0);
        }
    }
}
