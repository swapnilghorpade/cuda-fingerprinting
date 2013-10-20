using CUDAFingerprinting.Common;

//using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinning
{
    public static class GlobalBinarization
    {
        public static double[,] Binarization(double [,] img, double threshold)
        {
            return img.Select2D(x => x > threshold ? 255.0 : 0);
        }
    }
}
