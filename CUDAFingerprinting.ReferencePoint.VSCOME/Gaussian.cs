using System;

namespace CUDAFingerprinting.ReferencePoint.VSCOME
{
    internal class Gaussian
    {
        internal static double CalculateGaussian(double x, double y)
        {
            double result = -1 * (x*x + y*y);
            double denominator = 2 * Constants.Sigma * Constants.Sigma;

            result = Math.Exp(result / denominator);

            return result;
        }
    }
}
