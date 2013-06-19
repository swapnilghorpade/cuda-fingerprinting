using System;

namespace CUDAFingerprinting.ReferencePoint.VSCOME
{
    internal static class XY_WithLine
    {
        public static double GetX_WithLine(double x, double y)
        {
            if (y < 0 || (y == 0 && x == 0))
            {
                return 0;
            }

            double result = 2 * x * y;
            double denominator = GetDenominator(x, y);

            return result / denominator;
        }

        public static double GetY_WithLine(double x, double y)
        {
            if (y < 0 || (y == 0 && x == 0))
            {
                return 0;
            }

            double result = x * x - y * y;
            double denominator = GetDenominator(x, y);

            return result / denominator;
        }

        private static double GetDenominator(double x, double y)
        {
            return Math.Sqrt(x * x + y * y);
        }
    }
}
