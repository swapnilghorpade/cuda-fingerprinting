using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace AlgorithmVSCOME
{
    internal static class Filter
    {
        public static Complex[,] GetFilter(double[,] arr)
        {
            int xLength = arr.GetLength(0);
            int yLength = arr.GetLength(1);
            int upperBoundX = (int)(xLength / 2);
            int lowerBoundX = -1 * upperBoundX;
            int upperBoundY = (int)(yLength / 2);
            int lowerBoundY = -1 * upperBoundY;

            Complex[,] filter = new Complex[xLength, yLength];
            double x_, y_ = 0;

            for (int x = lowerBoundX; x <= upperBoundX; x++)
            {
                for (int y = lowerBoundY; y <= upperBoundY; y++)
                {
                    if (y < 0)
                    {
                        filter[upperBoundX + x, upperBoundY + y] = 0;
                        continue;
                    }

                    x_ = XY_WithLine.GetX_WithLine(upperBoundX + x, upperBoundY + y);
                    y_ = XY_WithLine.GetY_WithLine(upperBoundX + x, upperBoundY + y);
                    filter[upperBoundX + x, upperBoundY + y] = 
                        (2 * x_ * y_ + Complex.ImaginaryOne * (x_ * x_ - y_ * y_)) * Gaussian.CalculateGaussian(x_, y_);
                }
            }

            return filter;
        }
    }
}
