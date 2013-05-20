using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmVSCOME
{
    internal static class XY_Function
    {
        private static double result_X, result_Y = 0;

        public static double X_Function
        {
            get { return result_X; }
        }

        public static double Y_Function
        {
            get { return result_Y; }
        }

        internal static bool CalculateFunction(int u, int v, double[,] orientationField)
        {
            int upperBound = (int)(Constants.W / 2);
            int lowerBound = -1 * upperBound;
            double gaussian, xWithLine, yWithLine, arg, sin, cos = 0;
            double sum_X = 0;
            double sum_Y = 0;

            for (int x = lowerBound; x < upperBound; x++)
            {
                for (int y = lowerBound; y < upperBound; y++)
                {
                    if (u + y < 0 || v + x < 0 || u + y >= orientationField.GetLength(0) || v + x >= orientationField.GetLength(1)) 
                    {
                        continue;
                    } 

                    gaussian = Gaussian.CalculateGaussian(x, y);
                    xWithLine = XY_WithLine.GetX_WithLine(x, y); 
                    yWithLine = XY_WithLine.GetY_WithLine(x, y);
                    arg = 2 * orientationField[u + y, v + x];
                    sin = Math.Sin(arg);
                    cos = Math.Cos(arg);

                    sum_X += gaussian * (xWithLine * cos - yWithLine * sin);
                    sum_Y += gaussian * (yWithLine * cos + xWithLine * sin);
                }

                result_X += sum_X;
                result_Y += sum_Y;
                sum_X = 0;
                sum_Y = 0;
            }

            return true;
        }
    }
}
