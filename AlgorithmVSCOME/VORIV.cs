using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmVSCOME
{
    internal static class VORIV
    {
        internal static double CalculateVoriv(int u, int v, double[,] orientationField)
        {
            int halfK = (int)(Constants.K / 2);
            int maxValueX = orientationField.GetLength(0);
            int maxValueY = orientationField.GetLength(1);
            double sum = 0;
            double result = 0;

            for (int k = -halfK; k <= halfK; k++)
            {
                for (int i = Constants.N - 1; i >= 0; i--)
                {
                    if (v + k < 0 || u - i < 0 || u - i + 1 >= maxValueX || v + k >= maxValueY)
                    {
                        continue;
                    }

                    sum += FunctionF(orientationField[u - i, v + k] - orientationField[u - i + 1, v + k]);
                }

                result += sum;
            }

            return result / Math.PI;
        }

        private static double FunctionF(double arg)
        {
            if (arg >= 0 && arg <= Math.PI)
            {
                return arg;
            }

            if (arg < 0 && arg > Math.PI* (-1))
            {
                return arg * (-1);
            }

            return 0;
        }
    }
}
