using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmVSCOME
{
    internal static class VORIV
    {
        internal static double CalculateVoriv(double u, double v)
        {
            int halfK = (int)(Constants.K / 2);
            double sum = 0;
            double result = 0;

            for (int k = -halfK; k <= halfK; k++)
            {
                for (int i = 0; i <= Constants.N - 1; i++)
                {
                    sum += FunctionF(Tetta.GetTetta(u - i, v + k)
                        - Tetta.GetTetta(u - i - 1, v + k));
                }

                result += sum;
            }

            return result / Math.PI;
        }

        private static double FunctionF(double arg)
        {
            if (Math.Abs(arg) <= Math.PI / 2)
            {
                return arg;
            }

            if (arg > Math.PI / 2)
            {
                return Math.PI - arg;
            }

            if (arg < Math.PI / -2)
            {
                return Math.PI + arg;
            }

            throw new ArgumentException("FunctionF");
        }
    }
}
