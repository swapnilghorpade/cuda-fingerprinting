using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace AlgorithmVSCOME
{
    internal static class Symmetry
    {
        internal static double CalculateSymmetry(int u, int v, int m, double[,] filteredField)
        {
            if (XY_Function.CalculateFunction(u, v, filteredField))
            {
                double result = XY_Function.X_Function * XY_Function.X_Function +
                                XY_Function.Y_Function * XY_Function.Y_Function;

                return Math.Sqrt(result) / m;
            }

            throw new ArgumentException("CalculateSymmetry");
        }
    }
}
