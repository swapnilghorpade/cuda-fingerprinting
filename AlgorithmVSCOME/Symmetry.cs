using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmVSCOME
{
    internal static class Symmetry
    {
        internal static double CalculateSymmetry(double u, double v, int m)
        {
            XY_Function function = new XY_Function(u, v);

            function.CalculateFunction();

            double result = function.X_Function * function.X_Function +
                            function.Y_Function * function.Y_Function;

            return Math.Sqrt(result) / m;
        }
    }
}
