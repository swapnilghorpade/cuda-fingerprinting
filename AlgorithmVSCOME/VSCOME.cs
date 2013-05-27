using System;
using System.Numerics;
using ComplexFilterQA;

namespace AlgorithmVSCOME
{
    internal static class VSCOME
    {
        public static double[,] CalculateVscomeValue(double[,] orientationField, double[,] filteredField, double[,] symmetry)
        {
            double vorivValue = 0;
            double[,] vscome = new double[orientationField.GetLength(0), orientationField.GetLength(1)];

            for (int u = 0; u < orientationField.GetLength(0); u++)
            {
                for (int v = 0; v < orientationField.GetLength(1); v++)
                {
                    vorivValue = VORIV.CalculateVoriv(u, v, orientationField);
                    vscome[u, v] = (vorivValue + symmetry[u, v]) / 2;
                }
            }

            return vscome;
        }
    }
}
