using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using ComplexFilterQA;

namespace AlgorithmVSCOME
{
    internal class VSCOME
    {
        private double[,] orientationField;
        private Complex[,] filteredField;

        public VSCOME(double[,] orientationField, Complex[,] filteredField)
        {
            this.orientationField = orientationField;
            this.filteredField = filteredField;
        }

        internal double[,] CalculateVscomeValue()
        {
            double symmetryValue = 0;
            double vorivValue = 0;
            double[,] vscome = new double[orientationField.GetLength(0), orientationField.GetLength(1)];

            for (int u = 0; u < orientationField.GetLength(0); u++)
            {
                for (int v = 0; v < orientationField.GetLength(1); v++)
                {
                    // Comblex[,] filteredField
                    // but
                    // Symmetry.CalculateSymmetry(..., double[,] filteredField) 

                    //symmetryValue = Symmetry.CalculateSymmetry(u, v, orientationField.GetLength(0) * orientationField.GetLength(1), filteredField);
                    symmetryValue = Symmetry.CalculateSymmetry(u, v, orientationField.GetLength(0) * orientationField.GetLength(1), new double[1, 1]);
                    vorivValue = VORIV.CalculateVoriv(u, v, orientationField);
                    vscome[u, v] = (vorivValue + symmetryValue) / 2;
                }
            }

            return vscome;
        }
    }
}
