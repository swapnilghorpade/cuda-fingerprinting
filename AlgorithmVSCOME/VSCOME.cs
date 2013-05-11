using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;

namespace AlgorithmVSCOME
{
    internal class VSCOME
    {
        private double[,] orientationField;

        public VSCOME(double[,] orientationField)
        {
            this.orientationField = orientationField;
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
                    symmetryValue = Symmetry.CalculateSymmetry(u, v, orientationField.GetLength(0) * orientationField.GetLength(1));
                    vorivValue = VORIV.CalculateVoriv(u, v);
                    vscome[u, v] = (vorivValue + symmetryValue) / 2;
                }
            }

            return vscome;
        }
    }
}
