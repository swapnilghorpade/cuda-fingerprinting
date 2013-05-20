using System.Numerics;

namespace AlgorithmVSCOME
{
    internal class VSCOME
    {
        private double[,] orientationField;
        private double[,] filteredField;

        public VSCOME(double[,] orientationField, double[,] filteredField)
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
                    symmetryValue = Symmetry.CalculateSymmetry(u, v, orientationField.GetLength(0) * orientationField.GetLength(1), filteredField);
                    vorivValue = VORIV.CalculateVoriv(u, v, orientationField);
                    vscome[u, v] = (vorivValue + symmetryValue) / 2;
                }
            }

            return vscome;
        }
    }
}
