using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC
{
    public static class MCC
    {
        private static int[, ,] value;
        private static Dictionary<double, double> integralValues = new Dictionary<double, double>();
        private static double deltaS;
        private static double deltaD;

        public static void MCCMethod(Minutia[] minutiae)
        {
            bool[,] mask = new bool[Constants.Ns,Constants.Ns];

            deltaS = 2 * Constants.R / Constants.Ns;
            deltaD = 2 * Math.PI / Constants.Nd;
            MakeDictionary();

            for (int index = 0; index < minutiae.GetLength(0); index++)
            {
                value = new int[Constants.Ns, Constants.Ns, Constants.Nd];

                for (int i = 0; i < Constants.Ns; i++)
                {
                    for (int j = 0; j < Constants.Ns; j++)
                    {
                        mask[i, j] = CalculateMaskValue(minutiae[index].X, minutiae[index].Y, i, j);

                        for (int k = 0; k < Constants.Nd; k++)
                        {
                            //value[i, j, k] = ... ;
                        }
                    }
                }
            }
        }

        private static bool Psi(double v)
        {
            return v >= Constants.MuPsi;
        }

        private static Tuple<int, int> GetCoordinatesInFingerprint(Minutia m, int i, int j)
        {
            double halfNs = (1 + Constants.Ns) / 2;
            double sinTetta = Math.Sin(m.Angle);
            double cosTetta = Math.Cos(m.Angle);
            double iDelta = cosTetta * (i - halfNs) + sinTetta * (j - halfNs);
            double jDelta = -sinTetta * (i - halfNs) + cosTetta * (j - halfNs);

            return new Tuple<int, int>((int)(m.X + deltaS * iDelta), (int)(m.Y + deltaS * jDelta));
        }

        private static double GetDifferenceAngles(double tetta1, double tetta2)
        {
            double difference = tetta1 - tetta2;

            if (difference < -Math.PI)
            {
                return 2 * Math.PI + difference;
            }

            if (difference < Math.PI)
            {
                return difference;
            }

            return -2 * Math.PI + difference;
        }

        private static double GetAngleFromLevel(int k)
        {
            return Math.PI + (k - 1 / 2) * deltaD;
        }

        private static double GetDistance(Tuple<int, int> point1, Tuple<int, int> point2)
        {
            return Math.Sqrt((point1.Item1 - point2.Item1) * (point1.Item1 - point2.Item1) + (point1.Item2 - point2.Item2) * (point1.Item2 - point2.Item2));
        }

        private static double GetDirectionalContribution(double mAngle, double mtAngle, int k)
        {
            double angleFromLevel = GetAngleFromLevel(k);
            double differenceAngles = GetDifferenceAngles(mAngle, mtAngle);
            double param = GetDifferenceAngles(angleFromLevel, differenceAngles);
           
            return integralValues[param];
        }

        private static double GetIntegral(double parameter)
        {
            double factor = 1 / (Constants.SigmaD * Math.Sqrt(2 * Math.PI));

            double a = parameter - deltaD / 2;
            double h = deltaD / Constants.N;
            double result = 0;

            for (int i = 0; i < Constants.N; i++)
            {
                result += h * Integrand(a + ((2 * i + 1) * h) / 2);
            }

            return result * factor;
        }

        private static double Integrand(double parameter)
        {
            double result = (-parameter * parameter) / (2 * Constants.SigmaD * Constants.SigmaD);

            return Math.Exp(result);
        }

        private static void MakeDictionary()
        {
            for (int i = 0; i < Constants.DictionaryCount; i++)
            {
                integralValues.Add(i, GetIntegral(i));
            }
        }

        private static bool CalculateMaskValue(int iMinutia, int jMinutia, int i, int j)
        {
          /*  return (GetDistance(iMinutia, jMinutia, i, j) <= Constants.R) &&
                   IsBelongsToConvexHull(i, j); */

            return true;
        }
    }
}
