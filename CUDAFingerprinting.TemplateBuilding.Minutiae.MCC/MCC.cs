using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.ConvexHull;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC
{
    public static class MCC
    {
        private static Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> response = new Dictionary<Minutia, Tuple<int[, ,], int[, ,]>>();
        private static int[, ,] value;
        private static int[, ,] mask;
        private static Dictionary<double, double> integralValues = new Dictionary<double, double>();
        private static double deltaS;
        private static double deltaD;
        private static bool[,] workingArea;

        public static Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> MCCMethod(List<Minutia> minutiae, int rows, int columns)
        {
            deltaS = 2 * Constants.R / Constants.Ns;
            deltaD = 2 * Math.PI / Constants.Nd;
            MakeTableOfIntegrals();
            workingArea = WorkingArea.BuildWorkingArea(minutiae, Constants.R, rows, columns);

            for (int index = 0; index < minutiae.Count; index++)
            {
                value = new int[Constants.Ns, Constants.Ns, Constants.Nd];
                mask = new int[Constants.Ns, Constants.Ns, Constants.Nd];

                for (int i = 0; i < Constants.Ns; i++)
                {
                    for (int j = 0; j < Constants.Ns; j++)
                    {
                        int maskValue = CalculateMaskValue(minutiae[index], i, j, rows, columns);

                        for (int k = 0; k < Constants.Nd; k++)
                        {
                            mask[i, j, k] = maskValue;
                            value[i, j, k] = Psi(GetValue(minutiae, minutiae[index], i, j, k));
                        }
                    }
                }

                response.Add(minutiae[index], new Tuple<int[, ,], int[, ,]>(value, mask));
            }

            return response;
        }

        private static int Psi(double v)
        {
            return (v >= Constants.MuPsi) ? 1 : 0;
        }

        private static double GetValue(List<Minutia> allMinutiae, Minutia currentMinutia, int i, int j, int k)
        {
            double result = 0;
            Tuple<int, int> currentCoorpinate = GetCoordinatesInFingerprint(currentMinutia, i, j);
            List<Minutia> neighbourMinutiae = GetNeighbourMinutiae(allMinutiae, currentMinutia, currentCoorpinate);

            for (int counter = 0; counter < neighbourMinutiae.Count; counter++)
            {
                double spatialContribution = GetSpatialContribution(neighbourMinutiae[counter], currentCoorpinate);
                double directionalContribution = GetDirectionalContribution(currentMinutia.Angle, neighbourMinutiae[counter].Angle, k);
                result += spatialContribution * directionalContribution;
            }

            return result;
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
            return Math.Sqrt((point1.Item1 - point2.Item1) * (point1.Item1 - point2.Item1) +
                             (point1.Item2 - point2.Item2) * (point1.Item2 - point2.Item2));
        }

        private static double GetDistance(int x1, int y1, int x2, int y2)
        {
            return Math.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        }

        private static double GetDirectionalContribution(double mAngle, double mtAngle, int k)
        {
            double angleFromLevel = GetAngleFromLevel(k);
            double differenceAngles = GetDifferenceAngles(mAngle, mtAngle);
            double param = GetDifferenceAngles(angleFromLevel, differenceAngles);

            try
            {
                return integralValues[GetIntegralParameter(param)];
            }
            catch (ArgumentNullException e)
            {
                throw e;
            }
        }

        private static double GetIntegralParameter(double param)
        {
            foreach (double key in integralValues.Keys)
            {
                if (Math.Abs(key - param) < Math.PI/Constants.DictionaryCount)
                {
                    return key;
                }
            }

            throw new ArgumentNullException("param");
        }

        private static double GetIntegral_1(double parameter)
        {
            double factor = 1 / (Constants.SigmaD * Math.Sqrt(2 * Math.PI));
            double a = parameter - deltaD / 2;
            double h = deltaD / Constants.N;
            double result = Integrand(a) + Integrand(a + (Constants.N - 1) * h);

            for (int i = 1; i < Constants.N; i++)
            {
                result += 2 * Integrand(a + i*h);
            }

            for (int i = 0; i < Constants.N; i++)
            {
                result += 4 * Integrand(0.5 * h + a + i*h);
            }

            return factor * h * result / 6.0;

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

        private static List<Minutia> GetNeighbourMinutiae(List<Minutia> minutiae, Minutia minutia, Tuple<int, int> currentCell)
        {
            List<Minutia> result = new List<Minutia>();

            for (int i = 0; i < minutiae.Count; i++)
            {
                if (minutiae[i] != minutia &&
                    GetDistance(minutiae[i].X, minutiae[i].Y, currentCell.Item1, currentCell.Item2) <= 3 * Constants.SigmaS)
                {
                    result.Add(minutiae[i]);
                } 
            }

            // result < Constants.MinM => invalid

            return result;
        }

        private static double GetSpatialContribution(Minutia m, Tuple<int, int> currentCell)
        {
            double distance = GetDistance(m.X, m.Y, currentCell.Item1, currentCell.Item2);

            return Gaussian.Gaussian1D(distance, Constants.SigmaS);
        }

        private static double Integrand(double parameter)
        {
            double result = (-parameter * parameter) / (2 * Constants.SigmaD * Constants.SigmaD);

            return Math.Exp(result);
        }

        private static void MakeTableOfIntegrals()
        {
            double value = -Math.PI;
            double step = 2 * Math.PI / Constants.DictionaryCount;

            for (int i = 0; i < Constants.DictionaryCount; i++)
            {
                integralValues.Add(value, GetIntegral(i));
                value += step;
            }
        }

        private static int CalculateMaskValue(Minutia m, int i, int j, int rows, int columns)
        {
            Tuple<int, int> point = GetCoordinatesInFingerprint(m, i, j);

            if (point.Item1 < 0 || point.Item1 >= columns ||
                point.Item2 < 0 || point.Item2 >= rows)
            {
                return 0;
            }

            // all points < Constants.MinVC => invalid

            return ((GetDistance(m.X, m.Y, point.Item1, point.Item2) <= Constants.R) && workingArea[point.Item1, point.Item2]) 
                    ? 1 
                    : 0;
        }
    }
}
