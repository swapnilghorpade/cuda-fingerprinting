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
        private static Dictionary<double, double> integralValues;
        private static List<Minutia> neighbourMinutiae = new List<Minutia>();
        private static Tuple<int, int> currentСoordinate;
        private static double deltaS;
        private static double deltaD;
        private static bool[,] workingArea;

        public static Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> MCCMethod(List<Minutia> minutiae, int rows, int columns)
        {
            integralValues = new Dictionary<double, double>();
            List<Minutia> allNeighbours;
            deltaS = 2 * Constants.R / Constants.Ns;
            deltaD = 2 * Math.PI / Constants.Nd;
            MakeTableOfIntegrals();
            workingArea = WorkingArea.BuildWorkingArea(minutiae, Constants.R, rows, columns);

            for (int index = 0; index < minutiae.Count; index++)
            {
                value = new int[Constants.Ns, Constants.Ns, Constants.Nd];
                mask = new int[Constants.Ns, Constants.Ns, Constants.Nd];
                allNeighbours = new List<Minutia>();

                for (int i = 0; i < Constants.Ns; i++)
                {
                    for (int j = 0; j < Constants.Ns; j++)
                    {
                        int maskValue = CalculateMaskValue(minutiae[index], i, j, rows, columns);

                        for (int k = 0; k < Constants.Nd; k++)
                        {
                            currentСoordinate = GetCoordinatesInFingerprint(minutiae[index], i, j);
                            if (currentСoordinate.Item1 < 0 ||
                                currentСoordinate.Item1 >= rows ||
                                currentСoordinate.Item2 < 0 ||
                                currentСoordinate.Item2 >= columns)
                            {
                                value[i, j, k] = 0;
                                mask[i, j, k] = 0;
                                continue;
                            }
                            mask[i, j, k] = maskValue;

                            neighbourMinutiae = GetNeighbourMinutiae(minutiae, minutiae[index], currentСoordinate);

                            allNeighbours.AddRange(neighbourMinutiae);
                            allNeighbours = MinutiaListDistinct(allNeighbours);

                            value[i, j, k] = Psi(GetValue(minutiae[index], k));
                        }
                    }
                }

                if (allNeighbours.Count < Constants.MinM && !IsValidMask())
                {
                    continue;
                }

                response.Add(minutiae[index], new Tuple<int[, ,], int[, ,]>(value, mask));
            }

            return response;
        }

        private static int Psi(double v)
        {
            return (v >= Constants.MuPsi) ? 1 : 0;
        }

        private static double GetValue(Minutia currentMinutia, int k)
        {
            double result = 0;

            for (int counter = 0; counter < neighbourMinutiae.Count; counter++)
            {
                double spatialContribution = GetSpatialContribution(neighbourMinutiae[counter], currentСoordinate);
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

        private static double NormalizeAngle(double angle)
        {
            if (angle < -Math.PI)
            {
                return 2 * Math.PI + angle;
            }

            if (angle < Math.PI)
            {
                return angle;
            }

            return -2 * Math.PI + angle;
        }

        private static double GetAngleFromLevel(int k)
        {
            return Math.PI + (k - 1 / 2) * deltaD;
        }

        private static double GetDistance(int x1, int y1, int x2, int y2)
        {
            return Math.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        }

        private static double GetDirectionalContribution(double mAngle, double mtAngle, int k)
        {
            double angleFromLevel = NormalizeAngle(GetAngleFromLevel(k));
            double differenceAngles = NormalizeAngle(mAngle - mtAngle);
            double param = NormalizeAngle(angleFromLevel - differenceAngles);

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
                if (Math.Abs(key - param) <= Math.PI / Constants.DictionaryCount)
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
                result += 2 * Integrand(a + i * h);
            }

            for (int i = 0; i < Constants.N; i++)
            {
                result += 4 * Integrand(0.5 * h + a + i * h);
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
            double key = -Math.PI;
            double step = 2 * Math.PI / Constants.DictionaryCount;

            for (int i = 0; i <= Constants.DictionaryCount; i++)
            {
                integralValues.Add(key, GetIntegral(key));
                key += step;
            }
        }

        private static int CalculateMaskValue(Minutia m, int i, int j, int rows, int columns)
        {
            Tuple<int, int> point = GetCoordinatesInFingerprint(m, i, j);

            if (point.Item1 < 0 || point.Item1 >= rows ||
                point.Item2 < 0 || point.Item2 >= columns)
            {
                return 0;
            }

            return ((GetDistance(m.X, m.Y, point.Item1, point.Item2) <= Constants.R) && workingArea[point.Item1, point.Item2])
                    ? 1
                    : 0;
        }

        private static List<Minutia> MinutiaListDistinct(List<Minutia> list)
        {
            List<Minutia> listForeach = new List<Minutia>(list);

            list.Clear();

            foreach (Minutia minutia in listForeach)
            {
                if (!list.Contains(minutia))
                {
                    list.Add(minutia);
                }
            }

            return list;
        }

        private static bool IsValidMask()
        {
            int result = 0;

            foreach (int maskValue in mask)
            {
                if (maskValue == 1)
                {
                    result++;
                }
            }

            return result >= Constants.MinVC;
        }
    }
}
