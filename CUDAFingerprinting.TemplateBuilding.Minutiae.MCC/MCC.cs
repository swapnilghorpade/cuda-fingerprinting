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

        public static Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> Response
        {
            get { return response;}
        }

        public static void MCCMethod(List<Minutia> minutiae, int rows, int columns)
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
                        int maskValue = CalculateMaskValue(minutiae[index].X, minutiae[index].Y, i, j);
                        
                        for (int k = 0; k < Constants.Nd; k++)
                        {
                            mask[i, j, k] = maskValue;
                            value[i, j, k] = Psi(GetValue(minutiae, minutiae[index], i, j, k));
                        }
                    }
                }

                response.Add(minutiae[index],new Tuple<int[,,], int[,,]>(value, mask));
            }
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

            for (int myLovelyCounterInThisCycle = 0; myLovelyCounterInThisCycle < neighbourMinutiae.Count; myLovelyCounterInThisCycle++)
            {
                double spatialContribution = GetSpatialContribution(neighbourMinutiae[myLovelyCounterInThisCycle], currentCoorpinate);
                double directionalContribution = GetDirectionalContribution(currentMinutia.Angle, neighbourMinutiae[myLovelyCounterInThisCycle].Angle, k);
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

            return integralValues[(int)param];
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
                    GetDistance(new Tuple<int, int>(minutiae[i].X, minutiae[i].Y), currentCell) <= 3 * Constants.SigmaS)
                {
                    result.Add(minutiae[i]);
                }
            }
            return result;
        }

        private static double GetSpatialContribution(Minutia mt, Tuple<int, int> currentCellXY)
        {
            double distance = GetDistance(new Tuple<int, int>(mt.X, mt.Y), currentCellXY);

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

        private static int CalculateMaskValue(int iMinutia, int jMinutia, int i, int j)
        {
            return ((GetDistance(iMinutia, jMinutia, i, j) <= Constants.R) && workingArea[i, j]) ? 1 : 0;
        }
    }
}
