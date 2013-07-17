using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC
{
    public class MCC
    {
        private int[,,] value;
        private double deltaS;
        private double deltaD;

        public void MCCMethod(Minutia[] minutiae)
        {
            for (int index = 0; index < minutiae.GetLength(0); index++)
            {
                value = new int[Constants.Ns, Constants.Ns, Constants.Nd];
                deltaS = 2 * Constants.R / Constants.Ns;
                deltaD = 2 * Math.PI / Constants.Nd;

                for (int i = 0; i < Constants.Ns; i++)
                {
                    for (int j = 0; j < Constants.Ns; j++)
                    {
                        for (int k = 0; k < Constants.Nd; k++)
                        {
                            //value[i, j, k] = ... ;
                        }
                    }
                }
            }
        }

        private bool Psi(double value)
        {
            return value >= Constants.MuPsi;
        }

        private Tuple<int, int> GetCoordinatesInFingerprint(Minutia m, int i, int j)
        {
            double halfNs = (1 + Constants.Ns)/2;
            double sinTetta = Math.Sin(m.Angle);
            double cosTetta = Math.Cos(m.Angle);
            double iDelta = cosTetta*(i - halfNs) + sinTetta * (j - halfNs);
            double jDelta = -sinTetta*(i - halfNs) + cosTetta*(j - halfNs);

            return new Tuple<int, int>((int)(m.X + deltaS * iDelta), (int)(m.Y + deltaS * jDelta));
        }

        private double GetDifferenceAngles(double tetta1, double tetta2)
        {
            double difference = tetta1 - tetta2;

            if (difference < - Math.PI)
            {
                return 2*Math.PI + difference;
            }
            
            if (difference < Math.PI)
            {
                return difference;
            }
            
            return -2*Math.PI + difference;
        }

        private double GetDirectionalContribution(Minutia m, Tuple<int, int> coordinatesInFingerprint, int k)
        {
           // double angleFromLevel = GetAngleFromLevel(k);
           // double param = GetDifferenceAngles()
            return 0;
        }

        private double GetAngleFromLevel(int k)
        {
            return Math.PI + (k - 1 / 2) * deltaD;
        }
    }
}
