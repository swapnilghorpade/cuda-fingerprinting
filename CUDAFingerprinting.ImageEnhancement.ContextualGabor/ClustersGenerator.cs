using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor
{
    public static class ClustersGenerator
    {
        const int clustersAmount = 6;
        readonly static public string ClusterPath = Path.GetTempPath() + "clusters.txt";

        public class ClusterPoint
        {
            // Modificators for distance calculation
            public const double DA = 55;
            public const double DF = 19;
            public const double DV = 1;

            public int ClusterNumber { get; set; }
            public double Amplitude { get; set; }
            public double Frequency { get; set; }
            public double Variance { get; set; }

            public ClusterPoint(double a, double f, double v)
            {
                Amplitude = a;
                Frequency = f;
                Variance = v;
                ClusterNumber = 0;
            }

            public double Distance(ClusterPoint p)
            {
                double da = Amplitude - p.Amplitude;
                double df = Frequency - p.Frequency;
                double dv = Variance - p.Variance;
                return Math.Sqrt(DA * da * da + DF * df * df + DV * dv * dv);
            }

            public int FindClosestIndex(ClusterPoint[] centers)
            {
                int index = 0;
                double min = Distance(centers[0]);
                for (int i = 1; i < clustersAmount; i++)
                {
                    if (Distance(centers[i]) < min)
                    {
                        min = Distance(centers[i]);
                        index = i;
                    }
                }
                return index;
            }

        }

        private static void Iterate(List<ClusterPoint> points, ClusterPoint[] centers)
        {
            foreach (var p in points)
            {
                p.ClusterNumber = p.FindClosestIndex(centers);
            }
        }


        private static bool IsClustered(ClusterPoint[] oldCenters, ClusterPoint[] newCenters)
        {
            for (int i = 0; i < clustersAmount; i++)
            {
                if (oldCenters[i].Amplitude != newCenters[i].Amplitude || oldCenters[i].Frequency != newCenters[i].Frequency
                    || oldCenters[i].Variance != newCenters[i].Variance)
                    return false;
            }
            return true;
        }

        private static ClusterPoint[] FindCenters(List<ClusterPoint> points)
        {
            ClusterPoint[] result = new ClusterPoint[clustersAmount];
            double[] pointsAmount = new double[clustersAmount];
            for (int i = 0; i < clustersAmount; i++)
            {
                pointsAmount[i] = 0;
                result[i] = new ClusterPoint(0, 0, 0);
            }

            foreach (var p in points)
            {
                result[p.ClusterNumber].Amplitude += p.Amplitude;
                result[p.ClusterNumber].Frequency += p.Frequency;
                result[p.ClusterNumber].Variance += p.Variance;
                ++pointsAmount[p.ClusterNumber];
            }

            for (int i = 0; i < clustersAmount; i++)
            {
                result[i].Amplitude /= pointsAmount[i];
                result[i].Frequency /= pointsAmount[i];
                result[i].Variance /= pointsAmount[i];
            }
            return result;
        }

        public static ClusterPoint[] Clusterization(List<ClusterPoint> points, ClusterPoint[] centers)
        {
            ClusterPoint[] newCenters = new ClusterPoint[clustersAmount];
            do
            {
                Array.Copy(centers, newCenters, clustersAmount);
                Iterate(points, centers);
                centers = FindCenters(points);
            } while (!IsClustered(centers, newCenters));
            
            return centers;
        }
    }
 }
