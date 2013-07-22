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

        public class ClusterPoint
        {
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
                return Math.Sqrt(da * da + df * df + dv * dv);
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

        private void Iterate(List<ClusterPoint> points, ClusterPoint[] centers)
        {
            foreach (var p in points)
            {
                p.ClusterNumber = p.FindClosestIndex(centers);
            }
        }


        private bool IsClustered(ClusterPoint[] oldCenters, ClusterPoint[] newCenters)
        {
            if (newCenters == null)
                return false;

            for (int i = 0; i < clustersAmount; i++)
            {
                if (oldCenters[i].Amplitude != newCenters[i].Amplitude || oldCenters[i].Frequency != newCenters[i].Frequency
                    || oldCenters[i].Variance != newCenters[i].Variance)
                    return false;
            }
            return true;
        }

        private ClusterPoint[] FindCenters(List<ClusterPoint> points)
        {
            ClusterPoint[] result = new ClusterPoint[clustersAmount];
            double[] pointsAmount = new double[clustersAmount];
            for (int i = 0; i < clustersAmount; i++)
            {
                pointsAmount[i] = 0;
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

        public void Clusterization(List<ClusterPoint> points, ClusterPoint[] centers)
        {
            ClusterPoint[] newCenters = new ClusterPoint[clustersAmount];
            Array.Copy(centers, newCenters, clustersAmount);
            Iterate(points, centers);
            centers = FindCenters(points);

            while (!IsClustered(centers, newCenters))
            {
                Array.Copy(centers, newCenters, clustersAmount);
                Iterate(points, centers);
                centers = FindCenters(points);
            }

            // Writing result to file
            StreamWriter file = new System.IO.StreamWriter(Path.GetTempPath() + "clusters.txt");
            for (int i = 0; i < centers.Length; i++)
            {
                file.WriteLine(i);
                file.WriteLine(centers[i].Amplitude);
                file.WriteLine(centers[i].Frequency);
                file.WriteLine(centers[i].Variance);
            }
        }
    }
 }
