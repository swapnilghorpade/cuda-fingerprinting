using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Numerics;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.ComplexFilters;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.ComplexFiltering
{
    public class FeatureExtractor
    {
        private static double sigma1 = 0.6;
        private static double sigma2 = 3.2;
        // private static double K0 = 1.7;
        // private static double K = 1.3;
        const int NeighborhoodSize = 9;
        // static double tau1 = 0.1;
        // static double tau2 = 0.3;
        static double tauLS = 0.9;
        static double tauPS = 0.5;
        static int ringInnerRadius = 4;
        static int ringOuterRadius = 6;
        static int MaxMinutiaeCount = 32;

        public static List<Minutia> ExtractMinutiae(double[,] imgBytes)
        {
            var lsEnhanced = SymmetryHelper.EstimateLS(imgBytes, sigma1, sigma2);
            var psEnhanced = SymmetryHelper.EstimatePS(imgBytes, sigma1, sigma2);
            //ImageHelper.SaveComplexArrayAsHSV(lsEnhanced,"C:\\temp\\lsenh.png");

            //ImageHelper.SaveArray(NormalizeArray(psEnhanced.Select2D(x=>x.Magnitude)), "C:\\temp\\psenh.png");

            var psi = KernelHelper.Zip2D(psEnhanced,
                lsEnhanced.Select2D(x => x.Magnitude), (x, y) =>
                    x * (1.0d - y));

            return SearchMinutiae(psi, lsEnhanced, psEnhanced);
        }

        private static List<Minutia> SearchMinutiae(Complex[,] psi, Complex[,] lsEnhanced, Complex[,] ps)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            var size = new Size(psi.GetLength(0), psi.GetLength(1));

            for (int x = 0; x < size.Width; x++)
            {
                for (int y = 0; y < size.Height; y++)
                {
                    if (psi[x, y].Magnitude < tauPS) psi[x, y] = 0;
                }
            }

            var grid = new int[psi.GetLength(0), psi.GetLength(1)];

            var maxs = psi.Select2D((x, row, column) =>
            {
                Point maxP = new Point();
                double maxM = tauPS;
                for (int dRow = -NeighborhoodSize / 2; dRow <= NeighborhoodSize / 2; dRow++)
                {
                    for (int dColumn = -NeighborhoodSize / 2; dColumn <= NeighborhoodSize / 2; dColumn++)
                    {
                        var correctRow = row + dRow;
                        var correctColumn = column + dColumn;

                        if (correctRow > 9 && correctColumn > 9
                            && correctColumn < psi.GetLength(1) - 10 && correctRow < psi.GetLength(0) - 10)
                        {
                            var value = psi[correctRow, correctColumn];
                            if (value.Magnitude > maxM)
                            {
                                maxM = value.Magnitude;
                                maxP = new Point(correctRow, correctColumn);
                            }
                        }
                    }
                }
                if (!maxP.IsEmpty)
                    grid[maxP.X, maxP.Y]++;
                return maxP;
            });

            Dictionary<Point, int> responses = new Dictionary<Point, int>();

            foreach (var point in maxs)
            {
                if (!responses.ContainsKey(point)) responses[point] = 0;
                responses[point]++;
            }


            ImageHelper.SaveArrayAsBinary(grid, "C:\\temp\\grid.bin");
            var orderedListOfCandidates =
                responses.Where(x => x.Value >= 20 && x.Key.X > 0 && x.Key.Y > 0 && x.Key.X < psi.GetLength(0) - 1 && x.Key.Y < psi.GetLength(1) - 1)
                         .OrderByDescending(x => x.Value)
                         .Select(x => x.Key)
                         .Where(x => psi[x.X, x.Y].Magnitude >= tauPS);
            List<Minutia> minutiae = new List<Minutia>();
            int cnt = 0;
            foreach (var candidate in orderedListOfCandidates)
            {
                int count = 0;
                double sum = 0;
                for (int dx = -ringOuterRadius; dx <= ringOuterRadius; dx++)
                {
                    for (int dy = -ringOuterRadius; dy <= ringOuterRadius; dy++)
                    {
                        if (Math.Abs(dx) < ringInnerRadius && Math.Abs(dy) < ringInnerRadius) continue;
                        count++;
                        int xx = candidate.X + dx;
                        if (xx < 0) xx = 0;
                        if (xx >= size.Width) xx = size.Width - 1;
                        int yy = candidate.Y + dy;
                        if (yy < 0) yy = 0;
                        if (yy >= size.Height) yy = size.Height - 1;
                        sum += lsEnhanced[xx, yy].Magnitude;
                    }
                }
                if (sum / count > tauLS)
                {
                    cnt++;
                    if (!minutiae.Any(pt => (pt.X - candidate.X) * (pt.X - candidate.X) + (pt.Y - candidate.Y) * (pt.Y - candidate.Y) < 30))
                        minutiae.Add(new Minutia() { X = candidate.X, Y = candidate.Y, Angle = ps[candidate.X, candidate.Y].Phase });
                }
            }

            var endList = minutiae.OrderByDescending(x =>
                ps[x.X, x.Y].Magnitude)
                .Take(MaxMinutiaeCount)
                .ToList();
            sw.Stop();
            return endList;
        }      
    }
}
