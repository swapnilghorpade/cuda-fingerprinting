using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Collections;
using FingerprintPhD.Common;

namespace ComplexFilterQA
{
    internal class Program
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

        static double sigmaDirection = 2d;
        static int directionSize = KernelHelper.GetKernelSizeForGaussianSigma(sigmaDirection);

        private static Point[,] directions = new Point[directionSize, 20];

        
        private static void Main(string[] args)
        {
            ImageHelper.SaveBinaryAsImage("C:\\temp\\check.bin","C:\\temp\\check.png",true);

            //for (int i = 1; i <= 110; i++)
            //{
            //    for (int j = 1; j <=8; j++)
            //    {
            //        try
            //        {
            //            ImageHelper.SaveBinaryAsImage(string.Format("C:\\temp\\enh\\{0}_{1}.bin", i, j), string.Format("C:\\temp\\enh_img\\{0}_{1}.png", i, j), true);
            //        }
            //        catch (Exception)
            //        {
            //            Console.WriteLine("Check {0}_{1}", i, j);
            //        }

            //    }
            //}

            //for (int i = 1; i <= 80; i++)
            //{
            //    for (int j = 5; j <= 5; j++)
            //    {
            //        try
            //        {
            //            var mins = MinutiaeMatcher.LoadMinutiae(string.Format("C:\\temp\\min\\{0}_{1}.min", i, j));
            //            ImageHelper.MarkMinutiae(string.Format("C:\\temp\\enh_img\\{0}_{1}.png", i, j), mins,
            //                string.Format("C:\\temp\\mark_img\\{0}_{1}.png", i, j));
            //        }
            //        catch (Exception)
            //        {
            //            Console.WriteLine("Check {0}_{1}", i, j);
            //        }

            //    }
            //}

            using(var br = new BinaryReader(new FileStream("C:\\temp\\ZeeBigResult.bin", FileMode.Open, FileAccess.Read)))
            {
                int[] same = new int[33];
                int[] diff = new int[33];

                for(int i=0;i<33;i++)
                {
                    same[i] = br.ReadInt32();
                }
                for (int i = 0; i < 33; i++)
                {
                    diff[i] = br.ReadInt32();
                }
                using(var sw = new StreamWriter(new FileStream("C:\\temp\\ZeeBigResult.csv", FileMode.Create, FileAccess.Write)))
                {
                    for(int i=0;i<33;i++)
                    {
                        sw.WriteLine("{0};{1};{2}",i,same[i],diff[i]);
                    }
                }
            }

            FillDirections();
            //var path1 = "C:\\temp\\enh\\6_7.tif";
            //var path2 = "C:\\temp\\enh\\6_6.tif";
            var path3 = "C:\\temp\\104_6_enh.png";
            var path4 = "C:\\temp\\enh\\108_8.tif";
            
            //ImageHelper.MarkMinutiae(path, ProcessFingerprint(ImageHelper.LoadImage(path)), "C:\\temp\\6_7_out.png");
            //var minutiae1 = ProcessFingerprint(ImageHelper.LoadImage(path1));
            //var minutiae2 = ProcessFingerprint(ImageHelper.LoadImage(path2));
            var minutiae3 = ProcessFingerprint(ImageHelper.LoadImage(path3));
            ImageHelper.MarkMinutiae(path3,minutiae3,"C:\\temp\\104_6_enh_marked.png");
            var minutiae4 = ProcessFingerprint(ImageHelper.LoadImage(path4));

            //var score1 = MinutiaeMatcher.Match(minutiae1, minutiae2);
            //var score2 = MinutiaeMatcher.Match(minutiae1, minutiae3);
            //var score3 = MinutiaeMatcher.Match(minutiae1, minutiae4);
        }

        private static void FillDirections()
        {
            for (int n = 0; n < 10; n++)
            {
                var angle = Math.PI*n/20;

                directions[directionSize/2, n] = new Point(0, 0);
                var tan = Math.Tan(angle);
                if (angle <= Math.PI/4)
                {
                    for (int x = 1; x <= directionSize/2; x++)
                    {
                        var y = (int) Math.Round(tan*x);
                        directions[directionSize/2 + x, n] = new Point(x, y);
                        directions[directionSize/2 - x, n] = new Point(-x, -y);
                    }
                }
                else
                {
                    for (int y = 1; y <= directionSize/2; y++)
                    {
                        var x = (int) Math.Round((double) y/tan);
                        directions[directionSize/2 + y, n] = new Point(x, y);
                        directions[directionSize / 2 - y, n] = new Point(-x, -y);
                    }
                }
            }
            for (int n = 10; n < 20; n++)
            {
                for (int i = 0; i < directionSize; i++)
                {
                    var p = directions[i, n - 10];
                    directions[i, n] = new Point(p.Y, -p.X);
                }
            }
        }

        private static List<Minutia> ProcessFingerprint(double[,] imgBytes)
        {
            var lsEnhanced = SymmetryHelper.EstimateLS(imgBytes, sigma1, sigma2);
            var psEnhanced = SymmetryHelper.EstimatePS(imgBytes, sigma1, sigma2);
            //ImageHelper.SaveComplexArrayAsHSV(lsEnhanced,"C:\\temp\\lsenh.png");

            //ImageHelper.SaveArray(NormalizeArray(psEnhanced.Select2D(x=>x.Magnitude)), "C:\\temp\\psenh.png");

            var psi = KernelHelper.Zip2D(psEnhanced,
                lsEnhanced.Select2D(x=>x.Magnitude), (x, y) => 
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
                    for (int dRow = -NeighborhoodSize/2; dRow <= NeighborhoodSize/2; dRow++)
                    {
                        for (int dColumn = -NeighborhoodSize / 2; dColumn <= NeighborhoodSize / 2; dColumn++)
                        {
                            var correctRow = row + dRow;
                            var correctColumn = column + dColumn;
                            
                            if( correctRow > 9 && correctColumn > 9 
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
                    if(!maxP.IsEmpty)
                        grid[maxP.X, maxP.Y]++;
                    return maxP;
                });

            Dictionary<Point, int> responses = new Dictionary<Point, int>();

            foreach (var point in maxs)
            {
                if (!responses.ContainsKey(point)) responses[point] = 0;
                responses[point] ++;
            }


            ImageHelper.SaveArrayAsBinary(grid,"C:\\temp\\grid.bin");
            var orderedListOfCandidates =
                responses.Where(x => x.Value >= 20 && x.Key.X > 0 && x.Key.Y > 0 && x.Key.X < psi.GetLength(0)-1 && x.Key.Y < psi.GetLength(1)-1)
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

        private static double[,] NormalizeArray(double[,] data)
        {
            return ImageEnhancementHelper.RearrangeArray(data, 0, 1);
        }
    }
}
