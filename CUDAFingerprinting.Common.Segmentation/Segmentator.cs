using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.Common.Segmentation
{
    public static class Segmentator
    {
        private static int N;
        private static int M;
        private static bool[,] mask = new bool[N, M];

        public static bool[,] GetMask(int[] mask1D, int maskY, int imgX, int imgY, int windowSize)
        {
            bool[,] bigMask = new bool[imgX, imgY];

            bigMask = bigMask.Select2D((value, x, y) =>
                {
                    int xBlock = (int)(((double)x) / windowSize);
                    int yBlock = (int)(((double)y) / windowSize);
                    return mask1D[xBlock + yBlock * maskY] == 1;
                });

            return bigMask;
        }

        public static double[,] Segmetator(double[,] img, int windowSize, double weight, int threshold)
        {
            int[,] xGradients = OrientationFieldGenerator.GenerateXGradients(img.Select2D(a => (int)a));
            int[,] yGradients = OrientationFieldGenerator.GenerateYGradients(img.Select2D(a => (int)a));
            double[,] magnitudes =
                xGradients.Select2D(
                    (value, x, y) => Math.Sqrt(xGradients[x, y] * xGradients[x, y] + yGradients[x, y] * yGradients[x, y]));
            double averege = KernelHelper.Average(magnitudes);
            double[,] window = new double[windowSize, windowSize];

            N = (int)Math.Ceiling(((double)img.GetLength(0)) / windowSize);
            M = (int)Math.Ceiling(((double)img.GetLength(1)) / windowSize);

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    window = window.Select2D((x, y, value) =>
                    {
                        if (i * windowSize + x >= magnitudes.GetLength(0)
                            || j * windowSize + y >= magnitudes.GetLength(1))
                        {
                            return 0;
                        }

                        return magnitudes[(int)(i * windowSize + x), j * windowSize + y];
                    });

                    if (KernelHelper.Average(window) < averege * weight)
                    {
                        mask[i, j] = false;
                    }
                    else
                    {
                        mask[i, j] = true;
                    }
                }
            }

            mask = PostProcessing(mask, threshold);

            return ColorImage(img, mask, windowSize);
        }

        public static double[,] ColorImage(double[,] img, bool[,] mask, int windowSize)
        {
            img = img.Select2D((value, x, y) =>
                {
                    int xBlock = (int)(((double)x) / windowSize);
                    int yBlock = (int)(((double)y) / windowSize);
                    return mask[xBlock, yBlock] ? img[x, y] : 0;
                });

            return img;
        }

        private static bool[,] PostProcessing(bool[,] mask, int threshold)
        {
            var blackAreas = GenerateBlackAreas(mask);
            var toRestore = new List<Tuple<int, int>>();


            foreach (var blackArea in blackAreas)
            {
                if (blackArea.Value.Count < threshold &&
                    !IsNearBorder(blackArea.Value, mask.GetLength(0), mask.GetLength(1)))
                {
                    toRestore.AddRange(blackArea.Value);
                }
            }

            var newMask = ChangeColor(toRestore, mask);
            var imageAreas = GenerateImageAreas(newMask);
            toRestore.Clear();


            foreach (var imageArea in imageAreas)
            {
                if (imageArea.Value.Count < threshold &&
                    !IsNearBorder(imageArea.Value, mask.GetLength(0), mask.GetLength(1)))
                {
                    toRestore.AddRange(imageArea.Value);
                }
            }

            return ChangeColor(toRestore, newMask);
        }

        private static Dictionary<int, List<Tuple<int, int>>> GenerateBlackAreas(bool[,] mask)
        {
            Dictionary<int, List<Tuple<int, int>>> areas = new Dictionary<int, List<Tuple<int, int>>>();

            int areaIndex = 0;

            for (int i = 0; i < mask.GetLength(0); i++)
            {
                for (int j = 0; j < mask.GetLength(1); j++)
                {
                    if (mask[i, j])
                    {
                        continue;
                    }
                    if (i - 1 >= 0 && !mask[i - 1, j] //left block is black
                        && (j - 1 >= 0 && mask[i, j - 1] || j - 1 < 0)) //top block is not black or not exist
                    {
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i - 1, j)))
                            {
                                area.Value.Add((new Tuple<int, int>(i, j)));
                            }
                        }
                        continue;
                    }
                    if (j - 1 >= 0 && !mask[i, j - 1] //top block is black 
                        && (i - 1 >= 0 && mask[i - 1, j] || i - 1 < 0)) //left block is not black or not exist
                    {
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i, j - 1)))
                            {
                                area.Value.Add((new Tuple<int, int>(i, j)));
                            }
                        }
                        continue;

                    }
                    if (j - 1 >= 0 && !mask[i, j - 1] //top block is black 
                        && i - 1 >= 0 && !mask[i - 1, j]) //left block is black
                    {
                        int areaNumberi = 0;
                        int areaNumberj = 0;
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i, j - 1)))
                            {
                                areaNumberj = area.Key;
                            }
                            if (area.Value.Contains(new Tuple<int, int>(i - 1, j)))
                            {
                                areaNumberi = area.Key;
                            }
                        }

                        if (areaNumberi != areaNumberj)
                        {
                            areas[areaNumberi].AddRange(areas[areaNumberj]);
                            areas[areaNumberi] = new List<Tuple<int, int>>(areas[areaNumberi].Distinct());
                            areas.Remove(areaNumberj);
                        }

                        areas[areaNumberi].Add(new Tuple<int, int>(i, j));
                        continue;
                    }
                    areas.Add(areaIndex, new List<Tuple<int, int>>());
                    areas[areaIndex].Add(new Tuple<int, int>(i, j));
                    areaIndex++;
                }

            }
            return areas;
        }

        private static Dictionary<int, List<Tuple<int, int>>> GenerateImageAreas(bool[,] mask)
        {
            var areas = new Dictionary<int, List<Tuple<int, int>>>();
            int areaIndex = 0;

            for (int i = 0; i < mask.GetLength(0); i++)
            {
                for (int j = 0; j < mask.GetLength(1); j++)
                {
                    if (!mask[i, j])
                    {
                        continue;
                    }

                    if (i - 1 >= 0 && mask[i - 1, j] && (j - 1 >= 0 && !mask[i, j - 1] || j - 1 < 0))
                    {
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i - 1, j)))
                            {
                                area.Value.Add((new Tuple<int, int>(i, j)));
                            }
                        }
                        continue;
                    }

                    if (j - 1 >= 0 && mask[i, j - 1] && (i - 1 >= 0 && !mask[i - 1, j] || i - 1 < 0))
                    {
                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i, j - 1)))
                            {
                                area.Value.Add((new Tuple<int, int>(i, j)));
                            }
                        }
                        continue;

                    }

                    if (j - 1 >= 0 && mask[i, j - 1] && i - 1 >= 0 && mask[i - 1, j])
                    {
                        int areaNumberi = 0;
                        int areaNumberj = 0;

                        foreach (var area in areas)
                        {
                            if (area.Value.Contains(new Tuple<int, int>(i, j - 1)))
                            {
                                areaNumberj = area.Key;
                            }
                            if (area.Value.Contains(new Tuple<int, int>(i - 1, j)))
                            {
                                areaNumberi = area.Key;
                            }
                        }

                        if (areaNumberi != areaNumberj)
                        {
                            areas[areaNumberi].AddRange(areas[areaNumberj]);
                            areas[areaNumberi] = new List<Tuple<int, int>>(areas[areaNumberi].Distinct());
                            areas.Remove(areaNumberj);
                        }
                        areas[areaNumberi].Add(new Tuple<int, int>(i, j));
                        continue;
                    }

                    areas.Add(areaIndex, new List<Tuple<int, int>>());
                    areas[areaIndex].Add(new Tuple<int, int>(i, j));
                    areaIndex++;
                }
            }

            return areas;
        }

        private static bool IsNearBorder(List<Tuple<int, int>> areas, int xBorder, int yBorder)
        {
            return areas.FindAll(area => area.Item1 == 0 ||
                                         area.Item2 == 0 ||
                                         area.Item1 == xBorder ||
                                         area.Item2 == yBorder
                ).Any();
        }

        private static bool[,] ChangeColor(List<Tuple<int, int>> areas, bool[,] mask)
        {
            foreach (var area in areas)
            {
                mask[area.Item1, area.Item2] = !mask[area.Item1, area.Item2];
            }

            return mask;
        }

    }
}













