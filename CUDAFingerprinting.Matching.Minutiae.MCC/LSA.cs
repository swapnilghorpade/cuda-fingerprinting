using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.Matching.Minutiae.MCC
{
    public static class LSA
    {
        private static double FindMin(bool[] usedRows, bool[] usedColumns, double[,] arr)
        {
            int dim = arr.GetLength(0);
            double min = 5001;
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    if ((!usedRows[i]) && (!usedColumns[j]))
                        if (arr[i, j] <min)
                            min = arr[i, j];
            return min;
        }
        private static void ReduceMatrix(double[,] arr)
        {
            int dim = arr.GetLength(0);
            for (int i = 0; i < dim; i++)
            {
                double min = 5000;
                for (int j = 0; j < dim; j++)
                    if (arr[i, j] < min)
                        min = arr[i, j];
                for (int j = 0; j < dim; j++)
                    arr[i, j] = arr[i, j] - min;
            }
            for (int i = 0; i < dim; i++)
            {
                double min = 5000;
                for (int j = 0; j < dim; j++)
                    if (arr[j, i] < min)
                        min = arr[j, i];
                for (int j = 0; j < dim; j++)
                    arr[j, i] = arr[j, i] - min;
            }
        }

        private static void ReduceMatrixBy(double[,] arr, bool[] usedRows, bool[] usedColumns, double value)
        {
            int dim = arr.GetLength(0);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    if ((!usedRows[i]) && (!usedColumns[j]))
                        arr[i, j] = arr[i, j] - value;
        }

        private static bool FindWay(int FirstPoint, int[] tempUsedRows, int[] tempUsedColumns, bool[] usedRows,
                                    bool[] usedColumns, double[,] arr)
        {
            int dim = arr.GetLength(0);
            bool Find = false;
            if (FirstPoint < 0)
            {
                for (int i = 0; (i < dim) && (!Find); i++)
                    if ((!usedRows[i]) && (tempUsedRows[i] < 0))
                        for (int j = 0; (j < dim) && (!Find); j++)
                            if (!usedColumns[j])
                                if (arr[i, j] == 0)
                                    if (tempUsedColumns[j] < 0)
                                    {
                                        tempUsedRows[i] = j;
                                        tempUsedColumns[j] = i;
                                        Find = true;
                                    }
                                    else
                                    {
                                        int k = tempUsedColumns[j];
                                        tempUsedRows[i] = j;
                                        tempUsedColumns[j] = i;
                                        tempUsedRows[k] = -1;
                                        arr[k, j] = 1;
                                        if (
                                            !FindWay(k, tempUsedRows, tempUsedColumns, usedRows, usedColumns,
                                                     arr))
                                        {
                                            tempUsedRows[k] = j;
                                            tempUsedRows[i] = -1;
                                            tempUsedColumns[j] = k;
                                        }
                                        else
                                            Find = true;
                                        arr[k, j] = 0;
                                    }
            }
            else
            {
                for (int j = 0; (j < dim) && (!Find); j++)
                    if (!usedColumns[j])
                        if (arr[FirstPoint, j] == 0)
                            if (tempUsedColumns[j] < 0)
                            {
                                tempUsedRows[FirstPoint] = j;
                                tempUsedColumns[j] = FirstPoint;
                                Find = true;
                            }
                            else
                            {
                                int k = tempUsedColumns[j];
                                tempUsedRows[FirstPoint] = j;
                                tempUsedColumns[j] = FirstPoint;
                                tempUsedRows[k] = -1;
                                arr[k, j] = 1;
                                if (
                                    !FindWay(k, tempUsedRows, tempUsedColumns, usedRows, usedColumns,
                                             arr))
                                {
                                    tempUsedRows[k] = j;
                                    tempUsedRows[FirstPoint] = -1;
                                    tempUsedColumns[j] = k;
                                }
                                else
                                    Find = true;
                                arr[k, j] = 0;
                            }
            }
            return Find;
        }


        private static void BuildMatching(List<double> Result, bool[] usedRows, bool[] usedColumns, double[,] arr,
                                          double[,] Gamma)
        {
            int dim = arr.GetLength(0);
            int[] tempUsedRows = new int[dim];
            int[] tempUsedColumns = new int[dim];
            for (int i = 0; i < dim; i++)
            {
                tempUsedRows[i] = -1;
                tempUsedColumns[i] = -1;
            }
            while (FindWay(-1, tempUsedRows, tempUsedColumns, usedRows, usedColumns, arr)) ;
            for (int i = 0; i < dim; i++)
                if (tempUsedRows[i] >= 0)
                {
                    if ((i < Gamma.GetLength(0)) && (tempUsedRows[i] < Gamma.GetLength(1)))
                        Result.Add(Gamma[i, tempUsedRows[i]]);
                    usedRows[i] = true;
                    usedColumns[tempUsedRows[i]] = true;
                }
        }

        private static double HungarianMethod(double[,] Gamma, int np)
        {
            double sum = 0;
            double max = 0;
            int dim = Math.Max(Gamma.GetLength(0), Gamma.GetLength(1));
            double[,] arr = new double[dim, dim];

            for (int i = 0; i < Gamma.GetLength(0); i++)
                for (int j = 0; j < Gamma.GetLength(1); j++)
                    if (Gamma[i, j] > max)
                        max = Gamma[i, j];

            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    if ((i < Gamma.GetLength(0)) && (j < Gamma.GetLength(1)))
                        arr[i, j] = max - Gamma[i, j];
                    else
                        arr[i, j] = 5000;
                }
            }
            ReduceMatrix(arr);
            List<double> Result = new List<double>();
            bool[] usedRows = new bool[dim];
            bool[] usedColumns = new bool[dim];
            for (int i = 0; i < dim; i++)
            {
                usedRows[i] = false;
                usedColumns[i] = false;
            }
            BuildMatching(Result, usedRows, usedColumns, arr, Gamma);
            double min = Math.Min(Gamma.GetLength(0), Gamma.GetLength(1));
            while (Result.Count < min)
            {
                double minValue = FindMin(usedRows, usedColumns, arr);
                ReduceMatrixBy(arr, usedRows, usedColumns, minValue);
                BuildMatching(Result, usedRows, usedColumns, arr, Gamma);
            }
            Result.Sort();
            Result.Reverse();
            for (int i = 0; i < np; i++)
                sum += Result[i];
            return sum;
        }




        public static double GetScore(double[,] Gamma, int np)          //Local Similarity Assignment
        {
            double score = HungarianMethod(Gamma, np) / ((double)np);
            return score;
        }
    }
}
