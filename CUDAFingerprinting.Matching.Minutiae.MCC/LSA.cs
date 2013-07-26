using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.Matching.Minutiae.MCC
{
    public static class LSA
    {
        private static float FindMin(bool[] usedRows, bool[] usedColumns, float[,] arr)
        {
            int dim = arr.GetLength(0);
            float min = 5001;
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    if ((!usedRows[i]) && (!usedColumns[j]))
                        if (arr[i, j] <min)
                            min = arr[i, j];
            return min;
        } //Find min value
        private static void ReduceMatrix(float[,] arr)
        {
            int dim = arr.GetLength(0);
            for (int i = 0; i < dim; i++)
            {
                float min = 5000;
                for (int j = 0; j < dim; j++)
                    if (arr[i, j] < min)
                        min = arr[i, j];
                for (int j = 0; j < dim; j++)
                    arr[i, j] = arr[i, j] - min;
            }
            for (int i = 0; i < dim; i++)
            {
                float min = 5000;
                for (int j = 0; j < dim; j++)
                    if (arr[j, i] < min)
                        min = arr[j, i];
                for (int j = 0; j < dim; j++)
                    arr[j, i] = arr[j, i] - min;
            }
        }//Find min value in matrix and reduce matrix by this value

        private static void ReduceMatrixBy(float[,] arr, bool[] usedRows, bool[] usedColumns, float value)
        {
            int dim = arr.GetLength(0);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    if ((!usedRows[i]) && (!usedColumns[j]))
                        arr[i, j] = arr[i, j] - value;
        }//Reduce matrix by the value

        private static bool FindWay(int FirstPoint, int[] tempUsedRows, int[] tempUsedColumns, bool[] usedRows,
                                    bool[] usedColumns, float[,] arr)
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
        }// Find Way from left to Right in Graph, that was built on our matrix


        private static void BuildMatching(List<float> Result, bool[] usedRows, bool[] usedColumns, float[,] arr,
                                          float[,] Gamma)
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
        }//Build Maximal Matching for our matrix

        private static float HungarianMethod(float[,] Gamma, int np)
        {
            float sum = 0;
            float max = 0;
            int dim = Math.Max(Gamma.GetLength(0), Gamma.GetLength(1));
            float[,] arr = new float[dim, dim];

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
            List<float> Result = new List<float>();
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
                float minValue = FindMin(usedRows, usedColumns, arr);
                ReduceMatrixBy(arr, usedRows, usedColumns, minValue);
                BuildMatching(Result, usedRows, usedColumns, arr, Gamma);
            }
            Result.Sort();
            Result.Reverse();
            for (int i = 0; i < np; i++)
                sum += Result[i];
            return sum;
        }  //Hungarian Method




        public static float GetScore(float[,] Gamma, int np)          //Local Similarity Assignment
        {
            float score = HungarianMethod(Gamma, np) / ((float)np);
            return score;
        }
    }
}
