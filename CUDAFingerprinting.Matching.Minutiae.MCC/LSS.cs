using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
namespace CUDAFingerprinting.Matching.Minutiae.MCC
{
    public static class LSS
    {
        private static void TryToAdd(double[] arr, double value)
        {
            bool InsertionOccurred = false;
            for (int i = 0; (i < arr.GetLength(0)) && (!InsertionOccurred); i++)
                if (value > arr[i])
                {
                    InsertionOccurred = true;
                    for (int j = arr.GetLength(0)-1; j > i; j--)
                        arr[j] = arr[j - 1];
                    arr[i] = value;
                }
        }


        public static double GetScore(double[,] Gamma, int np)        //Local Similarity Sort
        {
            double[] arrOfMax = new double[np];
            for (int i = 0; i < np; i++)
                arrOfMax[i] = 0;
            for (int i = 0; i < Gamma.GetLength(0); i++)
                for (int j = 0; j < Gamma.GetLength(1); j++)
                    TryToAdd(arrOfMax, Gamma[i, j]);
            double score = 0;
            for (int i = 0; i < np; i++)
                score += arrOfMax[i];
            score = score/((double) np);
                return score;
        }
    }
}
