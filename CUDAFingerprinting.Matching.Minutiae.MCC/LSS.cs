using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
namespace CUDAFingerprinting.Matching.Minutiae.MCC
{
    public static class LSS
    {
        private static void TryToAdd(float[] arr, float value)
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


        public static float GetScore(float[,] Gamma, int np)        //Local Similarity Sort
        {
            float[] arrOfMax = new float[np];
            for (int i = 0; i < np; i++)
                arrOfMax[i] = 0;
            for (int i = 0; i < Gamma.GetLength(0); i++)
                for (int j = 0; j < Gamma.GetLength(1); j++)
                    TryToAdd(arrOfMax, Gamma[i, j]);
            float score = 0;
            for (int i = 0; i < np; i++)
                score += arrOfMax[i];
            score = score/((float) np);
                return score;
        }

        public static float GetScoreVersion2(float[,] Gamma, int np)
        {
            float[] arr = new float[Gamma.GetLength(0) * Gamma.GetLength(1)];
            for (int i = 0; i < Gamma.GetLength(0); i++)
                for (int j = 0; j < Gamma.GetLength(1); j++)
                    arr[i*Gamma.GetLength(1) + j] = Gamma[i, j];
            Array.Sort(arr);
            Array.Reverse(arr);
            float score = 0;
            int count = Math.Min(np, arr.Length);
            for (int i = 0; i < count; i++)
                score += arr[i];
            score = score / (float)count;
            return score;
        }
    }
}
