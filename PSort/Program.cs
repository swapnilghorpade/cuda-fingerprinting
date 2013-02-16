using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;

namespace PSort
{
    internal class Program
    {
        private static double FAR = 10e-6;
        private static double ln = Math.Log(1.0d - FAR);

        private static double phi(double x)
        {
            double power = Math.Pow(1.0d - FAR, x - 1);
            double f = x*power - 10.0d;
            double fD = power*(x*ln + 1);
            return x - f/fD;
        }

        private static double Combinations(long n, long k)
        {
            var limit = Math.Max(n - k, k);
            var divisor = Math.Min(n - k, k);

            BigInteger b = new BigInteger(1);
            for (long i = limit + 1; i <= n; i++) b *= i;
            for (long i = 2; i <= divisor; i++) b /= i;
            return (double) b;
        }

        private static void Main(string[] args)
        {
            using(var fs = new FileStream(".\\FingerprintPhD.Common.dll",FileMode.Open))
            using (BinaryReader sr = new BinaryReader(fs))
            {
                var arr = sr.ReadBytes(10000000);
                var ass = Assembly.Load(arr);
            }

            long M = 250000;
            long N = 4;
            long K = 1000;
            double FAR = 10e-9;
            double FRR = 10e-6;
            double PK = 1.0d - 10e-5;
            double result = 0;
            // FAR
            //for(int i=1;i<=N;i++)
            //{
            //    var semiResult = Math.Pow(FAR, i)*Math.Pow(1.0d - FAR, K - i)*M*Combinations(K,i);
            //    for(int j=0;j<=i-1;j++)
            //    {
            //        semiResult = semiResult*(N - j)/(M*N - j);
            //    }
            //    result += semiResult;
            //}

            //Console.WriteLine(result);
            // result = 0;
            //for (int i = 1; i <= N; i++)
            //{
            //    var semiResult = Math.Pow(FAR, i) * Math.Pow(1.0d - FAR, K - i) * M * Combinations(K, i);
            //    for (int j = 0; j <= i - 1; j++)
            //    {
            //        semiResult = semiResult * (N - j) / (M * N - j);
            //    }
            //    result += semiResult;
            //    break;
            //}

            for (int i = 1; i <= N; i++)
            {
                double semiResult =
                    Combinations(N, i)*
                    Math.Pow(PK, i)*Math.Pow(1.0d - PK, N - i);

                double otherHalf = 0;
                for (int p = 1; p <= i; p++)
                {
                    otherHalf +=
                        Combinations(i, p)*
                        Math.Pow(1.0d - FRR, p)*Math.Pow(1.0d - FAR, K - i)*
                        Math.Pow(FRR, i - p);
                }
                result += semiResult*otherHalf;
            }
            // no P
            Console.WriteLine(result);
            N = 4;
            result = 0;



            result +=
                (1.0d - Math.Pow(FRR, N))*Math.Pow(1.0d - FAR, M*N - N);
            



            Console.WriteLine(result);

            Console.ReadKey();
        }

    }
}
