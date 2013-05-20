using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;

namespace ModelBasedAlgorithm
{
    internal static class PoincareIndexMethod
    {
        public static Dictionary<int, int> FindSingularPoins(double[,] bytes)
        {
            Dictionary<int, int> points = new Dictionary<int, int>();
            KeyValuePair<int, int> newPoint = new KeyValuePair<int, int>();

            for (int i = 0; i < bytes.GetLength(0); i++)
            {
                for (int j = 0; j < bytes.GetLength(1); j++)
                {
                    if (IsSingularPoint(bytes[i,j]))
                    {
                        points.Add(i,j);
                    }
                }
            }

            return points;
        }

        private static bool IsSingularPoint(double point)
        {
            double result;

            ///

            return false;
        }
    }
}
