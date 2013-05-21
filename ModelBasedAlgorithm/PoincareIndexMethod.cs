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
        public static List<Point> FindSingularPoins(double[,] orientationField)
        {
            List<Point> result = new List<Point>();

            for (int i = 0; i < orientationField.GetLength(0); i++)
            {
                for (int j = 0; j < orientationField.GetLength(1); j++)
                {
                    if (IsSingularPoint(orientationField, i, j))
                    {
                        result.Add(new Point() { X = i, Y = j });
                    }
                }
            }

            return result;
        }

        private static bool IsSingularPoint(double[,] orientationField, int xPoint, int yPoint)
        {
            ////

            return false;
        }
    }
}
