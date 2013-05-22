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
        public static List<Tuple<int, int>> FindSingularPoins(double[,] orientationField)
        {
            List<Tuple<int, int>> result = new List<Tuple<int, int>>();

            for (int i = 0; i < orientationField.GetLength(0); i++)
            {
                for (int j = 0; j < orientationField.GetLength(1); j++)
                {
                    if (IsSingularPoint(orientationField, i, j))
                    {
                        result.Add(new Tuple<int, int>(i, j));
                    }
                }
            }

            return result;
        }

        private static bool IsSingularPoint(double[,] orientationField, int xPoint, int yPoint)
        {
            if (xPoint - Constants.AreaPI < 0 || xPoint + Constants.AreaPI >= orientationField.GetLength(0)
                || yPoint - Constants.AreaPI < 0 || yPoint + Constants.AreaPI >= orientationField.GetLength(1))
            {
                return false;
            }

            List<Tuple<int, int>> points = new List<Tuple<int, int>>();
            int x = xPoint - Constants.AreaPI;
            int y = yPoint - Constants.AreaPI;
            Tuple<int, int> currentPoint;
            Tuple<int, int> nextPoint = new Tuple<int, int>(x, y);
            double sum = 0;

            for (int k = 0; k < 8; k++)
            {
               points.Add(nextPoint);

                currentPoint = nextPoint;

                if (x < xPoint + Constants.AreaPI && y == yPoint - Constants.AreaPI)
                {
                    x++;
                    nextPoint = new Tuple<int, int>(x, y);
                    
                }
                else if (x == xPoint + Constants.AreaPI && y < yPoint + Constants.AreaPI)
                {
                    y++;
                    nextPoint = new Tuple<int, int>(x, y);
                }
                else if (x != xPoint - Constants.AreaPI && y == yPoint + Constants.AreaPI)
                {
                    x--;
                    nextPoint = new Tuple<int, int>(x, y);
                }
                else if (x == xPoint - Constants.AreaPI && y == yPoint + Constants.AreaPI)
                {
                    y--;
                    nextPoint = new Tuple<int, int>(x, y);
                }
                else
                {
                    nextPoint = points[0];
                }

                sum += FunctionF(orientationField[nextPoint.Item1, nextPoint.Item2] 
                                - orientationField[currentPoint.Item1, currentPoint.Item2]);
            }

            sum = sum / Math.PI;

            return (int)sum != 0;
        }

        private static double FunctionF(double arg)
        {
            if (Math.Abs(arg) <= Math.PI / 2)
            {
                return arg;
            }

            if (arg > Math.PI / 2)
            {
                return Math.PI - arg;
            }

            if (arg < Math.PI / -2)
            {
                return Math.PI + arg;
            }

            throw new ArgumentException("FunctionF");
        }
    }
}
