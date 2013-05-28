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

            int x = xPoint - Constants.AreaPI;
            int y = yPoint + Constants.AreaPI;
            Tuple<int, int> currentPoint;
            Tuple<int, int> nextPoint = new Tuple<int, int>(x, y);
            Tuple<int, int> firstPoint = new Tuple<int, int>(x, y);
            double indexPoincare = 0;
            int closedCurveSquare = (Constants.AreaPI * 2) * 4;

            for (int k = 0; k < closedCurveSquare; k++)
            {
                currentPoint = nextPoint;

                if (k == 7)
                {
                    nextPoint = firstPoint;
                }
                else if (x == xPoint - Constants.AreaPI && y > yPoint - Constants.AreaPI)
                {
                    y--;
                    nextPoint = new Tuple<int, int>(x, y);
                }
                else if (x < xPoint + Constants.AreaPI && y == yPoint - Constants.AreaPI)
                {
                    x++;
                    nextPoint = new Tuple<int, int>(x, y);
                }
                else if (x == xPoint + Constants.AreaPI && y < yPoint + Constants.AreaPI)
                {
                    y++;
                    nextPoint = new Tuple<int, int>(x, y);
                }
                else if (x > xPoint - Constants.AreaPI + 1 && y == yPoint + Constants.AreaPI)
                {
                    x--;
                    nextPoint = new Tuple<int, int>(x, y);
                }

                indexPoincare += FunctionF(orientationField[currentPoint.Item1, currentPoint.Item2]
                                            - orientationField[nextPoint.Item1, nextPoint.Item2]);
            }


            indexPoincare = indexPoincare / Math.PI;
            
            return indexPoincare == 1.0 || indexPoincare == 2.0;
        }

        private static double FunctionFVORIV(double arg)
        {
            if (arg >= 0 && arg <= Math.PI)
            {
                return arg;
            }

            if (arg < 0 && arg > Math.PI * (-1))
            {
                return arg * (-1);
            }

            return 0;
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
