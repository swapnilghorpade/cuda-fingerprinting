using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;

namespace ModelBasedAlgorithm
{
    internal class ModelBasedAlgorithm
    {
        internal static List<Tuple<int, int>> FindSingularPoints(double[,] orientationField, List<Tuple<int, int>> singularPointsPI)
        {
            List<double[,]> blocks = new List<double[,]>();
            HoughTransform houghTransform = new HoughTransform(singularPointsPI);

            foreach (Tuple<int, int> point in singularPointsPI)
            {
                blocks = GetBlocks(orientationField, point);

                foreach (double[,] block in blocks)
                {
                    houghTransform.Transform(point, GetFeatureSpace(orientationField, point), GetBackgroundOrientation(block));
                }
            }

            // сравниваем значения вероятностей с порогом

            return new List<Tuple<int, int>>();
        }

        private static double GetBackgroundOrientation(double[,] block)
        {
            int xLength =  block.GetLength(0);
            int yLength =  block.GetLength(1);
            double value = 0;

            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < yLength; j++)
                {
                    value += block[i, j];
                }
            }

            return value / (xLength * yLength);
        }

        private static double[,] GetFeatureSpace(double[,] orientationField, Tuple<int, int> point)
        {
            double[,] result = new double[Constants.W, Constants.W]();

            int upperBound = (int)(Constants.W / 2);
            int lowerBound = -1 * upperBound;

            for (int x = lowerBound; x < upperBound; x++)
            {
                for (int y = lowerBound; y < upperBound; y++)
                {
                    result[x - lowerBound, y - lowerBound] = orientationField[point.Item1 + x, point.Item2 + y];
                }
            }

            return result;
        }

        private static List<double[,]> GetBlocks(double[,] orientationField, Tuple<int, int> point)
        {
            List<double[,]> result = new List<double[,]>();
            int upperBound = (int)(Constants.WNum * Constants.WSize / 2);
            int lowerBound = -1 * upperBound;

            for (int i = 0; i < Constants.WNum * Constants.WNum; i++)
            {
                result.Add(new double[Constants.WSize, Constants.WSize]);
            }

            for (int x = lowerBound; x < upperBound; x++)
            {
                for (int y = lowerBound; y < upperBound; y++)
                {
                    if (point.Item1 + x < 0 || point.Item2 + y < 0
                        || point.Item1 + x >= orientationField.GetLength(0)
                        || point.Item2 + y >= orientationField.GetLength(1))
                    {
                        Console.WriteLine("Block out bounds.");
                        continue;
                    }

                    int numberOfBlock = Constants.WNum * (int)((y - lowerBound) / Constants.WSize)
                                                       + (int)((x - lowerBound) / Constants.WSize);
                    int xBlock = (x - lowerBound) - Constants.WSize * ((x - lowerBound) / Constants.WSize);
                    int yBlock = (y - lowerBound) - Constants.WSize * ((y - lowerBound) / Constants.WSize);

                    (result[numberOfBlock])[xBlock, yBlock] = orientationField[point.Item1 + x, point.Item2 + y];
                }
            }

            return result;
        }
    }
}
