using System;
using System.Collections.Generic;

namespace ModelBasedAlgorithmCUDAFingerprinting.ReferencePoint.HoughPoincare
{
    internal class ModelBasedAlgorithm
    {
        private double[,] orientationField;
        private int upperBound = 0;
        private int lowerBound = 0;

        public ModelBasedAlgorithm(double[,] orientationField)
        {
            this.orientationField = orientationField;
            upperBound = (int)(Constants.WNum * Constants.WSize / 2);
            lowerBound = -1 * upperBound;
        }

        internal List<Tuple<int, int>> FindSingularPoints(List<Tuple<int, int>> singularPointsPI)
        {
            List<Tuple<int, int>> result = new List<Tuple<int, int>>();
            HoughTransform houghTransform = new HoughTransform(singularPointsPI, orientationField);
            List<double[,]> blocks = new List<double[,]>();
            double backgroundOrientation;

            foreach (Tuple<int, int> point in singularPointsPI)
            {
                if (!IsValidPointPosition(point))
                {
                    continue;
                }

               blocks = GetBlocks(point);

                foreach (double[,] block in blocks)
                {
                    backgroundOrientation = GetBackgroundOrientation(block);
                    houghTransform.Transform(point, backgroundOrientation);
                }
            }

            return houghTransform.FilterThreshold();
        }

        private bool IsValidPointPosition(Tuple<int, int> point)
        {
            return point.Item1 + lowerBound >= 0 && point.Item2 + lowerBound >= 0
              && point.Item1 + upperBound < orientationField.GetLength(0)
              && point.Item2 + upperBound < orientationField.GetLength(1);
        }

        private double GetBackgroundOrientation(double[,] block)
        {
            int xLength = block.GetLength(0);
            int yLength = block.GetLength(1);
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

        private List<double[,]> GetBlocks(Tuple<int, int> point)
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
