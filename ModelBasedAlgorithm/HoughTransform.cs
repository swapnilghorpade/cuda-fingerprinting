using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelBasedAlgorithm
{
    internal class HoughTransform
    {
        private List<Tuple<int, int>> singularPointsPI = new List<Tuple<int, int>>();
        private double[,] orientationField;
        private List<int> votes = new List<int>();
        private Tuple<int, int> point;

        public HoughTransform(List<Tuple<int, int>> singularPointsPI, double[,] orientationField)
        {
            this.singularPointsPI = singularPointsPI;
            this.orientationField = orientationField;
            int initialValue = 0;

            foreach (Tuple<int, int> singularPoint in singularPointsPI)
            {
                votes.Add(initialValue);
            }
        }

        public void Transform(Tuple<int, int> point, double backgroundOrientation)
        {
            this.point = point;

            int xLength = orientationField.GetLength(0);
            int yLength = orientationField.GetLength(1);
            double coefficient = 0;

            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < yLength; j++)
                {
                    coefficient = Math.Tan(2 * (orientationField[i, j] - backgroundOrientation));

                    if (WhetherToVote(i, j, coefficient))
                    {
                        votes[singularPointsPI.IndexOf(point)]++;
                    }
                }
            }
        }

        private bool WhetherToVote(int i, int j, double coefficient)
        {
            int upperBound = (int)(Constants.W / 2);
            int lowerBound = -1 * upperBound;

            for (int xPointError = lowerBound; xPointError < upperBound; xPointError++)
            {
                for (int yPointError = lowerBound; yPointError < upperBound; yPointError++)
                {
                    if (coefficient * (i - point.Item1 - xPointError) == (j - point.Item2 - yPointError))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        public List<Tuple<int, int>> FilterThreshold()
        {
            if (votes.Count == 0)
            {
                return new List<Tuple<int, int>>();
            }

            List<Tuple<int, int>> result = new List<Tuple<int, int>>();
            int threshold = (int)((orientationField.GetLength(0) * orientationField.GetLength(1)) / 5);
            int max = votes.Max();

          /*  if (max < threshold)
            {
                Console.WriteLine("No singular points. Sorry.");
            }
           */

            for (int i = 0; i < votes.Count; i++)
            {
                if (votes[i] == max)
                {
                    result.Add(singularPointsPI[i]);
                }
            }

            return result;
        }
    }
}
