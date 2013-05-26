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

        public HoughTransform(List<Tuple<int, int>> singularPointsPI, double[,] orientationField)
        {
            this.singularPointsPI = singularPointsPI;
            this.orientationField = orientationField;

            foreach (Tuple<int, int> singularPoint in singularPointsPI)
            {
                votes.Add(0);
            }
        }

        public void Transform(Tuple<int, int> point, PointsOfInterestStruct pointsOfInterest, double backgroundOrientation)
        {
            int xLength = pointsOfInterest.Points.GetLength(0);
            int yLength = pointsOfInterest.Points.GetLength(1);
            double coefficient = 0;

            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < yLength; j++)
                {
                    coefficient = Math.Tan(2 * (pointsOfInterest.Points[i, j] - backgroundOrientation));

                    if (coefficient * (i + pointsOfInterest.ShiftX - point.Item1) == j + pointsOfInterest.ShiftY - point.Item2)
                    {
                        votes[singularPointsPI.IndexOf(point)]++;
                    }
                }
            }
        }

        internal List<Tuple<int, int>> FilterThreshold()
        {
            int threshold = (int)((Constants.W * Constants.W) / 2);
            List<Tuple<int, int>> result = new List<Tuple<int, int>>();
            int max = 0;

            //votes = votes.FindAll(vote => vote > threshold);

            for (int i = 0; i < votes.Count; i++)
            {
                if (max < votes[i])
                {
                    max = votes[i];
                }

                if (votes[i] > 0)
                {
                    result.Add(singularPointsPI[i]);
                }
            }

            return result;
        }
    }
}
