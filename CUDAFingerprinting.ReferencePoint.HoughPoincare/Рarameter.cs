using System;

namespace ModelBasedAlgorithmCUDAFingerprinting.ReferencePoint.HoughPoincare
{
    internal class Рarameter
    {
        private int x;
        private int y;
        private int vote = 0;

        public int X 
        {
            get { return x; }
        }

        public int Y 
        {
            get { return y; }
        }

        public int Vote
        {
            get { return vote; }
        }

        public Рarameter(int x, int y)
        {
            this.x = x;
            this.y = y;
        }

        public Рarameter(Tuple<int, int> point)
        {
            this.x = point.Item1;
            this.y = point.Item2;
        }

        public void IncreaseVote()
        {
            vote++;
        }
    }
}
