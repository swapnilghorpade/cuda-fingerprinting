using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelBasedAlgorithm
{
    internal class Рarameter
    {
        private int x;
        private int y;
        private double p;
        private double tetta;
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

        public Рarameter(int x, int y, double p, double tetta)
        {
            this.x =x;
            this.y = y;
            this.p = p;
            this.tetta = tetta;
        }

        public void IncreaseVote()
        {
            vote++;
        }
    }
}
