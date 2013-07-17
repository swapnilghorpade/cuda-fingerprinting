using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.Common.MCC
{
    class Constants
    {
        public static readonly int R = 70;
        public static readonly int Ns = 16; // 8
        public static readonly int Nd = 6;
        public static readonly double SigmaS = 28 / 3;
        public static readonly double SigmaD = 2*Math.PI / 9;
        public static readonly double MuPsi = 0.001;
        public static readonly int BigSigma = 50;
        public static readonly double MinVC = 0.75;
        public static readonly double MinM = 2;
        public static readonly double MinME = 0.6;
        public static readonly double SigmaTetta = Math.PI / 2;
    }
}
