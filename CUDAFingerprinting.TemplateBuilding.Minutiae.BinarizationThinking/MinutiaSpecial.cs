using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking
{
    public struct MinutiaSpecial
    {
        public int X;
        public int Y;
        public double Angle;
        public int numberMinutiaeInCircle;
        public bool belongToBig;
    }
}
