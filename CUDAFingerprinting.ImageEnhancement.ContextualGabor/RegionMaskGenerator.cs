using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor
{
    public class RegionMaskGenerator
    {
        public static void GenerateRegionMask(int[,] image)
        {

        }

        public static bool IsRecoverable(bool[,] regionMask)
        {
            int counter = 0;
            int unrecoverableCounter = 0;
            foreach (var r in regionMask)
            {
                ++counter;
                if (!r)
                    ++unrecoverableCounter;
            }
            return (double)unrecoverableCounter / (double)counter < 0.4;
        }
    }
}
