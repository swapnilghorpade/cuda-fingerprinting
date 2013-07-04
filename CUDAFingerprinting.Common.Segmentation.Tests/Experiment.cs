using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.Segmentation.Tests
{
    [TestClass]
    public class Experiment
    {
        [TestMethod]
        public void ExperimentMethod()
        {
            double[,] img = ImageHelper.LoadImage(Resources._104_6);
            double[,] resultImg;
            int windowSize = 20;
            double minValue = 0.3;
            double maxValue = 1;
            int threshold = 3;

            for (double weight = minValue; weight <= maxValue; weight += 0.1)
            {
                resultImg = Segmentator.Segmetator(img, windowSize, weight, threshold);
                ImageHelper.SaveArray(resultImg, Path.GetTempPath() + "resultImg_104_6__" + weight + ".png");
            }
        }
    }
}
