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
            double[,] img = ImageHelper.LoadImage(Resources.TestImage);
            double[,] resultImg;
            int minValue = 5;
            int maxValue = 6;

            for (int windowRadius = minValue; windowRadius < maxValue; windowRadius++)
            {
               resultImg = Segmentator.Segmetator(img, windowRadius);
               ImageHelper.SaveArray(resultImg, Path.GetTempPath() + "resultImg" + windowRadius + ".png");
            }
        }
    }
}
