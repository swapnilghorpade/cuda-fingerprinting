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
            double[,] img1 = ImageHelper.LoadImage(Resources._104_6);
            double[,] img2 = ImageHelper.LoadImage(Resources._65_8);
            double[,] img3 = ImageHelper.LoadImage(Resources._103_7);
            double[,] resultImg1;
            double[,] resultImg2;
            double[,] resultImg3;

            int windowSize = 20;
            double minValue = 0.3;
            double maxValue = 1;
            int minThreshold = 2;
            int maxThreshold = 7;
            for (double weight = minValue; weight <= maxValue; weight+= 0.1)
            {
                for (int currentThreshold = minThreshold; currentThreshold <= maxThreshold; currentThreshold++)
                {
                    resultImg1 = Segmentator.Segmetator(img1, windowSize, weight, currentThreshold);
                    ImageHelper.SaveArray(resultImg1, Path.GetTempPath() + "/104_6/resultImg_" + weight + "_" + currentThreshold + ".png");
                    resultImg2 = Segmentator.Segmetator(img2, windowSize, weight, currentThreshold);
                    ImageHelper.SaveArray(resultImg2, Path.GetTempPath() + "/65_8/resultImg_" + weight + "_" + currentThreshold + ".png");
                    resultImg3 = Segmentator.Segmetator(img3, windowSize, weight, currentThreshold);
                    ImageHelper.SaveArray(resultImg3, Path.GetTempPath() + "/103_7/resultImg_" + weight + "_" + currentThreshold + ".png");

                }
            }
        }
    }
}
