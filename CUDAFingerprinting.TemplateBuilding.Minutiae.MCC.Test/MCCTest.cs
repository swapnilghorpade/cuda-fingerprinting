using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.Segmentation;
using CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC.Test
{
    [TestClass]
    public class MCCTest
    {
        private int windowSize = 12;
        private double weight = 0.3;
        private int threshold = 5;

        [TestMethod]
        public void TestMcc()
        {
            var img = ImageHelper.LoadImage(Resources._104_6);
            bool[,] mask = Segmentator.GetMask(img, windowSize, weight, threshold);
            double board = 150;
            var thining = Thining.ThiningPicture(GlobalBinarization.Binarization(img, board));
            List<Minutia> minutiaList = MinutiaeDetection.FindMinutiae(thining);
            MCC.MCCMethod(minutiaList, thining.GetLength(1), thining.GetLength(0)); // is it right?
        }
    }
}
