using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC.Test
{
    [TestClass]
    public class MCCTest
    {
        [TestMethod]
        public void TestMcc()
        {
            var img = ImageHelper.LoadImage(Resources._104_6);
            double board = 5;
            var thining = Thining.ThiningPicture(GlobalBinarization.Binarization(img, board));
            List<Minutia> minutiaList = MinutiaeDetection.FindMinutiae(thining);
            MCC.MCCMethod(minutiaList, thining.GetLength(1), thining.GetLength(0)); // is it right?
        }
    }
}
