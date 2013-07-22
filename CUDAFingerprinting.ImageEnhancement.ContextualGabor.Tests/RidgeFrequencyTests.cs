using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Diagnostics;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.ImageEnhancement.ContextualGabor;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor.Tests
{
    [TestClass]
    public class RidgeFrequencyTests
    {
        [TestMethod]
        public void LocalMaxTest()
        {
            var testInput = new double[,,] { { {10, 3, 7, 3, 10, 1, 1, 1, 1, 1, 1, 12} } };
            double average = RidgeFrequencyGenerator.AverageDistanceBetweenLocalMax(testInput, 0, 0);
           // Assert.IsTrue(average == 2);
            var img = ImageHelper.LoadImageAsInt(TestResources.goodFP);
            Normalizer.Normalize(100, 500, img);
            // Check it in debug mode
            var x = RidgeFrequencyGenerator.GenerateInterpolatedFrequency(img);
            int a = 2;
        }
    }
}
