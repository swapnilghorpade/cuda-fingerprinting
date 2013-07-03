using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking;
using CUDAFingerprinting.ImageEnhancement.ContextualGabor;
using System.IO;
using System.Diagnostics;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking.Tests
{
    [TestClass]
    public class GlobalBinarizationTest
    {
        [TestMethod]
        public void TestMethod1()
        {
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] binarization = GlobalBinarization.Binarization(img, 200d);
            var path = Path.GetTempPath() + "binarization.png";
            ImageHelper.SaveArray(img, path);
            Process.Start(path);
        }
    }
}
