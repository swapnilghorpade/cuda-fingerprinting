using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.ImageEnhancement.ContextualGabor;
using System.IO;
using System.Diagnostics;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor.Tests
{
    [TestClass]
    public class NormalizatorTest
    {
        [TestMethod]
        public void Normalize()
        {
            var img = ImageHelper.LoadImageAsInt(TestResources.sample);
            Normalizer.Normalize(100, 100, img);
            double x = Normalizer.Variance(img);
            double y = Normalizer.Mean(img);
            var path = Path.GetTempPath() + "normalizied.png";
            ImageHelper.SaveIntArray(img, path);
            Process.Start(path);
        }
    }
}
