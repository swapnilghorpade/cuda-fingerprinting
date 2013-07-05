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
        public void NormalizeTest()
        {
            var img = ImageHelper.LoadImageAsInt(TestResources.sample);
            var f = ImageHelper.LoadImageAsInt(TestResources.fake);
            Normalizer.Normalize(100, 5000, img);
            Normalizer.LinearNormalize(0, 255, img);
            var path = Path.GetTempPath() + "normalizied.png";
            ImageHelper.SaveIntArray(img, path);
            Process.Start(path);
        }
    }
}
