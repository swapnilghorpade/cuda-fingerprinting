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
            var img = ImageHelper.LoadImage(TestResources.sample);
            Normalizer.Normalize(100, 100, img);
            var path = Path.GetTempPath() + "normalizied.png";
            ImageHelper.SaveArray(img, path);
            Process.Start(path);
        }
    }
}
