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
    public class GaussianBlurTest
    {
        [TestMethod]
        public void BlurTest()
        {
            var img = ImageHelper.LoadImageAsInt(TestResources.goodFP);
            var blur = OrientationFieldGenerator.GenerateBlur(img.Select2D(x => (double)x));
            var path = Path.GetTempPath() + "blur.png";
            ImageHelper.SaveIntArray(blur.Select2D(x => (int)x), path);
            Process.Start(path);
        }
    }
}
