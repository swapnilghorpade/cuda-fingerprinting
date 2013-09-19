using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Diagnostics;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.ImageEnhancement.EightGabor;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.ImageEnhancement.EightGaborTest
{
    [TestClass]
    public class FilterTest
    {
        [TestMethod]
        public void SingleFilterTest()
        {
            var img = ImageHelper.LoadImageAsInt(Resources._5);
            Normalizer.Normalize(100, 500, img);
            // 0.0174532925 - deg to rads
            var result = GaborGenerator.GenerateGaborFilter(img, 0);
            string path = Path.GetTempPath() + "0.png";
            ImageHelper.SaveArray(result, path);
            Process.Start(path);
        }
    }
}
