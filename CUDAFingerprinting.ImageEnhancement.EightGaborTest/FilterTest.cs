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
            var radialOrientation = new double[8] { 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5 };
            for (int i = 0; i < 8; i++)
            {
                // 0.0174532925 - deg to rads
                var result = GaborGenerator.GenerateGaborFilter(img, radialOrientation[i] * 0.0174532925);
                string path = Path.GetTempPath() + radialOrientation[i].ToString() + ".png";
                ImageHelper.SaveArray(result, path);
                Process.Start(path);
            }
        }
    }
}
