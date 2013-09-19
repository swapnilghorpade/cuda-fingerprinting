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
    public class GaborTest
    {
        [TestMethod]
        public void GaborFilterTest()
        {
            var img = ImageHelper.LoadImageAsInt(TestResources._3);
            Normalizer.Normalize(100, 500, img);
            var path = Path.GetTempPath() + "final.png";
            var result = FilteredImageGenerator.GenerateFilteredImage(img);
            /*
             * так работает еще лучше :D
                var r = result.Select2D(x => (int)x);
                Normalizer.Normalize(150, 10000, r);
                var f = FilteredImageGenerator.GenerateFilteredImage(r.Select2D(x => (int)x));
                var s = f.Select2D(x => (int)x);
                Normalizer.Normalize(150, 10000, s);
             */
           // var r = result.Select2D(x => (int)x);
           // Normalizer.Normalize(150, 10000, r);
            //ImageHelper.SaveIntArray(r, path);
            var r = result.Select2D(x => (int)x);
            //Normalizer.Normalize(150, 10000, r);
            ImageHelper.SaveArray(r.Select2D(x => (double)x), path);
            Process.Start(path);
        }
    }
}
