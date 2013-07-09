using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.ImageEnhancement.ContextualGabor;
using System.IO;
using System.Diagnostics;
using System.Drawing;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor.Tests
{
    [TestClass]
    public class LocalRidgeOrientationTest
    {
        [TestMethod]
        public void LROTest()
        {
            var img = ImageHelper.LoadImageAsInt(TestResources.sample);
            Normalizer.Normalize(100, 500, img);
            var path = Path.GetTempPath() + "oriented.png";
            var lro = Common.OrientationField.OrientationFieldGenerator.GenerateOrientationField(img);
            var image = ImageHelper.SaveArrayToBitmap(img.Select2D(x => (double)x));
            const int frameSize = 24;
            var p = new Pen(Color.White);
            var g = Graphics.FromImage(image);
            var test = lro.Select2D(x => x * 180 / Math.PI);
            int maxY = img.GetLength(0) / frameSize;
            int maxX = img.GetLength(1) / frameSize;

            var myLRO = OrientationFieldGenerator.GenerateLocalRidgeOrientation(img);

            // My test
            int w = 16;
            for (int i = 0; i < (img.GetLength(0) / w - 1); i++)
            {
                for (int j = 0; j < (img.GetLength(1) / w - 1); j++)
                {
                    g.DrawString(Convert.ToInt32(myLRO[i * w + w / 2, j * w + w / 2] * 180 / Math.PI).ToString(),
                        new Font(FontFamily.GenericSansSerif, 10, FontStyle.Regular), new SolidBrush(Color.White),
                        new Point(j * w, i * w));

                    g.DrawLine(p, new Point(j * w, i * w), new Point(j * w + w, i * w));
                    g.DrawLine(p, new Point(j * w, i * w), new Point(j * w, i * w + w));
                    g.DrawLine(p, new Point(j * w + w, i * w + w), new Point(j * w, i * w + w));
                    g.DrawLine(p, new Point(j * w + w, i * w + w), new Point(j * w + w, i * w));
                }
            }



           /* for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {

                    if ((lro[i, j] >= 0 && lro[i, j] < Math.PI / 2) || (lro[i, j] >= Math.PI && lro[i, j] < 3 * Math.PI / 2))
                    {
                        if (lro[i, j] >= Math.PI && lro[i, j] < 3 * Math.PI / 2)
                            lro[i, j] -= Math.PI;
                        lro[i, j] = Math.PI - lro[i, j];
                        int xAxis = j * frameSize + frameSize;
                        int yAxis = i * frameSize;
                        var middle = new Point(j * frameSize + frameSize / 2, i * frameSize + frameSize / 2);
                        int x = xAxis;
                        int y = Convert.ToInt32((x - middle.X) * Math.Tan(lro[i, j]) + middle.Y);
                        if (y < yAxis)
                        {
                            y = yAxis;
                            x = Convert.ToInt32((y - middle.Y) / Math.Tan(lro[i, j]) + middle.X);
                        }
                        g.DrawLine(p, middle, new Point(x, y));
                    }
                    
                    else if ((lro[i, j] > Math.PI / 2 && lro[i, j] < Math.PI) || (lro[i, j] > 3 * Math.PI / 2))
                    {
                        if (lro[i, j] > 3 * Math.PI / 2)
                            lro[i, j] -= Math.PI;
                        lro[i, j] = Math.PI - lro[i, j];
                        int xAxis = j * frameSize;
                        int yAxis = i * frameSize;
                        var middle = new Point(j * frameSize + frameSize / 2, i * frameSize + frameSize / 2);
                        int x = xAxis;
                        int y = Convert.ToInt32((x - middle.X) * Math.Tan(lro[i, j]) + middle.Y);
                        if (y < yAxis)
                        {
                            y = yAxis;
                            x = Convert.ToInt32((y - middle.Y) / Math.Tan(lro[i, j]) + middle.X);
                        }
                        g.DrawLine(p, middle, new Point(x, y));

                    }
                   
                    g.DrawLine(p, new Point(j * frameSize, i * frameSize), new Point(j * frameSize + frameSize, i * frameSize));
                    g.DrawLine(p, new Point(j * frameSize, i * frameSize), new Point(j * frameSize, i * frameSize + frameSize));
                    g.DrawLine(p, new Point(j * frameSize + frameSize, i * frameSize + frameSize), new Point(j * frameSize, i * frameSize + frameSize));
                    g.DrawLine(p, new Point(j * frameSize + frameSize, i * frameSize + frameSize), new Point(j * frameSize + frameSize, i * frameSize));
                }
            }*/

            g.Save();
            var res = ImageHelper.LoadImageAsInt(image);
            ImageHelper.SaveIntArray(res, path);
            Process.Start(path);
        }
    }
}
