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
    public class OldLROTest
    {
        [TestMethod]
        public void LROTest()
        {
            var img = ImageHelper.LoadImageAsInt(TestResources.sample1);
            Normalizer.Normalize(100, 500, img);
            var path = Path.GetTempPath() + "oldnumbers.png";
            var image = ImageHelper.SaveArrayToBitmap(img.Select2D(x => (double)x));            
            const int W = 15;
            int maxY = img.GetLength(0) / W;
            int maxX = img.GetLength(1) / W;
            var p = new Pen(Color.White);
            var g = Graphics.FromImage(image);
            var lro = Common.OrientationField.OrientationFieldGenerator.GenerateOrientationField(img);
            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    g.DrawString(Convert.ToInt32(lro[i, j] * 180 / Math.PI).ToString(),
                        new Font("Times New Roman", 8), Brushes.White, new Point(j * W, i * W));
                    g.DrawLine(p, new Point(j * W, i * W), new Point(j * W + W, i * W));
                    g.DrawLine(p, new Point(j * W, i * W), new Point(j * W, i * W + W));
                    g.DrawLine(p, new Point(j * W + W, i * W + W), new Point(j * W, i * W + W));
                    g.DrawLine(p, new Point(j * W + W, i * W + W), new Point(j * W + W, i * W));
                }
            }
            g.Save();
            g.Dispose();
            var res = ImageHelper.LoadImageAsInt(image);
            ImageHelper.SaveIntArray(res, path);
            Process.Start(path);


            image = ImageHelper.SaveArrayToBitmap(img.Select2D(x => (double)x));
            g = Graphics.FromImage(image);
            path = Path.GetTempPath() + "oldlines.png";
            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    // point in the middle of block
                    var middle = new Point(j * W + W / 2, i * W + W / 2);

                    if (lro[i, j] < 0)
                        lro[i, j] += 2 * Math.PI;

                    // y = (x - m.x) * tan(a) + m.y: through point 'm' with specified angle

                    // No difference between tan(a) and tan(a - pi), but it's more convenient to work with [0; pi]
                    if ((lro[i, j] >= Math.PI && lro[i, j] <= 3 * Math.PI / 2) || lro[i, j] >= 3 * Math.PI / 2 && lro[i, j] <= 2 * Math.PI)
                        lro[i, j] -= Math.PI;

                    // If yes, then line intersects with y-axis of current block
                    if (lro[i, j] <= Math.PI / 4 || lro[i, j] >= 3 * Math.PI / 4)
                    {
                        // reverse of y-axis
                        lro[i, j] = Math.PI - lro[i, j];
                        Point b = new Point(j * W, Convert.ToInt32((j * W - middle.X) * Math.Tan(lro[i, j]) + middle.Y));
                        Point e = new Point(j * W + W, Convert.ToInt32((j * W + W - middle.X) * Math.Tan(lro[i, j]) + middle.Y));
                        g.DrawLine(p, b, e);
                    }
                    else // else intersecrs with x-axis of current block
                    {
                        // reverse of y-axis
                        lro[i, j] = Math.PI - lro[i, j];
                        Point b = new Point(Convert.ToInt32((i * W + W - middle.Y) / Math.Tan(lro[i, j]) + middle.X), i * W + W);
                        Point e = new Point(Convert.ToInt32((i * W - middle.Y) / Math.Tan(lro[i, j]) + middle.X), i * W);
                        g.DrawLine(p, b, e);
                    }
                    


                    g.DrawLine(p, new Point(j * W, i * W), new Point(j * W + W, i * W));
                    g.DrawLine(p, new Point(j * W, i * W), new Point(j * W, i * W + W));
                    g.DrawLine(p, new Point(j * W + W, i * W + W), new Point(j * W, i * W + W));
                    g.DrawLine(p, new Point(j * W + W, i * W + W), new Point(j * W + W, i * W));
                }
            }
            g.Save();
            res = ImageHelper.LoadImageAsInt(image);
            ImageHelper.SaveIntArray(res, path);
            Process.Start(path);
        }
    }
}
