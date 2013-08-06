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
            var img = ImageHelper.LoadImageAsInt(TestResources._2);
            Normalizer.Normalize(100, 500, img);
            var path = Path.GetTempPath() + "numbers.png";
            var lro = OrientationFieldGenerator.GenerateLocalRidgeOrientation(img);
            var image = ImageHelper.SaveArrayToBitmap(img.Select2D(x => (double)x));

            const int W = OrientationFieldGenerator.W;  
            int maxY = img.GetLength(0) / W;
            int maxX = img.GetLength(1) / W;
            var p = new Pen(Color.White);
            var g = Graphics.FromImage(image);
                      

            path = Path.GetTempPath() + "lines.png";
            image = ImageHelper.SaveArrayToBitmap(img.Select2D(x => (double)x));
            g = Graphics.FromImage(image);

            // Draw grid and lines with specified tangent
            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    // point in the middle of block
                    var middle = new Point(j * W + W / 2, i * W + W / 2);
                    // y = (x - m.x)*tan(a) + y1 
                    double angle = lro[i, j];

                    if (angle < 0)
                        angle += 2 * Math.PI;


                    // No difference between tan(a) and tan(a - pi), but it's more convenient to work with [0; pi]
                    if ((angle >= Math.PI && angle <= 3 * Math.PI / 2) || angle >= 3 * Math.PI / 2 && angle <= 2 * Math.PI)
                        angle -= Math.PI;

                    // If yes, then line intersects with y-axis of current block
                    if (angle <= Math.PI / 4 || angle >= 3 * Math.PI / 4)
                    {
                        // reverse of y-axis
                        angle = Math.PI - angle ;
                        Point b = new Point(j * W, Convert.ToInt32((j * W - middle.X) * Math.Tan(angle) + middle.Y));
                        Point e = new Point(j * W + W, Convert.ToInt32((j * W + W - middle.X) * Math.Tan(angle) + middle.Y));
                        g.DrawLine(p, b, e);
                    }
                    else // else intersecrs with x-axis of current block
                    {
                        // reverse of y-axis
                        angle = Math.PI - angle;
                        Point b = new Point(Convert.ToInt32((i * W + W - middle.Y) / Math.Tan(angle) + middle.X), i * W + W);
                        Point e = new Point(Convert.ToInt32((i * W - middle.Y) / Math.Tan(angle) + middle.X), i * W);
                        g.DrawLine(p, b, e);
                    }
                }
            }
            g.Save();
            var res = ImageHelper.LoadImageAsInt(image);
            ImageHelper.SaveIntArray(res, path);
            Process.Start(path);
        }
    }
}
