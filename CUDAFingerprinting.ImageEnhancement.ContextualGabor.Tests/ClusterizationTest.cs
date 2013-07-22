using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.ImageEnhancement.ContextualGabor;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor.Tests
{
    [TestClass]
    public class ClusterizationTest
    {
        [TestMethod]
        public void TestClusterization()
        {
            var img = ImageHelper.LoadImageAsInt(TestResources.goodFP);
            Normalizer.Normalize(100, 500, img);
            var info = RidgeFrequencyGenerator.GenerateBlocksInfo(img);
            var centers = new ClustersGenerator.ClusterPoint[6];

            img = ImageHelper.LoadImageAsInt(TestResources._1);
            Normalizer.Normalize(100, 500, img);
            info.AddRange(RidgeFrequencyGenerator.GenerateBlocksInfo(img));
            img = ImageHelper.LoadImageAsInt(TestResources._2);
            Normalizer.Normalize(100, 500, img);
            info.AddRange(RidgeFrequencyGenerator.GenerateBlocksInfo(img));
            img = ImageHelper.LoadImageAsInt(TestResources._3);
            Normalizer.Normalize(100, 500, img);
            info.AddRange(RidgeFrequencyGenerator.GenerateBlocksInfo(img));
            img = ImageHelper.LoadImageAsInt(TestResources._4);
            Normalizer.Normalize(100, 500, img);
            info.AddRange(RidgeFrequencyGenerator.GenerateBlocksInfo(img));
            img = ImageHelper.LoadImageAsInt(TestResources._5);
            Normalizer.Normalize(100, 500, img);
            info.AddRange(RidgeFrequencyGenerator.GenerateBlocksInfo(img));
            img = ImageHelper.LoadImageAsInt(TestResources._6);
            Normalizer.Normalize(100, 500, img);
            info.AddRange(RidgeFrequencyGenerator.GenerateBlocksInfo(img));
            img = ImageHelper.LoadImageAsInt(TestResources._7);
            Normalizer.Normalize(100, 500, img);
            info.AddRange(RidgeFrequencyGenerator.GenerateBlocksInfo(img));
            img = ImageHelper.LoadImageAsInt(TestResources._8);
            Normalizer.Normalize(100, 500, img);
            info.AddRange(RidgeFrequencyGenerator.GenerateBlocksInfo(img));


            Random rand = new Random();
            for (int i = 0; i < 6; i++)
            {
                centers[i] = info[rand.Next(0, info.Count)];
            }
            var centroids = ClustersGenerator.Clusterization(info, centers);




            img = ImageHelper.LoadImageAsInt(TestResources._1);
            Normalizer.Normalize(100, 500, img);
            int W = 16;
            int maxY = img.GetLength(0) / W;
            int maxX = img.GetLength(1) / W;
            var p = new Pen(Color.White);
            var image = ImageHelper.SaveArrayToBitmap(img.Select2D(x => (double)x));
            var g = Graphics.FromImage(image);
            info = RidgeFrequencyGenerator.GenerateBlocksInfo(img);
            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    g.DrawLine(p, new Point(j * W, i * W), new Point(j * W, i * W + W));
                    g.DrawLine(p, new Point(j * W, i * W), new Point(j * W + W, i * W));
                    g.DrawLine(p, new Point(j * W + W, i * W + W), new Point(j * W, i * W + W));
                    g.DrawLine(p, new Point(j * W + W, i * W + W), new Point(j * W + W, i * W));
                    int cl = info[i * W + j].FindClosestIndex(centroids);
                    g.DrawString(cl.ToString(), new Font("Times New Roman", 12), Brushes.Black, new PointF(j * W, i * W));
                }
            }

            /*var path = ClustersGenerator.ClusterPath;
            Process.Start(path);*/
            g.Save();
            string path = Path.GetTempPath() + "clusters.png";
            var res = ImageHelper.LoadImageAsInt(image);
            ImageHelper.SaveIntArray(res, path);
            Process.Start(path);
        }
    }
}
