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
        public void ShowClusters(int[,] img, ClustersGenerator.ClusterPoint[] centroids, int W)
        {
            Normalizer.Normalize(100, 500, img);
            int maxY = img.GetLength(0) / W;
            int maxX = img.GetLength(1) / W;
            var p = new Pen(Color.White);
            var image = ImageHelper.SaveArrayToBitmap(img.Select2D(x => (double)x));
            var g = Graphics.FromImage(image);
            var info = RidgeFrequencyGenerator.GenerateBlocksInfo(img);
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
            g.Save();
            var path = Path.GetTempPath() + "1.png";
            var tmp = ImageHelper.LoadImageAsInt(image);
            ImageHelper.SaveIntArray(tmp, path);
            Process.Start(path);
        }
        [TestMethod]
        public void TestClusterization()
        {
            const int W = 16;
            var img = ImageHelper.LoadImageAsInt(TestResources._1);
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
            
           
            var middle = new ClustersGenerator.ClusterPoint(0, 0, 0);

            foreach (var bl in info)
            {
                middle.Amplitude += bl.Amplitude;
                middle.Frequency += bl.Frequency;
                middle.Variance += bl.Variance;
            }

            middle.Amplitude /= info.Count;
            middle.Frequency /= info.Count;
            middle.Variance /= info.Count;


            Random rand = new Random();
            for (int i = 0; i < 6; i++)
            {
                centers[i] = info[rand.Next(0, info.Count)];
            }
            var centroids = ClustersGenerator.Clusterization(info, centers);


            // Writing result to file

            var path = ClustersGenerator.ClusterPath;
            using (StreamWriter file = new StreamWriter(path))
            {
                for (int i = 0; i < centers.Length; i++)
                {
                    file.WriteLine(i);
                    file.WriteLine(centroids[i].Amplitude);
                    file.WriteLine(centroids[i].Frequency);
                    file.WriteLine(centroids[i].Variance);
                    file.WriteLine();
                }
            }
            Process.Start(path);

            img = ImageHelper.LoadImageAsInt(TestResources._1);
            ShowClusters(img, centroids, W);
            img = ImageHelper.LoadImageAsInt(TestResources._2);
            ShowClusters(img, centroids, W);
            img = ImageHelper.LoadImageAsInt(TestResources._3);
            ShowClusters(img, centroids, W);
            img = ImageHelper.LoadImageAsInt(TestResources._4);
            ShowClusters(img, centroids, W);
            img = ImageHelper.LoadImageAsInt(TestResources._5);
            ShowClusters(img, centroids, W);
        }
    }
}
