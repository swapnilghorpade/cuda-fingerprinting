using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC
{
    public class Img3DHelper
    {
        /// <summary>
        /// Create and save as image layers of 3D-array with the layer number been the third index
        /// Type of files - .png
        /// </summary>
        /// <param name="value"></param>
        public static void Save3DAs2D(int[, ,] value, string path)
        {
            int[,] intLayer = new int[value.GetLength(0), value.GetLength(1)];
            for (int i = 0; i < value.GetLength(2); i++)
            {
                intLayer = intLayer.Select2D((a, x, y) => (value[x, y, i]));
                Bitmap layer = ImageHelper.SaveArrayToBitmap(intLayer);
                layer.Save(path+"_" + i + ".png", ImageFormat.Png);
            }
        }
    }
}
