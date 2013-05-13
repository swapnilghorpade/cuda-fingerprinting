using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;
using FingerprintLib;

namespace ModelBasedAlgorithm
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Write path, please.");

            string path = Console.ReadLine();

            double[,] imgBytes = ImageHelper.LoadImage(path);
            imgBytes = ImageEnhancementHelper.EnhanceImage(imgBytes);

            var singularPoints = ModelBasedAlgorithm.FindSingularPoints(imgBytes,
                PoincareIndexMethod.FindSingularPoins(imgBytes));
        }
    }
}
