using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;

namespace AlgorithmVSCOME
{
    internal static class Symmetry
    {
        public static double[,] GetSymmetry(Complex[,] complexOrientationField)
        {
            var kernelSymmetry = KernelHelper.MakeComplexKernel(
                           (x, y) =>
                           y < 0
                               ? 0
                               : ComplexFilterQA.Gaussian.Gaussian2D(x, y, Constants.Sigma) * y * x * 2.0d /
                                 (x == 0 && y == 0 ? 1 : Math.Sqrt(x * x + y * y)),
                           (x, y) =>
                           y < 0
                               ? 0
                               : ComplexFilterQA.Gaussian.Gaussian2D(x, y, Constants.Sigma) * (x * x - y * y) /
                                 (x == 0 && y == 0 ? 1 : Math.Sqrt(x * x + y * y)),
                           KernelHelper.GetKernelSizeForGaussianSigma(Constants.Sigma));

            Complex[,] complexSymmetry = ConvolutionHelper.ComplexConvolve(complexOrientationField, kernelSymmetry);

            return complexSymmetry.Select2D(x => x.Magnitude);
        }
    }
}
