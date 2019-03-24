using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace VisualRecognition
{
    public sealed class VisualModel
    {
        public class LayerInfo
        {
            public readonly string Name;
            public readonly int Channels;
            public readonly int Width;
            public readonly int Height;

            public LayerInfo(string name, int channels, int width, int height)
            {
                Name = name;
                Channels = channels;
                Width = width;
                Height = height;
            }
        }

        //http://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
        const int OrientationCode = 0x0112;

        const int Task = NativeConstants.TaskClassification;

        static readonly Dictionary<byte, RotateFlipType> OrientationToRotateFlipType = new Dictionary<byte, RotateFlipType>
        {
            {2, RotateFlipType.RotateNoneFlipX},
            {3, RotateFlipType.Rotate180FlipNone},
            {4, RotateFlipType.Rotate180FlipX},
            {5, RotateFlipType.Rotate90FlipX},
            {6, RotateFlipType.Rotate90FlipNone},
            {7, RotateFlipType.Rotate270FlipX},
            {8, RotateFlipType.Rotate270FlipNone}
        };

        static Option _option;
        
        readonly IntPtr _handle;

        static VisualModel()
        {
            _option = default(Option);
            _option.Init();
        }

        VisualModel()
        {
            Contracts.Check<InvalidOperationException>(
                NativeMethods.IUCreateHandle(ref _handle) == NativeConstants.Ok,
                "Error creating ImageRecognitionSDK handle!");
        }

        public VisualModel(string configFile, int threads = 1) : this()
        {
            Contracts.Check(configFile != null && File.Exists(configFile));
            Contracts.Check<InvalidOperationException>(
                NativeMethods.IULoadModel(_handle, Task, configFile) == NativeConstants.Ok,
                "Error loading ImageRecognitionSDK model file {0}", configFile);
            SetNumThreads(threads);
        }

        /// <summary>
        /// Create a new model from another model, using shared-memory for model parameters
        /// </summary>
        public VisualModel(VisualModel other) : this()
        {
            Contracts.Check<InvalidOperationException>(
                NativeMethods.IULoadSharedMemoryModel(_handle, other._handle) == NativeConstants.Ok,
                "Error copying model");
        }

        public void SetNumThreads(int threads)
        {
            if (threads == -1)
                threads = Environment.ProcessorCount;
            Contracts.Check(threads > 0);
            NativeMethods.IUSetNumThreads(_handle, threads);
        }

        public int GetLayerDim(string layerName)
        {
            int channels, width, height;
            NativeMethods.IUGetClassificationLayerDim(_handle, layerName, out channels, out width, out height);
            return channels * width * height;
        }

        public LayerInfo[] GetLayerInfos()
        {
            int layerCount;
            NativeMethods.IUGetClassificationLayerNum(_handle, out layerCount);
            var layerInfos = new LayerInfo[layerCount];
            for (int il = 0; il < layerCount; il++)
            {
                string layerName;
                NativeMethods.IUGetClassificationLayerName(_handle, il, out layerName);
                int channels, width, height;
                NativeMethods.IUGetClassificationLayerDim(_handle, layerName, out channels, out width, out height);
                layerInfos[il] = new LayerInfo(layerName, channels, width, height);
            }

            return layerInfos;
        }
        
        public void Compute(Bitmap image)
        {
            //Image pre-processing
            if (Array.IndexOf(image.PropertyIdList, OrientationCode) > -1)
            {
                var orientation = image.GetPropertyItem(OrientationCode).Value[0];
                if (1 < orientation && orientation <= 8)
                {
                    image.RotateFlip(OrientationToRotateFlipType[orientation]);
                    image.RemovePropertyItem(OrientationCode);
                }
            }

            BitmapData imageData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            const int channel = 3;
            Contracts.Check(
                NativeMethods.IUDoTasks(_handle, imageData.Scan0, image.Width, image.Height, imageData.Stride, channel, Task, _option)
                == NativeConstants.Ok);
            image.UnlockBits(imageData);
        }

        public void GetLayerValues(string layerName, float[] output)
        {
            NativeMethods.IUGetClassificationLayerResponse(_handle, layerName, output.Length, output);
        }
    }
}