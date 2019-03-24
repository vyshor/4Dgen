using System;
using System.Runtime.InteropServices;

namespace VisualRecognition
{
    public class NativeConstants
    {
        /// IUTask_Classification -> 1
        public const int TaskClassification = 1;

        /// IUTask_Detection -> 2
        public const int TaskDetection = 2;

        /// IUTask_Segmentation -> 4
        public const int TaskSegmentation = 4;

        /// IUClassification_FullView -> 1
        public const int ClassificationFullView = 1;

        /// IUClassification_NineView -> 2
        public const int ClassificationNineView = 2;

        public const int Ok = 0;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ClassificationOption
    {
        public void Init()
        {
            classificationView = NativeConstants.ClassificationFullView;
            cropRatio = 0.875f;
        }

        /// int
        public int classificationView;

        /// float
        public float cropRatio;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DetectionOption
    {
        public int maxNumPreFilter;
        public int maxNumPostFilter;
        public float nmsThreshold;
        
        public void Init()
        {
            maxNumPreFilter = 1024;
            maxNumPostFilter = 128;
            nmsThreshold = 0.3f;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Option
    {
        public ClassificationOption classificationOption;
        public DetectionOption detectionOption;
        public void Init()
        {
            classificationOption.Init();
            detectionOption.Init();
        }
    }

    public static class NativeMethods
    {
        const string Dll = "ImageRecognitionSDK.dll";

        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        [DllImport(Dll, SetLastError = true, BestFitMapping = true,
            CallingConvention = CallingConvention.Cdecl, EntryPoint = "IUCreateHandle")]
        public static extern int IUCreateHandle(ref IntPtr handle);

        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUReleaseHandle")]
        public static extern int IUReleaseHandle(IntPtr handle);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// tasks: int
        /// modelFolder: wchar_t*
        [DllImport(Dll, SetLastError = true, BestFitMapping = true,
            CallingConvention = CallingConvention.Cdecl, EntryPoint = "IULoadModel", CharSet = CharSet.Unicode)]
        public static extern int IULoadModel(IntPtr handle, int tasks, [MarshalAs(UnmanagedType.LPWStr)] string modelFolder);

        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// tasks: int
        /// modelFolder: wchar_t*
        [DllImport(Dll, SetLastError = true, BestFitMapping = true,
            CallingConvention = CallingConvention.Cdecl, EntryPoint = "IULoadSharedMemoryModel",
            CharSet = CharSet.Unicode)]
        public static extern int IULoadSharedMemoryModel(IntPtr handle, IntPtr handleInitedModel);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// free memory up, can load model again afterwards
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "IUUnloadModel")]
        public static extern int IUUnloadModel(IntPtr handle);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// pImage: BYTE*
        /// width: int
        /// height: int
        /// stride: int
        /// channel: int
        /// tasks: int
        /// opts: IUOption
        /// no image resizing is required before calling the API,
        /// as the image will be resized internally
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "IUDoTasks")]
        public static extern int IUDoTasks(IntPtr handle, [In] IntPtr pImage, int width, int height, int stride, int channel, int tasks, Option opts);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// pCategoryNumber: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationClassifierNumber")]
        public static extern int IUGetClassificationClassifierNumber(IntPtr handle, ref int pClassifierNumber);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// pCategoryNumber: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationCategoryNumber")]
        public static extern int IUGetClassificationCategoryNumber(IntPtr handle, IntPtr pCategoryNumber);

        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// classifierId: int
        /// pCategoryNumber: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationCategoryNumberEx")]
        public static extern int IUGetClassificationCategoryNumberEx(IntPtr handle, int classifierId,
            ref int pCategoryNumber);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// count: int
        /// pCategories: IUClassificationInfo*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationTopNResults")]
        public static extern int IUGetClassificationTopNResults(IntPtr handle, int count, IntPtr pCategories);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// classifierId: int
        /// count: int
        /// pCategories: IUClassificationInfo*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationTopNResultsEx")]
        public static extern int IUGetClassificationTopNResultsEx(IntPtr handle, int classifierId, int count,
            IntPtr pCategories);

        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// num: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationLayerNum")]
        public static extern int IUGetClassificationLayerNum(IntPtr handle, out int num);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// layerIdx: int
        /// layerName: char**
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationLayerName")]
        public static extern int IUGetClassificationLayerName(IntPtr handle, int layerIdx, [MarshalAs(UnmanagedType.LPStr)] out string layerName);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// layerName: char*
        /// channels: int*
        /// width: int*
        /// height: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationLayerDim")]
        public static extern int IUGetClassificationLayerDim(IntPtr handle,
            [In] [MarshalAs(UnmanagedType.LPStr)] string layerName, out int channels, out int width, out int height);

        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// layerName: char*
        /// bufferSize: int
        /// buffer: float*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetClassificationLayerResponse")]
        public static extern int IUGetClassificationLayerResponse(IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string layerName, int bufferSize, float[] buffer);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// pCategoryNumber: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetDetectionCategoryNumber")]
        public static extern int IUGetDetectionCategoryNumber(IntPtr handle, ref int pCategoryNumber);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// pCount: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetDetectedObjectCount")]
        public static extern int IUGetDetectedObjectCount(IntPtr handle, ref int pCount);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// count: int
        /// pObjects: IUObjectInfo*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetDetectedObjects")]
        public static extern int IUGetDetectedObjects(IntPtr handle, int count, IntPtr pObjects);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// count: int
        /// pThresholds: float*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUSetDetectionPerClassThresholds")]
        public static extern int IUSetDetectionPerClassThresholds(IntPtr handle, int count,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4, SizeParamIndex = 1)] float[] pThresholds);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// pCategoryNumber: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetSegmentationCategoryNumber")]
        public static extern int IUGetSegmentationCategoryNumber(IntPtr handle, IntPtr pCategoryNumber);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// channels: int*
        /// width: int*
        /// height: int*
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetSegmentationResponseMapSize")]
        public static extern int IUGetSegmentationResponseMapSize(IntPtr handle, IntPtr channels, IntPtr width,
            IntPtr height);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// channels: int
        /// width: int
        /// height: int
        /// pResponseMap: float*
        // channel is fastest, then width, then height
        // idx = c + w * channels + h * width * channels
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUGetSegmentationResponseMap")]
        public static extern int IUGetSegmentationResponseMap(IntPtr handle, int channels, int width, int height,
            IntPtr pResponseMap);


        /// Return Type: HRESULT->LONG->int
        /// handle: intptr_t
        /// numThreads: int
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "IUSetNumThreads")]
        public static extern int IUSetNumThreads(IntPtr handle, int numThreads);
    }
}