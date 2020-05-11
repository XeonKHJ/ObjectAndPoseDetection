using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using Windows.Storage.Pickers;
using ObjectAndPoseDetection.Detector;
using Windows.Storage;
using System.Numerics.Tensors;
using ObjectAndPoseDetection.Detector.YoloParser;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Media;
using System.Drawing;
using Windows.UI.Xaml.Media.Imaging;
using Windows.Media.Transcoding;
using System.Threading.Tasks;
using Microsoft.Graphics.Canvas;
using Windows.Storage.Streams;
using Microsoft.Graphics.Canvas.Brushes;
using System.Numerics;
using Windows.Graphics.Display;

// https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x804 上介绍了“空白页”项模板

namespace ObjectAndPoseDetection.UWP
{
    /// <summary>
    /// 可用于自身或导航至 Frame 内部的空白页。
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            this.InitializeComponent();
            LoadModel();
        }

        private async void LoadModel()
        {
            var onnxFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/SingelObjectApeModelV8.onnx"));
            model = await MultiObjectDetectionModelv8Model.CreateFromStreamAsync(onnxFile);
        }
        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FolderPicker();
            picker.SuggestedStartLocation = PickerLocationId.Desktop;
            picker.FileTypeFilter.Add(".jpg");
            var chosenFolder = await picker.PickSingleFolderAsync();

            if(chosenFolder != null)
            {
                DetectObjectPose(chosenFolder);
            }
        }
        private MultiObjectDetectionModelv8Model model;
        private async void DetectObjectPose(StorageFolder folder)
        {
            var imageFiles = await folder.GetFilesAsync();
            var file = imageFiles[0];

            SoftwareBitmap softwareBitmap;
            var transform = new BitmapTransform() { ScaledWidth = 416, ScaledHeight = 416, InterpolationMode = BitmapInterpolationMode.Fant};
            TensorFloat imageTensor;
            using (var stream = await file.OpenAsync(FileAccessMode.Read))
            {
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);
                imageTensor = await ConvertImageToTensorAsync(decoder);


                var input = new MultiObjectDetectionModelv8Input() { image = imageTensor };

                var output = await model.EvaluateAsync(input);

                var shape = output.grid.Shape;
                var content = output.grid.GetAsVectorView().ToArray();
                List<float[]> abc = new List<float[]>();
                abc.Add(content);
                OutputParser outputParser = new OutputParser(abc, 1, 1);
                var boxes = outputParser.BoundingBoxes;
                DrawBoxes(stream, boxes);

                //using(var session = new CanvasDrawingSesson())
            }
        }

        public async Task<TensorFloat> ConvertImageToTensorAsync(BitmapDecoder imageDecoder)
        {
            var transform = new BitmapTransform() { ScaledWidth = 416, ScaledHeight = 416, InterpolationMode = BitmapInterpolationMode.Fant };
            var pixelData = await imageDecoder.GetPixelDataAsync(BitmapPixelFormat.Rgba8, BitmapAlphaMode.Ignore, transform, ExifOrientationMode.IgnoreExifOrientation, ColorManagementMode.DoNotColorManage);
            var pixelsWithAlpha = pixelData.DetachPixelData();
            List<float> pixels = new List<float>();
            List<float> Reds = new List<float>();
            List<float> Greens = new List<float>();
            List<float> Blues = new List<float>();
            for(int i  = 0; i < pixelsWithAlpha.Length; ++i)
            {
                switch (i % 4)
                {
                    case 0:
                        Reds.Add((float)pixelsWithAlpha[i] / 255);
                        break;
                    case 1:
                        Greens.Add((float)pixelsWithAlpha[i] / 255);
                        break;
                    case 2:
                        Blues.Add((float)pixelsWithAlpha[i] / 255);
                        break;
                }
            }
            List<float> sortedPixels = new List<float>();
            sortedPixels.AddRange(Reds);
            sortedPixels.AddRange(Greens);
            sortedPixels.AddRange(Blues);

            long[] dimensions = { 1, 3, 416, 416 };
            //Tensor<float> pixelTensor = new DenseTensor<float>(dimensions);
            var inputTesnor = TensorFloat.CreateFromShapeArrayAndDataArray(dimensions, sortedPixels.ToArray());


            return inputTesnor;
        }

        public async void DrawBoxes(IRandomAccessStream imageStream, List<CubicBoundingBox> boxes)
        {
            var device = CanvasDevice.GetSharedDevice();

            var image = await CanvasBitmap.LoadAsync(device,imageStream);

            var offscreen = new CanvasRenderTarget(device, (float)image.Bounds.Width, (float)image.Bounds.Height, 96);
            CanvasSolidColorBrush brush = new CanvasSolidColorBrush(device, Windows.UI.Color.FromArgb(255, 255, 0, 0));
            using(var ds = offscreen.CreateDrawingSession())
            {
                ds.DrawImage(image);
                foreach (var box in boxes)
                {
                    var points = (from p in box.ControlPoint
                                 select new Vector2((float)(p.X * image.Bounds.Width), (float)(p.Y * image.Bounds.Height))).ToArray();
                    
                    ds.DrawLine(points[0], points[1], brush);
                    ds.DrawLine(points[0], points[4], brush);
                    ds.DrawLine(points[1], points[5], brush);
                    ds.DrawLine(points[4], points[5], brush);
                    ds.DrawLine(points[5], points[7], brush);
                    ds.DrawLine(points[1], points[3], brush);
                    ds.DrawLine(points[4], points[6], brush);
                    ds.DrawLine(points[0], points[2], brush);
                    ds.DrawLine(points[2], points[6], brush);
                    ds.DrawLine(points[2], points[3], brush);
                    ds.DrawLine(points[3], points[7], brush);
                    ds.DrawLine(points[7], points[6], brush);
                }
            }

            await Window.Current.Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, async () =>
            {
                var savepicker = new FileSavePicker();
                savepicker.FileTypeChoices.Add("png", new List<string> { ".jpg" });
                var destFile = await savepicker.PickSaveFileAsync();

                var displayInformation = DisplayInformation.GetForCurrentView();

                if(destFile != null)
                {
                    using(var s =  await destFile.OpenAsync(FileAccessMode.ReadWrite))
                    {
                        var encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.JpegEncoderId, s);
                        encoder.SetPixelData(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore,
                            (uint)offscreen.Size.Width,
                            (uint)offscreen.Size.Height,
                            displayInformation.LogicalDpi,
                            displayInformation.LogicalDpi,
                            offscreen.GetPixelBytes());

                        await encoder.FlushAsync();
                    }
                }

            });
        }
    }
}
