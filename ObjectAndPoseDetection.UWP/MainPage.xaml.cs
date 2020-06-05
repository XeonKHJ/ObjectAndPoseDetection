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
using Windows.UI.Xaml.Media.Imaging;
using Windows.Media.Transcoding;
using System.Threading.Tasks;
using Microsoft.Graphics.Canvas;
using Windows.Storage.Streams;
using Microsoft.Graphics.Canvas.Brushes;
using System.Numerics;
using Windows.Graphics.Display;
using Windows.UI;
using Windows.UI.Popups;
using Windows.Media.Playback;
using Windows.Media.Core;
using Microsoft.Graphics.Canvas.UI.Xaml;
using System.Drawing;
using Microsoft.Graphics.Canvas.Effects;

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
            canvasDevice = CanvasDevice.GetSharedDevice();
            LoadModel();
        }

        private async void LoadModel()
        {
            var onnxFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/SingelObjectApeModelV8.onnx"));
            model = await SingelObjectApeModelV8Model.CreateFromStreamAsync(onnxFile);
        }
        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker();
            picker.SuggestedStartLocation = PickerLocationId.Desktop;
            picker.FileTypeFilter.Add(".jpg");
            picker.FileTypeFilter.Add(".mp4");
            var chosenFolder = await picker.PickSingleFileAsync();

            try
            {
                if (chosenFolder != null)
                {
                    switch (chosenFolder.FileType)
                    {
                        case ".jpg":
                            DetectObjectPoseFromPicFile(chosenFolder);
                            break;
                        case ".mp4":
                            DetectObectPoseRealtime(chosenFolder);
                            break;
                        default:
                            throw new InvalidOperationException("程序不支持打开类型为" + chosenFolder.FileType + "的文件");

                    }
                }
            }
            catch (InvalidOperationException exception)
            {
                var dialog = new MessageDialog(exception.Message);
                await dialog.ShowAsync();
            }
        }

        MediaPlayer _mediaPlayer;
        private void DetectObectPoseRealtime(StorageFile chosenFolder)
        {
            MediaPlayer mediaPlayer = new MediaPlayer()
            {
                IsVideoFrameServerEnabled = true,
                AutoPlay = true
            };
            _mediaPlayer = mediaPlayer;
            mediaPlayer.VideoFrameAvailable += MediaPlayer_VideoFrameAvailableAsync;
            mediaPlayer.Source = MediaSource.CreateFromStorageFile(chosenFolder);
            mediaPlayer.PlaybackSession.PlaybackStateChanged += PlaybackSession_PlaybackStateChanged;
        }

        private void PlaybackSession_PlaybackStateChanged(MediaPlaybackSession sender, object args)
        {
            System.Diagnostics.Debug.WriteLine(string.Format("PlayStatus{0}", sender.PlaybackState));
        }

        SoftwareBitmap frameServerDest;
        CanvasImageSource canvasImageSource;
        bool isRenderringFinished = true;
        CanvasDevice canvasDevice;
        private async void MediaPlayer_VideoFrameAvailableAsync(MediaPlayer sender, object args)
        {
            //if (!isRenderringFinished)
            //{
            //    return;
            //}
            isRenderringFinished = false;
            //CanvasDevice canvasDevice = CanvasDevice.GetSharedDevice();

            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, async () =>
            {
                if (frameServerDest == null)
                {
                    frameServerDest = new SoftwareBitmap(BitmapPixelFormat.Bgra8, (int)sender.PlaybackSession.NaturalVideoWidth, (int)sender.PlaybackSession.NaturalVideoHeight, BitmapAlphaMode.Ignore);
                }
                canvasImageSource = new CanvasImageSource(canvasDevice, (int)sender.PlaybackSession.NaturalVideoWidth, (int)sender.PlaybackSession.NaturalVideoWidth, DisplayInformation.GetForCurrentView().LogicalDpi);//96); 
                OutputImage.Source = canvasImageSource;

                try
                {
                    using (CanvasBitmap inputBitmap = CanvasBitmap.CreateFromSoftwareBitmap(canvasDevice, frameServerDest))
                    //using (CanvasDrawingSession ds = canvasImageSource.CreateDrawingSession(Colors.Black))
                    {
                        var canvasRenderTarget = new CanvasRenderTarget(canvasDevice, 416, 416, 96);
                        sender.CopyFrameToVideoSurface(inputBitmap);

                        using (var cds = canvasRenderTarget.CreateDrawingSession())
                        {
                            cds.DrawImage(inputBitmap, canvasRenderTarget.Bounds);
                        }

                        var pixelBytes = canvasRenderTarget.GetPixelBytes();

                        var boxes = await DetectObjectPoseFromImagePixelsAsync(pixelBytes);

                        DrawBoxes(inputBitmap, boxes, canvasImageSource);
                        //ds.DrawImage(inputBitmap);
                    }
                }
                catch (Exception exception)
                {
                    System.Diagnostics.Debug.WriteLine(exception.Message);
                }
            });
            isRenderringFinished = true;
        }

        private SingelObjectApeModelV8Model model;
        private async void DetectObjectPoseFromPicFile(StorageFile inputFile)
        {
            var file = inputFile;

            //var transform = new BitmapTransform() { ScaledWidth = 416, ScaledHeight = 416, InterpolationMode = BitmapInterpolationMode.Fant };
            TensorFloat imageTensor;

            using (var stream = await file.OpenAsync(FileAccessMode.Read))
            {
                BitmapSource bitmapSource = new BitmapImage();
                bitmapSource.SetSource(stream);
                InputImage.Source = bitmapSource;

                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);
                imageTensor = await ConvertImageToTensorAsync(decoder);

                var input = new SingelObjectApeModelV8Input() { image = imageTensor };

                var output = await model.EvaluateAsync(input);

                var shape = output.grid.Shape;
                var content = output.grid.GetAsVectorView().ToArray();
                List<float[]> rawOutput = new List<float[]>
                {
                    content
                };
                OutputParser outputParser = new OutputParser(rawOutput, 1, 1, 0.5f);
                var boxes = outputParser.BoundingBoxes;
                DrawBoxes(stream, boxes);
            }
        }

        public async Task<List<CubicBoundingBox>> DetectObjectPoseFromImagePixelsAsync(byte[] imagePixels)
        {
            List<CubicBoundingBox> boxes = new List<CubicBoundingBox>();
            using (var imageTensor = ConvertPixelsByteToTensor(imagePixels, BitmapPixelFormat.Bgra8))
            //using (var input = new SingelObjectApeModelV8Input() { image = imageTensor })
            //using (var output = await model.EvaluateAsync(input))
            {
                //var shape = output.grid.Shape;
                //var content = output.grid.GetAsVectorView().ToArray();
                //List<float[]> abc = new List<float[]>();
                //abc.Add(content);

                //using (OutputParser outputParser = new OutputParser(abc, 1, 1, 0.5f))
                //{
                //    foreach (var box in outputParser.BoundingBoxes)
                //    {
                //        var newBox = new CubicBoundingBox()
                //        {
                //            Confidence = box.Confidence,
                //            Identity = box.Identity
                //        };
                //        foreach (var point in box.ControlPoint)
                //        {
                //            newBox.ControlPoint.Append(point);
                //        }
                //        boxes.Add(box);
                //    }
                //}
            }



            return boxes;
        }

        public async Task<TensorFloat> ConvertImageToTensorAsync(BitmapDecoder imageDecoder)
        {
            var transform = new BitmapTransform() { ScaledWidth = 416, ScaledHeight = 416, InterpolationMode = BitmapInterpolationMode.Fant };
            var pixelData = await imageDecoder.GetPixelDataAsync(BitmapPixelFormat.Rgba8, BitmapAlphaMode.Ignore, transform, ExifOrientationMode.IgnoreExifOrientation, ColorManagementMode.DoNotColorManage);
            var pixelsWithAlpha = pixelData.DetachPixelData();

            var inputTensor = ConvertPixelsByteToTensor(pixelsWithAlpha, BitmapPixelFormat.Rgba8);

            return inputTensor;
        }

        public TensorFloat ConvertPixelsByteToTensor(byte[] imagePixels, BitmapPixelFormat bitmapPixelFormat)
        {
            var pixelsWithAlpha = imagePixels;
            List<float> reds = new List<float>();
            List<float> greens = new List<float>();
            List<float> blues = new List<float>();
            for (int i = 0; i < pixelsWithAlpha.Length; ++i)
            {
                switch (i % 4)
                {
                    case 0:
                        if(bitmapPixelFormat == BitmapPixelFormat.Rgba8)
                        {
                            reds.Add((float)pixelsWithAlpha[i] / 255);
                        }
                        else if(bitmapPixelFormat == BitmapPixelFormat.Bgra8)
                        {
                            blues.Add((float)pixelsWithAlpha[i] / 255);
                        }
                        break;
                    case 1:
                        greens.Add((float)pixelsWithAlpha[i] / 255);
                        break;
                    case 2:
                        if (bitmapPixelFormat == BitmapPixelFormat.Rgba8)
                        {
                            blues.Add((float)pixelsWithAlpha[i] / 255);
                        }
                        else if (bitmapPixelFormat == BitmapPixelFormat.Bgra8)
                        {
                            reds.Add((float)pixelsWithAlpha[i] / 255);
                        }
                        break;
                }
            }
            List<float> sortedPixels = new List<float>();
            sortedPixels.AddRange(reds);
            sortedPixels.AddRange(greens);
            sortedPixels.AddRange(blues);

            long[] dimensions = { 1, 3, 416, 416 };
            //Tensor<float> pixelTensor = new DenseTensor<float>(dimensions);
            var inputTesnor = TensorFloat.CreateFromShapeArrayAndDataArray(dimensions, sortedPixels.ToArray());
            //sortedPixels = null;
            sortedPixels.Clear();
            sortedPixels.TrimExcess();
            return inputTesnor;
        }


        public async void DrawBoxes(IRandomAccessStream imageStream, List<CubicBoundingBox> boxes)
        {
            var device = canvasDevice;

            var image = await CanvasBitmap.LoadAsync(device, imageStream);

            var offscreen = new CanvasRenderTarget(device, (float)image.Bounds.Width, (float)image.Bounds.Height, 96);
            //CanvasSolidColorBrush brush = new CanvasSolidColorBrush(device, Windows.UI.Color.FromArgb(255, 255, 0, 0));
            using (var ds = offscreen.CreateDrawingSession())
            {
                ds.DrawImage(image);
                foreach (var box in boxes)
                {
                    CanvasSolidColorBrush brush = new CanvasSolidColorBrush(device, Colors.Red);

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

            BitmapSource bitmapImageSouce;
            using (var stream = new InMemoryRandomAccessStream())
            {
                await offscreen.SaveAsync(stream, CanvasBitmapFileFormat.Jpeg);
                bitmapImageSouce = new BitmapImage();

                stream.Seek(0);
                await bitmapImageSouce.SetSourceAsync(stream);

            }

            await Window.Current.Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, () =>
             {
                 //var savepicker = new FileSavePicker();
                 //savepicker.FileTypeChoices.Add("JPEG", new List<string> { ".jpg" });
                 //var destFile = await savepicker.PickSaveFileAsync();

                 //var displayInformation = DisplayInformation.GetForCurrentView();

                 //if(destFile != null)
                 //{
                 //    using(var s =  await destFile.OpenAsync(FileAccessMode.ReadWrite))
                 //    {
                 //var encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.JpegEncoderId, s);
                 //encoder.SetPixelData(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore,
                 //    (uint)offscreen.Size.Width,
                 //    (uint)offscreen.Size.Height,
                 //    displayInformation.LogicalDpi,
                 //    displayInformation.LogicalDpi,
                 //    offscreen.GetPixelBytes());

                 //        await encoder.FlushAsync();
                 //    }
                 //}

                 OutputImage.Source = bitmapImageSouce;
             });
        }

        public void DrawBoxes(CanvasBitmap bitmap, List<CubicBoundingBox> boxes, CanvasImageSource canvasImageSource)
        {
            var device = canvasDevice;
            //CanvasSolidColorBrush brush = new CanvasSolidColorBrush(device, Windows.UI.Color.FromArgb(255, 255, 0, 0));
            using (var ds = canvasImageSource.CreateDrawingSession(Colors.Black))
            {
                ds.DrawImage(bitmap);
                foreach (var box in boxes)
                {
                    CanvasSolidColorBrush brush = new CanvasSolidColorBrush(device, Colors.Red);

                    var points = (from p in box.ControlPoint
                                  select new Vector2((float)(p.X * bitmap.Bounds.Width), (float)(p.Y * bitmap.Bounds.Height))).ToArray();

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
        }

        private void PlayButton_Click(object sender, RoutedEventArgs e)
        {
            if(_mediaPlayer != null)
            {
                switch(_mediaPlayer.PlaybackSession.PlaybackState)
                {
                    case MediaPlaybackState.Paused:
                        _mediaPlayer.Play();
                        break;
                    case MediaPlaybackState.Playing:
                        _mediaPlayer.Pause();
                        break;
                }
            }
        }
    }
}
