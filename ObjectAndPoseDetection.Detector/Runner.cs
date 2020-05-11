using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using System.Drawing.Drawing2D;
using System.Drawing;
using ObjectAndPoseDetection.Detector.DataStructures;
using ObjectAndPoseDetection.Detector.YoloParser;

namespace ObjectAndPoseDetection.Detector
{
    public class Runner
    {
        public static void Run(string[] args)
        {
            var assetsRelativePath = @"../../../../Assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            var modelFilePath = Path.Combine(assetsPath, "OnnxModel", "MultiObjectDetectionModel.onnx");
            var imagesFolder = Path.Combine(assetsPath, "images");
            var outputFolder = Path.Combine(assetsPath, "images", "output");

            MLContext mlContext = new MLContext();


            IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
            IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

            var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);
            var probabilities = modelScorer.Score(imageDataView);

            OutputParser outputParser = new OutputParser(probabilities, 13, 5);
            var boxes = outputParser.BoundingBoxes;

            DrawBoundingBox(Path.Combine(imagesFolder, "000005.jpg"), null, "fuckoff.jpg", boxes);
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Runner).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }

        private static void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<CubicBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromFile(inputImageLocation);
            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;
            
            foreach(var box in filteredBoundingBoxes)
            {
                var points = (from p in box.ControlPoint
                             select new PointF(p.X * image.Width, p.Y * image.Height)).ToList();
                using(Graphics thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;

                    Pen pen = new Pen(Color.Red, 1f);
                    SolidBrush colorBursh = new SolidBrush(Color.Red);
                    thumbnailGraphic.DrawLine(pen, points[0], points[1]);
                    thumbnailGraphic.DrawLine(pen, points[0], points[4]);
                    thumbnailGraphic.DrawLine(pen, points[1], points[5]);
                    thumbnailGraphic.DrawLine(pen, points[4], points[5]);
                    thumbnailGraphic.DrawLine(pen, points[5], points[7]);
                    thumbnailGraphic.DrawLine(pen, points[1], points[3]);
                    thumbnailGraphic.DrawLine(pen, points[4], points[6]);
                    thumbnailGraphic.DrawLine(pen, points[0], points[2]);
                    thumbnailGraphic.DrawLine(pen, points[2], points[6]);
                    thumbnailGraphic.DrawLine(pen, points[2], points[3]);
                    thumbnailGraphic.DrawLine(pen, points[3], points[7]);
                    thumbnailGraphic.DrawLine(pen, points[7], points[6]);
                }
            }

            image.Save("ab.jpg");
        }

        public static Image DrawBoundingBox(Stream inputImageStream, IList<CubicBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromStream(inputImageStream);
            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;

            foreach (var box in filteredBoundingBoxes)
            {
                var points = (from p in box.ControlPoint
                              select new PointF(p.X * image.Width, p.Y * image.Height)).ToList();
                using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;

                    Pen pen = new Pen(Color.Red, 1f);
                    SolidBrush colorBursh = new SolidBrush(Color.Red);
                    thumbnailGraphic.DrawLine(pen, points[0], points[1]);
                    thumbnailGraphic.DrawLine(pen, points[0], points[4]);
                    thumbnailGraphic.DrawLine(pen, points[1], points[5]);
                    thumbnailGraphic.DrawLine(pen, points[4], points[5]);
                    thumbnailGraphic.DrawLine(pen, points[5], points[7]);
                    thumbnailGraphic.DrawLine(pen, points[1], points[3]);
                    thumbnailGraphic.DrawLine(pen, points[4], points[6]);
                    thumbnailGraphic.DrawLine(pen, points[0], points[2]);
                    thumbnailGraphic.DrawLine(pen, points[2], points[6]);
                    thumbnailGraphic.DrawLine(pen, points[2], points[3]);
                    thumbnailGraphic.DrawLine(pen, points[3], points[7]);
                    thumbnailGraphic.DrawLine(pen, points[7], points[6]);
                }
            }

            return image;
        }
    }
}
