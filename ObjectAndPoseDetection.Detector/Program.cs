using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using ObjectAndPoseDetection.Detector.DataStructures;
using ObjectAndPoseDetection.Detector.YoloParser;

namespace ObjectAndPoseDetection.Detector
{
    class Program
    {
        static void Main(string[] args)
        {
            var assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            var modelFilePath = Path.Combine(assetsPath, "Model", "model.onnx");
            var imagesFolder = Path.Combine(assetsPath, "images");
            var outputFolder = Path.Combine(assetsPath, "images", "output");

            MLContext mlContext = new MLContext();


            IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
            IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

                var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);
                var probabilities = modelScorer.Score(imageDataView);

                YoloOutoutParser parser = new YoloOutoutParser();

                var abc = probabilities.Count();

                var boundingBoxes = probabilities.Select(probability => parser.ParseOutputs(probability))
                                                 .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5f)).ToList();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}
