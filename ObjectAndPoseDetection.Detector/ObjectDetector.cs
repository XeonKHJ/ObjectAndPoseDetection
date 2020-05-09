using Microsoft.ML;
using ObjectAndPoseDetection.Detector.DataStructures;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace ObjectAndPoseDetection.Detector
{
    public class ObjectDetector
    {
        private MLContext mlContext;
        private string onnxPath;
        public ObjectDetector(string onnxPath = "../Assets/OnnxModel/MultiObjectDetectionModel.onnx")
        {
            mlContext = new MLContext();


            this.onnxPath = onnxPath;


            
        }

        public List<CubicBoundingBox> DetectFromFolder(string path)
        {
            IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(path);
            IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

            var modelScorer = new OnnxModelScorer(path, onnxPath, mlContext);
            var probabilities = modelScorer.Score(imageDataView);

            OutputParser outputParser = new OutputParser(probabilities, 13, 5);
            var boxes = outputParser.BoundingBoxes;

            return boxes;
        }
    }
}
