using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectAndPoseDetection.Detector.DataStructures;
using ObjectAndPoseDetection.Detector.YoloParser;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms;
using System.Data;
using System.Numerics;
using System.CodeDom;
using System.Drawing;
using System.Threading.Tasks;

namespace ObjectAndPoseDetection.Detector
{
    public class OnnxModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mlContext;

        private IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();
        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.mlContext = mlContext;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 416;
            public const int imageWidth = 416;
        }

        public struct TinyYoloModelSettings
        {
            public const string ModelInput = "image";
            public const string ModelOutput = "grid";
        }

        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
            Action<ImagePixels, ImagePixels> mapping = (input, output) =>
            {
                output.Pixels = input.Pixels.Select(x => x / 255).ToArray();
            };

            var outputSchemaDefinition = SchemaDefinition.Create(typeof(ImagePixels));
            outputSchemaDefinition["image"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 3, 416, 416);

            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                                    .Append(mlContext.Transforms.ResizeImages("image", ImageNetSettings.imageWidth, ImageNetSettings.imageHeight, "image"))
                                    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image", colorsToExtract: ImagePixelExtractingEstimator.ColorBits.Rgb))
                                    .Append(mlContext.Transforms.CustomMapping(mapping, null, outputSchemaDefinition:outputSchemaDefinition))
                                    .Append(mlContext.Transforms.ApplyOnnxModel(modelFile:modelLocation, outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput}, inputColumnNames: new[] { "image" }));

            //var preview = pipeline.Preview(data);
            var model = pipeline.Fit(data);
            
            return model;
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            Console.WriteLine($"Images location: {imagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");

            var preview = model.Preview(testData);

            IDataView scoredData = model.Transform(testData);

            List<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput).ToList();

            

            return probabilities.AsEnumerable();
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(modelLocation);
            return PredictDataUsingModel(data, model);
        }
    }
}
