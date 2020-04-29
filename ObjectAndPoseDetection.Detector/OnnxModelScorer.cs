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

namespace ObjectAndPoseDetection.Detector
{
    //public class InputImageFloatDataView : IDataView
    //{
    //    private IDataView _oldView;
    //    private MLContext mLContext;
    //    public InputImageFloatDataView(MLContext mLContext, IDataView oldView)
    //    {
    //        var builder = new DataViewSchema.Builder();
    //        this.mLContext = mLContext;

            

    //        var pixels = oldView.GetColumn<float>("image");
    //    }

    //    public bool CanShuffle
    //    {
    //        get
    //        {
    //            return _oldView.CanShuffle;
    //        }
    //    }
    //    public DataViewSchema Schema
    //    {
    //        get
    //        {
    //            return _oldView.Schema;
    //        }
    //    }

    //    public long? GetRowCount()
    //    {
    //        return _oldView.GetRowCount();
    //    }

    //    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
    //    {
    //        return _oldView.GetRowCursor(columnsNeeded);
    //    }

    //    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
    //    {
    //        return _oldView.GetRowCursorSet(columnsNeeded, n, rand);
    //    }
    //}

    //public class ScaleWithinOneTransformer : ITransformer
    //{
    //    public bool IsRowToRowMapper => throw new NotImplementedException();

    //    private MLContext mLContext;
    //    public ScaleWithinOneTransformer(MLContext mLContext)
    //    {
    //        this.mLContext = mLContext;
    //    }
    //    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    //    {
    //        return inputSchema;
    //    }

    //    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    //    {
    //        throw new NotImplementedException();
    //    }

    //    public void Save(ModelSaveContext ctx)
    //    {
    //        System.Diagnostics.Debug.WriteLine("Do nothing");
    //    }

    //    public IDataView Transform(IDataView input)
    //    {

    //        IEnumerable<VBuffer<float>> pixels = null;

    //        var a = input.Preview().ColumnView.Last().Values;
    //        pixels = input.GetColumn<VBuffer<float>>("image");

    //        var builder = new DataViewSchema.Builder();

    //        if (pixels == null)
    //        {
    //            throw new Exception();
    //        }

    //        var imagePixels = (from p in pixels
    //                           select new ImagePixels() { Pixels = p }).AsEnumerable();

    //        var output = mLContext.Data.LoadFromEnumerable(imagePixels);
    //        //output.Schema[output.Schema.Count-1] = 
    //        //var outputPreview = output.Preview();

    //        return output;
    //    }
    //}
    //public class ScaleWithOneEstimator : IEstimator<ScaleWithinOneTransformer>
    //{
    //    private MLContext mLContext;
    //    public ScaleWithOneEstimator(MLContext mLContext)
    //    {
    //        this.mLContext = mLContext;
    //    }
    //    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    //    {
    //        //throw new NotImplementedException();
    //        return inputSchema;
    //    }

    //    ScaleWithinOneTransformer IEstimator<ScaleWithinOneTransformer>.Fit(IDataView input)
    //    {
    //        DataTable dataTable = new DataTable();
    //        return new ScaleWithinOneTransformer(mLContext);
    //    }
    //}

    //public static class NormlizationExtender
    //{
    //    public static NormalizingEstimator DivideByANumber(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null, string labelColumnName = "Label", long maximumExampleCount = 1000000000, bool fixZero = true, int maximumBinCount = 1024, int mininimumExamplesPerBin = 10)
    //    {
    //        return new ScaleWithOneEstimator(null);
    //    }
    //}
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
                float[] unifiedEncoding = new float[4];

                var indices = input.Pixels.Select(x => x / 255);

                output.Pixels = indices.ToArray();
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

            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);

            return probabilities;
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(modelLocation);
            return PredictDataUsingModel(data, model);
        }
    }
}
