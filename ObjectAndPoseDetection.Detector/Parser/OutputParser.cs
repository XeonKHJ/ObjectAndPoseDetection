using System;
using System.Collections.Generic;
using System.ComponentModel.Design;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Google.Protobuf.Reflection;
using MathNet.Numerics;
using MathNet.Numerics.RootFinding;
using Microsoft.ML.Trainers;
using Newtonsoft.Json;

public class OutputParser
{
    private List<float[]> _originalOutputs = new List<float[]>();
    private Dictionary<Point, ParsedOutput> Results;
    public OutputParser(IEnumerable<float[]> nnOutputs)
    {
        foreach(var detectionResult in nnOutputs)
        {
            var orignalOutput = new float[detectionResult.Length];
            detectionResult.CopyTo(orignalOutput, 0);
            _originalOutputs.Add(orignalOutput);

        }

        var batchOutpus = ClassifyOutputBySegements(_originalOutputs);

        //选择有物体的方块
        //获取信度
        var class_confs = GetIdentification(batchOutpus.First());

        //计算信度
        var det_conf = GetConfidence(batchOutpus.First());

        var confs = (from c in class_confs
                     select new KeyValuePair<Point, float>(c.Key, det_conf[c.Key] * class_confs[c.Key].First())).ToArray();

        Point maxConfCordniate;
        float maxConf = float.MinValue;
        foreach(var confPair in confs)
        {
            var conf = confPair.Value;
            var confCordinate = confPair.Key;
            if(conf > maxConf)
            {
                maxConf = conf;
                maxConfCordniate = confCordinate;
            }
        }

        //获取这些方块的边框
        var boundingBoxes = ExtractBoundingBox(batchOutpus.First());

    }

    private int classOut = 1;
    private int anchorCount = 1;
    private Dictionary<Point, float> GetConfidence(Dictionary<Point, float[]> dictionary)
    {
        Dictionary<Point, float> confs = new Dictionary<Point, float>();
        foreach(var pair in dictionary)
        {
            confs.Add(pair.Key, Sigmoid(pair.Value[2 * 9]));
        }
        return confs;
    }
    private Dictionary<Point, float[]> GetIdentification(Dictionary<Point, float[]> dictionary)
    {
        Dictionary<Point, float[]> results = new Dictionary<Point, float[]>();
        foreach(var pair in dictionary)
        {
            float[] probilities = pair.Value.Skip(2 * 9 + 1).Take(classOut).ToArray();
            results.Add(pair.Key ,Softmax(probilities));
        }
        return results;
    }
    
    private float[] Softmax(float[] values)
    {
        var maxVal = values.Max();
        var exp = values.Select(v => Math.Exp(v - maxVal));
        var sumExp = exp.Sum();

        return exp.Select(v => (float)(v / sumExp)).ToArray();
    }

    private List<CubicBoundingBox> cubicBoudingBoxes;

    public List<CubicBoundingBox> CubicBoudingBox { get; }
    private List<Dictionary<Point, float[]>> ClassifyOutputBySegements(IEnumerable<float[]> probabilities)
    {
        //1、先分割成像素
        int segementsPerRow = 13;
        int segementsPerColumn = 13;
        int segements = segementsPerRow * segementsPerColumn;
        //int anchors = 1;
        List<Dictionary<Point, float[]>> batchOutputs = new List<Dictionary<Point, float[]>>();

        foreach(var probability in probabilities)
        {
            int outputVectorLengthPerSegement = probability.Length / segements;
            Dictionary<Point, float[]> outputs = new Dictionary<Point, float[]>();
            for (int i = 0; i < probability.Length; ++i)
            {
                //确定是第几个数
                int segementNo = i % segements;

                //确定是在第几个分片中
                int outputVectorPerSegementOrder = i / segements;

                //确定是第几行
                int rowNo = segementNo / segementsPerRow;

                //确定是第几列
                int columnNo = segementNo % segementsPerRow;

                var cordinate = new Point(columnNo, rowNo);

                if (!outputs.ContainsKey(cordinate))
                {
                    outputs[cordinate] = new float[outputVectorLengthPerSegement];
                }

                outputs[cordinate][outputVectorPerSegementOrder] = probability[i];
            }

            batchOutputs.Add(outputs);
        }

        return batchOutputs;
    }

    /// <summary>
    /// 暂时只能先处理锚为1的。
    /// </summary>
    /// <param name="outputs"></param>
    /// <returns></returns>
    private Dictionary<Point, List<CubicBoundingBox>> ExtractBoundingBox(Dictionary<Point, float[]> outputs)
    {
       
        Dictionary<Point, List<CubicBoundingBox>> boxesInCells = new Dictionary<Point, List<CubicBoundingBox>>();
        int outputCountPerAnchor = 9 * 2 + classOut + 1;
        foreach (var output in outputs)
        {
            List<CubicBoundingBox> boxes = new List<CubicBoundingBox>();
            for (int i = 0; i < anchorCount; ++i)
            {
                int anchorOffset = i * outputCountPerAnchor;
                var outputValue = output.Value;

                //先算信心函数
                var confidentValue = Sigmoid(outputValue[2 * 9 + anchorOffset]);

                PointF centerPoint = new PointF(Sigmoid(outputValue[0 + anchorOffset]) + output.Key.X, Sigmoid(outputValue[1 + anchorOffset]) + output.Key.Y);
                CubicBoundingBox cubicBoundingBoxes = new CubicBoundingBox();

                cubicBoundingBoxes.ControlPoint[8] = centerPoint;

                for (int j = 2; j < 2 + 8 * 2; j += 2)
                {
                    PointF point = new PointF(outputValue[j + anchorOffset] + output.Key.X, outputValue[j + 1 + anchorOffset] + output.Key.Y);
                    cubicBoundingBoxes.ControlPoint[j / 2 - 1] = point;
                }

                boxes.Add(cubicBoundingBoxes);
            }
            boxesInCells.Add(output.Key, boxes);
        }
        return boxesInCells;
    }

    private static float Sigmoid(double value)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-value));
    }
}