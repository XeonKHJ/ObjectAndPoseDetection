using System;
using System.Collections.Generic;
using System.ComponentModel.Design;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices.ComTypes;
using System.Threading.Tasks;
using System.Xml.Schema;
using Google.Protobuf.Reflection;
using Microsoft.ML.Trainers;
using Newtonsoft.Json;

public class OutputParser
{
    Size SegementsSize { set; get; } = new Size(13, 13);
    private float[] _originalOutputs;
    private Dictionary<Point, ParsedOutput> Results;

    public OutputParser(IEnumerable<float[]> nnOutputs, int classCount = 1, int anchorCount = 1, float confThreshold = 0.05f)
    {
        //将输出结果复制
        foreach (var detectionResult in nnOutputs)
        {
            var orignalOutput = new float[detectionResult.Length];
            detectionResult.CopyTo(orignalOutput, 0);
            _originalOutputs = orignalOutput;
        }

        this.anchorCount = anchorCount;
        this.classOut = classCount;

        var parsedOutput = ClassifyOutputBySegements(_originalOutputs);


        var classProbabilities = GetClassProbabilities(parsedOutput);

        //计算每个锚中可能性最大的类
        var maxClassConfIndexs = new Dictionary<Point, List<int>>();
        foreach (var pointAnchorPair in classProbabilities)
        {
            List<int> maxIdxs = new List<int>();
            foreach (var anchor in pointAnchorPair.Value)
            {
                var maxIdx = anchor.Select((value, index) => new { Value = value, Index = index }).Aggregate((a, b) => (a.Value > b.Value) ? a : b).Index;
                maxIdxs.Add(maxIdx);
            }
            maxClassConfIndexs.Add(pointAnchorPair.Key, maxIdxs);
        }

        var det_conf = GetConfidence(parsedOutput);

        var finalConfs = new Dictionary<Point, Dictionary<int, float>>();
        var remainedOutput = new Dictionary<Point, List<float[]>>();
        foreach (var pointAnchorPair in maxClassConfIndexs)
        {
            var point = pointAnchorPair.Key;
            finalConfs.Add(point, new Dictionary<int, float>());

            for(int i = 0; i < anchorCount; ++i)
            {
                //在该锚中拥有最高概率的类。
                var classInAnchor = pointAnchorPair.Value[i];
                var classProbabilitiesInAnchor = classProbabilities[point][i][classInAnchor];
                var detConInAnchor = det_conf[point][i];

                var confidentValue = classProbabilitiesInAnchor * detConInAnchor;
                if (confidentValue > confThreshold)
                {
                    finalConfs[point].Add(i, confidentValue);
                }
            }
        }

        var boxes = ExtractBoundingBox(parsedOutput);

        var processedBoxes = (from b in boxes
                            where finalConfs[b.Key].Count != 0
                            select b).ToDictionary(b => b.Key, b => b.Value);

        var remainedBoxes = new List<CubicBoundingBox>();
        foreach(var pointBoxPair in processedBoxes)
        {
            var point = pointBoxPair.Key;
            for(int i = 0; i < anchorCount; ++i)
            {
                if (finalConfs[point].ContainsKey(i))
                {
                    remainedBoxes.Add(pointBoxPair.Value[i]);
                }
            }
        }

        BoundingBoxes = remainedBoxes;

        //淘汰掉所有信度小于阈值的锚

        //var remainedConfs = (from c in confs
        //                     where c.Value > confthreshold
        //                     select new KeyValuePair<Point, float[]>(c.Key, firstOutput[c.Key])).ToDictionary(c=>c.Key, c=>c.Value);

        ////Point maxConfCordniate;
        ////float maxConf = float.MinValue;
        ////foreach(var confPair in confs)
        ////{
        ////    var conf = confPair.Value;
        ////    var confCordinate = confPair.Key;
        ////    if(conf > maxConf)
        ////    {
        ////        maxConf = conf;
        ////        maxConfCordniate = confCordinate;
        ////    }
        ////}

        ////提取边框
        //var boundingBoxes = ExtractBoundingBox(remainedConfs);
        ////BoundingBoxes = boundingBoxes[maxConfCordniate];
    }
    public List<CubicBoundingBox> BoundingBoxes { private set; get; }
    private int classOut = 13;
    private int anchorCount = 1;
    private Dictionary<Point, List<float>> GetConfidence(Dictionary<Point, float[]> dictionary)
    {
        Dictionary<Point, List<float>> confs = new Dictionary<Point, List<float>>();
        foreach (var pair in dictionary)
        {
            confs.Add(pair.Key, new List<float>());
            for (int anchor = 0; anchor < anchorCount; ++anchor)
            {
                int anchorOffset = anchor * (2 * 9 + 1 + classOut);
                confs[pair.Key].Add(Sigmoid(pair.Value[2 * 9 + anchorOffset]));
            }

        }
        return confs;
    }
    private Dictionary<Point, List<float[]>> GetClassProbabilities(Dictionary<Point, float[]> dictionary)
    {
        Dictionary<Point, List<float[]>> results = new Dictionary<Point, List<float[]>>();
        foreach (var pair in dictionary)
        {
            results.Add(pair.Key, new List<float[]>());
            for (int anchor = 0; anchor < anchorCount; ++anchor)
            {
                var anchorOffset = (2 * 9 + 1 + classOut) * anchor;
                float[] probilities = pair.Value.Skip(2 * 9 + 1 + anchorOffset).Take(classOut).ToArray();
                results[pair.Key].Add(Softmax(probilities));
            }
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
    private Dictionary<Point, float[]> ClassifyOutputBySegements(float[] probability)
    {
        int segementsPerRow = 13;
        int segementsPerColumn = 13;
        int segements = segementsPerRow * segementsPerColumn;
        //int anchors = 1;

        int outputVectorLengthPerSegement = probability.Length / segements;
        Dictionary<Point, float[]> outputs = new Dictionary<Point, float[]>();
        for (int i = 0; i < probability.Length; ++i)
        {
            //分片序号
            int segementNo = i % segements;

            int outputVectorPerSegementOrder = i / segements;

            int rowNo = segementNo / segementsPerRow;

            int columnNo = segementNo % segementsPerRow;

            var cordinate = new Point(columnNo, rowNo);

            if (!outputs.ContainsKey(cordinate))
            {
                outputs[cordinate] = new float[outputVectorLengthPerSegement];
            }
            outputs[cordinate][outputVectorPerSegementOrder] = probability[i];
        }

        return outputs;
    }

    /// <summary>
    /// 提取出边框盒
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

                //�������ĺ���
                //var confidentValue = Sigmoid(outputValue[2 * 9 + anchorOffset]);

                PointF centerPoint = new PointF((Sigmoid(outputValue[0 + anchorOffset]) + output.Key.X) / SegementsSize.Width, (Sigmoid(outputValue[1 + anchorOffset]) + output.Key.Y) / SegementsSize.Height);
                CubicBoundingBox cubicBoundingBoxes = new CubicBoundingBox();

                cubicBoundingBoxes.ControlPoint[8] = centerPoint;

                for (int j = 2; j < 2 + 8 * 2; j += 2)
                {
                    PointF point = new PointF((outputValue[j + anchorOffset] + output.Key.X) / SegementsSize.Width, (outputValue[j + 1 + anchorOffset] + output.Key.Y) / SegementsSize.Height);
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