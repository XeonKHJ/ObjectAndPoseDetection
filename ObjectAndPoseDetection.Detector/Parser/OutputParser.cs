using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using Google.Protobuf.Reflection;
using MathNet.Numerics;
using MathNet.Numerics.RootFinding;

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
        var boundingBoxes = ExtractBoundingBox(batchOutpus.First());
    }

    private List<CubicBoundingBox> cubicBoudingBoxes;

    public List<CubicBoundingBox> CubicBoudingBox { get; }
    private List<Dictionary<Point, float[]>> ClassifyOutputBySegements(IEnumerable<float[]> probabilities)
    {
        //1���ȷָ������
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
                //ȷ���ǵڼ�����
                int segementNo = i % segements;

                //ȷ�����ڵڼ�����Ƭ��
                int outputVectorPerSegementOrder = i / segements;

                //ȷ���ǵڼ���
                int rowNo = segementNo / segementsPerRow;

                //ȷ���ǵڼ���
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

    private List<CubicBoundingBox> ExtractBoundingBox(Dictionary<Point, float[]> outputs)
    {
        List<CubicBoundingBox> boxes = new List<CubicBoundingBox>();
        foreach (var output in outputs)
        {
            var outputValue = output.Value;

            //�������ĺ���
            var confidentValue = Sigmoid(outputValue[2 * 9]);

            PointF centerPoint = new PointF(Sigmoid(outputValue[0]), Sigmoid(outputValue[1]));
            CubicBoundingBox cubicBoundingBoxes = new CubicBoundingBox();

            cubicBoundingBoxes.ControlPoint[8] = centerPoint;

            for (int i = 2; i < 2 + 8 * 2; i+=2)
            {
                PointF point = new PointF(Sigmoid(outputValue[i]), Sigmoid(outputValue[i + 1]));
                cubicBoundingBoxes.ControlPoint[i / 2 - 1] = point;
            }

            boxes.Add(cubicBoundingBoxes);
        }
        return boxes;
    }

    private static float Sigmoid(double value)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-value));
    }
}