using System;
using System.Collections.Generic;
using System.Drawing;
using MathNet.Numerics;

public class OutputParser
{
    private float[] _originalOutput;
    private Dictionary<Point, ParsedOutput> Results;
    public OutputParser(float[] nnOutput)
    {
        _originalOutput = new float[nnOutput.Length];
        nnOutput.CopyTo(_originalOutput, 0);
        Results = new Dictionary<Point, ParsedOutput>();
    }

    public Dictionary<Point, float[]> ClassifyOutputBySegemtns(float[] probability)
    {
        //1���ȷָ������
        int segementsPerRow = 13;
        int segementsPerColumn = 13;
        int segements = segementsPerRow * segementsPerColumn;
        int outputVectorLengthPerSegement = probability.Length / segements;
        //int anchors = 1;

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

        return outputs;
    }
}