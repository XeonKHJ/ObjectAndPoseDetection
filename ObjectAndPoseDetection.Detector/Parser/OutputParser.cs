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
}