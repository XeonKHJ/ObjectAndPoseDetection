using System;
using System.Collections.Generic;
using System.Drawing;

public class ParsedOutput
{
    public float ConfidenceValue{set;get;}
    public float[] Probabilities{set;get;}
    public Point CenterPoint{set;get;}

    public CubicBoundingBox BoudingBox{set;get;} = new CubicBoundingBox();
}