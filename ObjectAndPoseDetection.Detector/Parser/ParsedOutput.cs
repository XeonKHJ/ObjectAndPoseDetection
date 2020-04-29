using System;
using System.Collections.Generic;
using System.Drawing;

public class ParsedOutput
{
    public float ConfidenceValue{set;get;}
    public float[] Probabilities{set;get;}
    public Point CenterPoint{set;get;}

    public CubicBoudingBox BoudingBox{set;get;} = new CubicBoudingBox();
}