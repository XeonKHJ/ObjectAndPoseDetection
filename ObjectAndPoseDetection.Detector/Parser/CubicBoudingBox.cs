using System;
using System.Drawing;
using System.Collections.Generic;

public class CubicBoundingBox
{
    public string Identity{set;get;}
    private PointF[] _controlPoints = new PointF[9];
    public PointF[] ControlPoint
    {
        get
        {
            return _controlPoints;
        }
    }
}