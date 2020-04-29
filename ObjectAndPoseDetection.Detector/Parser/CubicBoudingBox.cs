using System;
using System.Drawing;
using System.Collections.Generic;

public class CubicBoudingBox
{
    public string Identity{set;get;}
    private Point[] _controlPoints = new Point[9];
    public Point[] ControlPoint
    {
        get
        {
            return _controlPoints;
        }
    }
}