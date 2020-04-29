using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ObjectAndPoseDetection.Detector.DataStructures
{
    public class ImagePixels
    {
        [ColumnName("image")]
        [VectorType]
        public float[] Pixels;
    }
}
