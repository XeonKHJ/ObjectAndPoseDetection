import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from scipy import spatial

import struct 
import imghdr 


def parseBoxes(boxes, clsCount, anchorCount, controlPointCount = 9):
    boxLength = (controlPointCount * 2 + 3)

    boxes = torch.tensor(boxes)
    boxes = boxes.view(-1, boxLength)

    return boxes

