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


def valid_corner_confidences(box_gt, box_pr, img_width, img_height, th=80.0, sharpness=2):
    for i in range(9):
        box_gt[2 * i] = box_gt[2 * i] * img_width
        box_pr[2 * i] = box_pr[2 * i] * img_width

        box_gt[2 * i + 1] = box_gt[2 * i + 1] * img_height
        box_pr[2 * i + 1] = box_pr[2 * i + 1] * img_height

    box_gt = box_gt[0:18].view(9, -1).cuda()
    box_pr = box_pr[0:18].view(9, -1).cuda()
        
    dist = box_pr - box_gt
    dist = torch.sqrt(torch.sum((dist)**2))

    conf = torch.exp(sharpness * (1.0 - torch.min(dist, torch.tensor(th).cuda())/th)) - 1

    conf0 = torch.exp(torch.FloatTensor([sharpness])) - 1

    conf = conf / conf0
    
    return conf
