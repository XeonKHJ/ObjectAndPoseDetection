import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib.pyplot as plt
import scipy.misc
import warnings
import sys
import argparse
warnings.filterwarnings("ignore")
from torch.autograd import Variable
from torchvision import datasets, transforms

import dataset_multi
from darknet_multi import Darknet
from utils_multi import *
from cfg import parse_cfg
from MeshPly import MeshPly



if __name__ == '__main__':
    #datacfg = 'cfg/ape.data'
    modelcfg = 'multi_obj_pose_estimation/cfg/yolo-pose-multi.cfg'
    weightfile = '../Assets/trained/multi.weights'

    model = Darknet(modelcfg)
    model.load_weights(weightfile)
    model = model.cuda()
    model.eval()

    test_width = 416
    test_height = 416

    imagesPath = ['../Assets/singleshotpose/LINEMOD/benchvise/JPEGImages/000000.jpg']

    #imgTensor = torch.tensor([1, 3, test_width, test_height])

    img = Image.open(imagesPath[0]).convert('RGB')
    img = img.resize((test_width, test_height))

    originalImg = img

    img = transforms.ToTensor()(img)
    imgTensor = torch.zeros([1, img.size(0), img.size(1), img.size(2)])
    imgTensor[0] = img

    data = imgTensor.cuda()

    output = model(data)
    print(output)

    #
    net_options = parse_cfg(cfgfile)[0]
    loss_options = parse_cfg(cfgfile)[-1]

    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]

    all_boxes = get_multi_region_boxes(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, int(trgt[0][0]), only_objectness=0);
    
    
