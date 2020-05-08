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


def to_numpy(tensor):

    inputs = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #inputs = inputs * 255
    #inputs = inputs.astype('int32')

    return inputs

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]

    box = boxes
 


    if color:
        rgb = color
    else:
        rgb = (255, 0, 0)

    thickness = 2
    #画线

    for box in boxes:
        points = list()
        for p in range(1, 9):
            points.append((box[2*p], box[2*p + 1]))

        img = cv2.line(img, points[0], points[1], rgb, thickness)
        img = cv2.line(img, points[0], points[4], rgb, thickness)
        img = cv2.line(img, points[1], points[5], rgb, thickness)
        img = cv2.line(img, points[4], points[5], rgb, thickness)
        img = cv2.line(img, points[5], points[7], rgb, thickness)
        img = cv2.line(img, points[1], points[3], rgb, thickness)
        img = cv2.line(img, points[4], points[6], rgb, thickness)
        img = cv2.line(img, points[0], points[2], rgb, thickness)
        img = cv2.line(img, points[2], points[6], rgb, thickness)
        img = cv2.line(img, points[2], points[3], rgb, thickness)
        img = cv2.line(img, points[3], points[7], rgb, thickness)
        img = cv2.line(img, points[7], points[6], rgb, thickness)
    #img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


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

    imagesPath = ['../Assets/singleshotpose/LINEMOD/benchvise/JPEGImages/000005.jpg']

    #imgTensor = torch.tensor([1, 3, test_width, test_height])

    img = Image.open(imagesPath[0]).convert('RGB')
    img = img.resize((test_width, test_height))

    originalImg = img

    img = transforms.ToTensor()(img)
    imgTensor = torch.zeros([1, img.size(0), img.size(1), img.size(2)])
    imgTensor[0] = img

    data = imgTensor.cuda()

    output = model(data)
    #print(output)

    #
    net_options = parse_cfg(modelcfg)[0]
    loss_options = parse_cfg(modelcfg)[-1]

    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]

    all_boxes = get_multi_region_boxes(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, 0, only_objectness=0);

    for box in all_boxes[0]:
        for point in box:
            if isinstance(point, int) == False:
                point = point.item()
            print(point)

    img = cv2.imread(imagesPath[0])
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    for batch in all_boxes:
        for box in batch:
            for i in range(9):
                box[2 * i] = int(box[2*i] * img_width)
                box[2 * i + 1] = int(box[2*i + 1] * img_height)


    candidateBoxes = list()
    for classidx in range(13):   
        maxConf = -1000000000
        maxBox = 0
        for batch in all_boxes:
            for box in batch:
                if(box[20] == classidx):
                    if(box[19] > maxConf):
                        maxBox = box
        if maxBox != 0:
            candidateBoxes.append(maxBox)

    
    plot_boxes_cv2(img, candidateBoxes, '../Assets/Test/multi.jpg', 'nothing')

    
    
    
