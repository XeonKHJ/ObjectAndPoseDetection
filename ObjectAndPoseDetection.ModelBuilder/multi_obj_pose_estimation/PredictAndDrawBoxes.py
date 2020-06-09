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
    x1 = int(round((box[0] - box[2]/2.0) * width))
    y1 = int(round((box[1] - box[3]/2.0) * height))
    x2 = int(round((box[0] + box[2]/2.0) * width))
    y2 = int(round((box[1] + box[3]/2.0) * height))
 
    points = list()
    for p in range(1, 9):
        points.append((box[2*p], box[2*p + 1]))

    if color:
        rgb = color
    else:
        rgb = (255, 0, 0)
    if len(box) >= 7 and class_names:
        cls_conf = box[18]
        cls_id = box[19]
        #print('%s: %f' % (class_names[cls_id], cls_conf))
        classes = len(class_names)
        offset = cls_id * 123457 % classes
        red   = get_color(2, offset, classes)
        green = get_color(1, offset, classes)
        blue  = get_color(0, offset, classes)
        if color is None:
            rgb = (red, green, blue)
        #img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)

    thickness = 2
    #画线
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

    #模型初始化
    #model = Darknet(modelcfg)
    #model.load_weights(weightfile)
    #model = model.cuda()
    #model.eval()

    #加载模型用
    net_options = parse_cfg(modelcfg)[0]
    loss_options = parse_cfg(modelcfg)[-1]
    
    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]
    test_width = 416
    test_height = 416

    datasetPath = '../Assets/DataSets/LINEMOD/'
    datasetImagePaths = datasetPath + 'benchvise/JPEGImages/'
    outputPath = '../Assets/Outputs/Multi/'

    labelFolders = list()

    for className in ['ape', 'benchvise', 'can', 'cat', 'duck', 'driller', 'glue']:
        if className == 'benchvise':
            path = datasetPath + className + '/labels/'
        else:
            path = datasetPath + className + '/labels_occlusion/'
        labelFolders.append(path)

    imageFiles = os.listdir(datasetImagePaths)

    #所有图片的路径
    imagePaths = list()
    imageNos = list()
    for file in imageFiles:
        splitPath = os.path.splitext(file)
        if splitPath[-1] == ".jpg":
            imagePaths.append(datasetImagePaths + file)
            imageNos.append(splitPath[-2])
    #imgTensor = torch.tensor([1, 3, test_width, test_height])

    classNoDict={"ape": 0,
                 "benchvise": 1,
                 "can": 3,
                 "cat": 4,
                 "driller":5,
                 "duck":6,
                 "glue": 8}
    colorDict={0:(255, 0, 0),
               1:(0, 255, 0),
               3:(0, 0, 255),
               4:(255, 125, 0), 
               5:(125, 125, 125),
               6:(255, 125, 125),
               8:(125, 255, 125)}

    for i in range(len(imagePaths)):

        #将要估计的图片转换成张量
        '''
        originalImg = Image.open(imagePaths[i]).convert('RGB').resize((test_width, test_height))
    
        img = transforms.ToTensor()(originalImg)
        imgTensor = torch.zeros([1, img.size(0), img.size(1), img.size(2)])
        imgTensor[0] = img
    
        data = imgTensor.cuda()
    
        output = model(data)

        all_boxes = get_multi_region_boxes(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, 0, only_objectness=0);
        
        for box in all_boxes[0]:
            for point in box:
                if isinstance(point, int) == False:
                    point = point.item()
                print(point)
        '''

        img = cv2.imread(imagePaths[i])
        img_width = img.shape[1]
        img_height = img.shape[0]
        
        '''
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
        '''
        imageNo = imageNos[i]

        labelFilePaths = list()
        for labelFolder in labelFolders:
            labelFilePaths.append(labelFolder + imageNo + ".txt")

        labels = list()
        #从中读取标签
        for file in labelFilePaths:
            if os.path.splitext(file)[-1] == ".txt":
                labelfile = open(file, "r")
                labelContent = labelfile.read()
                label = labelContent.split(' ')
                for i in range(21):
                    label[i] = float(label[i])
                labels.append(label)

        #添加标签盒子
        gt_boxes = list()
        for label in labels:
            box = label[1:19]
            box.append(1)
            box.append((int)(label[0]))
            for pi in range(9):
                box[2 * pi] = (int)(box[2 * pi] * img_width)
                box[2 * pi + 1] = (int)(box[2 * pi + 1] * img_height)
            gt_boxes.append(box)

        classNames = ['ape', 'benchvise', 'can', 'cat', 'duck', 'driller', 'glue']
        for bi in range(len(gt_boxes)):
            gt_box = gt_boxes[bi]
            if bi != (len(gt_boxes) - 1):
                plot_boxes_cv2(img, gt_boxes[bi], None,  "fuck", colorDict[gt_box[19]])
            else:
                plot_boxes_cv2(img, gt_boxes[bi], outputPath + imageNo + ".jpg",  "fuck", colorDict[gt_box[19]])


            