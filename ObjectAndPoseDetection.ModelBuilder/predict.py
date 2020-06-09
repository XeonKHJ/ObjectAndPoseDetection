import os

import time
import torch
import argparse
import scipy.io
import warnings
from torch.autograd import Variable
from torchvision import datasets, transforms
import dataset
from darknet import Darknet
from utils import *
from MeshPly import MeshPly

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
        cls_conf = box[19]
        cls_id = box[20]
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

    for className in ['benchvise']:
        datacfg = 'cfg/' + className + '.data'
        modelcfg = 'cfg/yolo-pose.cfg'
        weightfile = '../Assets/singleshotpose/backup/'+className+'/model_backup.weights'
        outputFolder = "../Assets/Outputs/OnlyGT/" + className+ "/"
        inputFolder = "../Assets/DataSets/LINEMOD/" + className + "/JPEGImages/"
        gtLabels = "../Assets/DataSets/LINEMOD/"+className+"/labels/"
        labelPath = list()
        imagesPath = list()
        labelFiles = os.listdir(gtLabels)
        ImageFiles = os.listdir(inputFolder)
        labels = list()
        for file in labelFiles:
            if os.path.splitext(file)[-1] == ".txt":
                labelPath.append(gtLabels + file)
                labelfile = open(gtLabels + file, "r")
                label = labelfile.read().split(' ')
                for i in range(len(label)):
                    label[i] = float(label[i])
                labels.append(label)

        for file in ImageFiles:
            if os.path.splitext(file)[-1] == ".jpg":
                imagesPath.append(inputFolder + file)


        #model = Darknet(modelcfg)
        #model.load_weights(weightfile)
        #model.eval()

        test_width = 416
        test_height = 416

        for i in range(len(imagesPath)):

            '''
            img = Image.open(imagesPath[i]).convert('RGB')
            img = img.resize((test_width, test_height))

            originalImg = img

            img = transforms.ToTensor()(img)
            #print("image:\n", img*255)
            imgTensor = torch.zeros([1, img.size(0), img.size(1), img.size(2)])
            imgTensor[0] = img

            #print(imgTensor * 255)

            t2 = time.time()
            data = imgTensor.cuda()
            data = Variable(data, volatile=True)
            model = model.cuda()
            model.eval()
            output = model(data).data

            output = get_region_boxes(output, 1, 9)
            #print("Output: \n", output)
            t3 = time.time()
            '''

            #output1 = list()
            gtBox = labels[i][1:19]
            gtBox.append(1)
            gtBox.append(1)
            gtBox.append(labels[i][0])

            '''
            for o in output:
                output1.append(o.item())
            '''

            #print('-------------------------------')
            #print(output1)

            classesName = list()
            classesName.append(className)

            img = cv2.imread(imagesPath[i])

            img_width = img.shape[1]
            img_height = img.shape[0]

            for j in range(9):
                #output1[2*j] = int(output1[2*j] * img_width)
                #output1[2*j + 1] = int(output1[2*j + 1] * img_height)
                gtBox[2*j] = int(gtBox[2*j] * img_width)
                gtBox[2*j + 1] = int(gtBox[2*j + 1] * img_height)

            #print('-------------------------------')
            #print(output1)
            #print("用时：", t3-t2)
            #img = plot_boxes_cv2(img, output1, None, classesName, (0, 0, 255))
            img = plot_boxes_cv2(img, gtBox, outputFolder + str(i) + ".jpg", classesName, (0, 255, 0))

