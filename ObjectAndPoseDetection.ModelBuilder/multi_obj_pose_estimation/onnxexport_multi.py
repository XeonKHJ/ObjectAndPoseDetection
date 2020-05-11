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

import onnx
import onnxruntime

def to_numpy(tensor):

    inputs = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #inputs = inputs * 255
    #inputs = inputs.astype('int32')

    return inputs

def exportToOnnx(model, path):
    input = torch.randn(1, 3, 416, 416, requires_grad=True)
   
    torch.onnx.export(model, 
                      input,
                      path,
                      export_params=True,
                      opset_version=8,
                      do_constant_folding=True,
                      input_names=['image'],
                      output_names=['grid'],
                      dynamic_axes={'image':{0:'batch_size'},
                                    'grid':{0:'batch_size'}})

def OpenImageAsTensor(path, height, width):
    #img = torch.randn(3, 416, 416, requires_grad=True)
    img = Image.open(path).convert('RGB')
    if height * width != 0:
        img = img.resize((test_width, test_height))
    return transforms.ToTensor()(img)

if __name__ == "__main__":
    modelcfg = 'multi_obj_pose_estimation/cfg/yolo-pose-multi.cfg'
    weightfile = '../Assets/trained/multi.weights'
    onnxOutputPath = '../Assets/OnnxModel/MultiObjectDetectionModelv8.onnx'

    model = Darknet(modelcfg)
    model.load_weights(weightfile)

    model.eval()

    test_width = 416
    test_height = 416

    exportToOnnx(model, onnxOutputPath)

    imagesPath = ['../Assets/singleshotpose/LINEMOD/benchvise/JPEGImages/000000.jpg']

    img = OpenImageAsTensor(imagesPath[0], test_height, test_width)

    imgTensor = img.unsqueeze(0)

    regularOutput = model(imgTensor).data
    

    #onnx测试
    ort_session = onnxruntime.InferenceSession(onnxOutputPath)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(imgTensor)}
    
    ort_outs = ort_session.run(None, ort_inputs)

    print("使用ONNX：\n", ort_outs)

    print("使用PyTorch：\n", regularOutput)