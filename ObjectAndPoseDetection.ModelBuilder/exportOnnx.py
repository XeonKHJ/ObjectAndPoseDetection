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
import onnx
import onnxruntime

def to_numpy(tensor):

    inputs = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    inputs = inputs * 255
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
    datacfg = 'cfg/ape.data'
    modelcfg = 'cfg/yolo-pose.cfg'
    weightfile = '../Assets/trained/ape.weights'
    onnxOutputPath = '../Assets/OnnxModel/SingelObjectApeModelV8.onnx'
    model = Darknet(modelcfg)
    model.load_weights(weightfile)
    model.eval()

    test_width = 416
    test_height = 416

    exportToOnnx(model, onnxOutputPath)

    #用onnx模型输出

    #打开图片
    #img = OpenImageAsTensor('LINEMOD/ape/JPEGImages/000068.jpg', test_height, test_width)
    img = OpenImageAsTensor('white.jpg', test_height, test_width)
    img.unsqueeze_(0)

    print("图片：\n", img)

    ort_session = onnxruntime.InferenceSession(onnxOutputPath)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    
    ort_outs = ort_session.run(None, ort_inputs)
    
    print("使用ONNX：\n", ort_outs)

    print("使用PyTorch：\n", model(img))