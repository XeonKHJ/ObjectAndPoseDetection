import torch
import torchvision
from PIL import Image

img = Image.open("../Assets/DataSets/LINEMOD/driller/JPEGImages/000007.jpg").convert('RGB')
imgTensor = torchvision.transforms.ToTensor()(img)

withReg = ""
withoutReg = ""

for channel in imgTensor:
    pixels = channel
    for rowPixel in pixels:
        for columnPixel in rowPixel:
            withReg = withReg + str(columnPixel.item()) + ','
        withReg = withReg + '\n'

print(withReg)

