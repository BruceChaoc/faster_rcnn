from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.io import read_image

def showSinglech(img):
    plt.imshow(img)
    plt.show()
    plt.close()

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    plt.close()

imgPaths = Path('/home/chaoc/Desktop/data_set/train/IMAGES/').glob('*.jpg')
imgtmpR, imgtmpG, imgtmpB = 0, 0, 0

for i, imgPath in enumerate(imgPaths):
    imgtmpR += read_image(str(imgPath))[0, :, :]
    imgtmpG += read_image(str(imgPath))[1, :, :]
    imgtmpB += read_image(str(imgPath))[2, :, :]

imgsMEANR = imgtmpR.reshape(-1).float().mean()/256
imgsMEANG = imgtmpG.reshape(-1).float().mean()/256
imgsMEANB = imgtmpB.reshape(-1).float().mean()/256

imgsSTDR = imgtmpR.reshape(-1).float().std()/256
imgsSTDG = imgtmpG.reshape(-1).float().std()/256
imgsSTDB = imgtmpB.reshape(-1).float().std()/256

print("imgsMEANR::{}\r\nimgsMEANG::{}\r\nimgsMEANB::{}\r\nimgsSTDR::{}\r\nimgsSTDG::{}\r\nimgsSTDB::{}".format(imgsMEANR, imgsMEANG, imgsMEANB, imgsSTDR, imgsSTDG, imgsSTDB))

