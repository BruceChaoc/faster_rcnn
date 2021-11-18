import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

def feature_imshow(inp, title=None):
    """

    Args:
        inp: inner Layer Tensor output
        title: the name of image

    Returns:

    """
    inp = inp.detach().numpy().transpose((1,2,0))  #https://blog.csdn.net/u012762410/article/details/78912667
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    plt.pause(0.001) #pause a bit so that plots are updated
