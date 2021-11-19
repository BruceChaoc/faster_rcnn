from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.io import read_image


plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

def distributedShow(imgs):
    img = imgs[0, :, :]
    img = img.reshape(-1).to('cpu')
    plt.scatter(range(len(img)), img, color="green", s=1, marker="o")
    plt.show()
    plt.close()


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.reshape(-1).to('cpu')
        axs[0, i].plot(img)
    #     img = T.ToPILImage()(img.to('cpu'))
    #     axs[0, i].imshow(np.asarray(img))
    #     axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    plt.close()


####################################
# The :func:`~torchvision.io.read_image` function allows to read an image and
# directly load it as a tensor

# ant = read_image(str(Path('/home/chaoc/Desktop/') / 'rgb.jpg'))
# bee = read_image(str(Path('/home/chaoc/Desktop/hymenoptera_data/train/bees') / '522104315_5d3cb2758e.jpg'))
ant = read_image(str(Path('/home/chaoc/Desktop/data_set/test/IMAGES/') / '1646.jpg'))
bee = read_image(str(Path('/home/chaoc/Desktop/data_set/train/IMAGES/') / '26.jpg'))
show([ant, bee])

####################################
# Transforming images on GPU
# --------------------------
# Most transforms natively support tensors on top of PIL images (to visualize
# the effect of the transforms, you may refer to see
# :ref:`sphx_glr_auto_examples_plot_transforms.py`).
# Using tensor images, we can run the transforms on GPUs if cuda is available!

import torch.nn as nn

transforms = torch.nn.Sequential(
    # T.RandomCrop(224),
    # T.RandomHorizontalFlip(p=0.3),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.499, 0.499, 0.499], [0.287, 0.287, 0.287])
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ant = ant.to(device)
bee = bee.to(device)

transformed_ant = transforms(ant)
transformed_bee = transforms(bee)
distributedShow(transformed_ant)
show([transformed_ant, transformed_bee])

####################################
# Scriptable transforms for easier deployment via torchscript
# -----------------------------------------------------------
# We now show how to combine image transformations and a model forward pass,
# while using ``torch.jit.script`` to obtain a single scripted module.
#
# Let's define a ``Predictor`` module that transforms the input tensor and then
# applies an ImageNet model on it.

from torchvision.models import resnet18


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True, progress=False).eval()
        self.transforms = nn.Sequential(
            T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)


####################################
# Now, let's define scripted and non-scripted instances of ``Predictor`` and
# apply it on multiple tensor images of the same size

predictor = Predictor().to(device)
scripted_predictor = torch.jit.script(predictor).to(device)

batch = torch.stack([ant, bee]).to(device)

res = predictor(batch)
res_scripted = scripted_predictor(batch)

####################################
# We can verify that the prediction of the scripted and non-scripted models are
# the same:

import json

with open(Path('assets') / 'imagenet_class_index.json', 'r') as labels_file:
    labels = json.load(labels_file)

for i, (pred, pred_scripted) in enumerate(zip(res, res_scripted)):
    assert pred == pred_scripted
    print(f"Prediction for Dog {i + 1}: {labels[str(pred.item())]}")

####################################
# Since the model is scripted, it can be easily dumped on disk and re-used

import tempfile

with tempfile.NamedTemporaryFile() as f:
    scripted_predictor.save(f.name)

    dumped_scripted_predictor = torch.jit.load(f.name)
    res_scripted_dumped = dumped_scripted_predictor(batch)
assert (res_scripted_dumped == res_scripted).all()
