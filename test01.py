import os
import time

import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN
from backbone import resnet50_fpn_backbone

def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model

total_feat_out = []
total_feat_in = []

# 定义  forward_hook_function
def hook_fn_forward(module, input, output):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    if class_name == 'GeneralizedRCNNTransform':
        plt.figure(figsize=(50, 10))
        layer_viz = output[0].tensors[0, :, :, :]
        layer_viz = layer_viz.data.cpu()
        layer_viz = layer_viz.transpose(0, 1)
        layer_viz = layer_viz.transpose(1, 2)
        plt.subplot(2, 8, 1)
        plt.imshow(layer_viz)
        plt.axis("off")
        plt.show()
        plt.close()
        pass
    # print(module)
    # print('input', input)
    # print('output', output)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=7)

    # load train weights
    train_weights = "/home/chaoc/Desktop/save_weights/resNetFpn-model-19.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    annotations_root = "/home/chaoc/Desktop/data_set/test/IMAGES"
    for i in os.listdir(annotations_root):
        # load image
        original_img = Image.open("/home/chaoc/Desktop/data_set/test/IMAGES/{}".format(i))
        # original_img = Image.open("/home/chaoc/Desktop/hymenoptera_data/train/bees/39747887_42df2855ee.jpg")
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)

        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        modules = model.named_children()
        for name, module in modules:
            module.register_forward_hook(hook_fn_forward)
        o = model(img.to(device))

if __name__ == '__main__':
    main()

