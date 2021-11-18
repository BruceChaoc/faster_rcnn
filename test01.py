import os
import time
import json
from numpy import mod

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
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
        layer_viz.save("/home/chaoc/Desktop/test_result.jpg")
        plt.subplot(2, 8, 1)
        plt.imshow(layer_viz)
        plt.axis("off")
        plt.show()
        plt.close()
        pass
        # for i, filters in enumerate(layer_viz):
        #     if i == 16:
        #         break
        #     plt.subplot(2, 8, i + 1)
        #     plt.imshow(filters)
        #     plt.axis("off")
        # plt.show()
        # plt.close()
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
    train_weights = "/home/chaoc/Desktop/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/save_weights/resNetFpn-model-19.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = '/home/chaoc/Desktop/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}
    annotations_root = "/home/chaoc/Desktop/deep-learning-for-image-processing/data_set/test/IMAGES"
    for i in os.listdir(annotations_root):
        # load image
        original_img = Image.open("/home/chaoc/Desktop/deep-learning-for-image-processing/data_set/test/IMAGES/{}".format(i))
        # original_img = Image.open("/home/chaoc/Desktop/macPycharm/hymenoptera_data/train/bees/39747887_42df2855ee.jpg")
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

        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.06,
                     line_thickness=3)
            # 保存预测的图片结果
            original_img.save(
                "/home/chaoc/Desktop/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/result/test_result{}.jpg".format(
                    i))


if __name__ == '__main__':
    main()

