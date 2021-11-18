import os
import datetime

import torch
import torchvision

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2
from my_dataset import VOCDataSet
from train_utils import train_eval_utils as utils

results_file = "/home/chaoc/Desktop/deep-learning-for-image-processing/results_file.txt"
def create_model(num_classes):
    # https://download.pytorch.org/models/vgg16-397923af.pth
    # 如果使用vgg16的话就下载对应预训练权重并取消下面注释，接着把mobilenetv2模型对应的两行代码注释掉
    # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    # backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    # backbone.out_channels = 512

    # https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    backbone = MobileNetV2().features
    backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    a=anchor_generator.num_anchors_per_location()[0]

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model

device = torch.device("cuda")

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

data_set = "/home/chaoc/Desktop/deep-learning-for-image-processing/data_set"
# check voc root
if os.path.exists(data_set) is False:
    raise FileNotFoundError("data_set dose not in path:'{}'.".format(data_set))

# load train data set
# VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
train_dataset = VOCDataSet(data_transform["train"], 0)
out=train_dataset[1]

# 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using %g dataloader workers' % nw)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn)

# load validation data set
# VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
val_dataset = VOCDataSet(data_transform["val"], 1)
val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=val_dataset.collate_fn)

# create model num_classes equal background + 20 classes
model = create_model(7)
# print(model)

model.to(device)

# define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.33)

train_loss = []
learning_rate = []
val_map = []

for epoch in range(0, 20):
    # train for one epoch, printing every 10 iterations
    mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                            device=device, epoch=epoch,
                                            print_freq=50, warmup=True)
    train_loss.append(mean_loss.item())
    learning_rate.append(lr)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    coco_info = utils.evaluate(model, val_data_set_loader, device=device)

    # write into txt
    with open(results_file, "a") as f:
        # 写入的数据包括coco指标还有loss和learning rate
        result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
        txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        f.write(txt + "\n")

    val_map.append(coco_info[1])  # pascal mAP

    # save weights
    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch}
    if epoch >=15:
        torch.save(save_files, "/home/chaoc/Desktop/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/save_weights/resNetFpn-model-{}.pth".format(epoch))

# plot loss and lr curve
if len(train_loss) != 0 and len(learning_rate) != 0:
    from plot_curve import plot_loss_and_lr
    plot_loss_and_lr(train_loss, learning_rate)

# plot mAP curve
if len(val_map) != 0:
    from plot_curve import plot_map
    plot_map(val_map)