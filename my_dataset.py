from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
import random
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np

def draw_box(image, xmin, ymin, xmax, ymax, label, line_thickness=2):
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    labeln={
        1: "rolled-in_scale",
        2: "patches",
        3: "crazing",
        4: "pitted_surface",
        5: "inclusion",
        6: "scratches"
    }
    label = label.to("cpu").item()
    # Draw all boxes onto image.
    draw = ImageDraw.Draw(image)
    draw.line([(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)], width=line_thickness)
    text_width, text_height = font.getsize(label)
    margin = np.ceil(0.05 * text_height)
    total_display_str_height = 1 + 2 * 0.05

    if ymax > total_display_str_height:
        text_bottom = ymax
    else:
        text_bottom = ymin + total_display_str_height
    draw.text((xmin + margin, text_bottom - text_height - margin),
              labeln[label],
              fill='black',
              font=font)

class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, transforms=None, settype=0):

        self.root = "/home/chaoc/Desktop/data_set/train"
        self.img_root = "/home/chaoc/Desktop/data_set/train/IMAGES"
        self.annotations_root = "/home/chaoc/Desktop/data_set/train/ANNOTATIONS"
        self.xml_list = os.listdir(self.annotations_root)
        num = len(self.xml_list)
        if settype==1:
            self.xml_list=random.sample(self.xml_list, k=int(num*0.2))
        

        # check file
        for xml_path in self.xml_list:
            assert os.path.exists(os.path.join(self.annotations_root,xml_path)), "not found '{}' file.".format(xml_path)
            self.xml_list = [os.path.join(self.annotations_root,xml_path) for xml_path in self.xml_list]

        # read class_indict
        json_file = '/home/chaoc/Desktop/faster_rcnn/pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        json_file.close()

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # #>>>>>打印带bounding box的图片,不使用请注释>>>>>
        # for box, label in zip(boxes, labels):
        #     draw_box(image, box[0], box[1], box[2], box[3], label)
        # # image.show()
        # image.save("/home/chaoc/Desktop/dataset_box/{}.jpg".format(idx))
        # #<<<<<打印带bounding box的图片,不使用请注释<<<<<

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
