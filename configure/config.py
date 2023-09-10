import torch
import numpy as np
from _utils.utils import get_classes_name

# ===annotation===
annotation_path = r"D:\dataset\image\BDD\BDD100K\Annotation\bdd100k_labels_images_train.json"
image_root = r"D:\dataset\image\BDD\BDD100K\Image\train"

# ===generator===
classes_path = r"C:\Users\chen\Desktop\zvan\HybridNets-main\model_data\classes.txt"
segmentation_classes_path = r"C:\Users\chen\Desktop\zvan\HybridNets-main\model_data\segmentation_classes.txt"
batch_size = 4
train_split = 0.6
input_size = (384, 640)
# 使用三组scale和aspect ratio固定anchor
scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])  # 比例，用于固定某个感受野下的base_anchor_size
ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]  # x,y的宽高比

# ===model===
backbone = "efficientnet_b0"  # 骨干网络
fpn_cells = 6
num_layers = 4
num_anchors = scales.__len__() * ratios.__len__()
num_features = 160
conv_channels = [24, 40, 112, 320]  # 根据huggingface timm官网查到
up_scale = (4, 4)
out_indices = (1, 2, 3, 4)  # efficient-b0输出5个值，拿到索引为1,2,3,4的输出

# ===training===
Epochs = 10
resume_train = True
learning_rate = 5e-4
weight_decay = 5e-4
class_names = get_classes_name(classes_path)
segmentation_class_names = get_classes_name(segmentation_classes_path)
device = torch.device('cuda') if torch.cuda.is_available() else None
ckpt_path = "./checkpoint"

# ===prediction===
iou_thresh = .5
nms_thresh = .3
font_color = (0, 255, 255)
rect_color = (0, 0, 255)
thickness = .5
per_sample_interval = 1000
segmentation_colors = np.array([[0, 0, 0],
                                [255, 0, 0],
                                [0, 255, 0]])  # adjustable
font_path = r"C:\Users\chen\Desktop\zvan\HybridNets-main\font\simhei.ttf"
train_sample_path = "./result/train/Batch{}.jpg"
val_sample_path = "./result/val/Batch{}.jpg"
