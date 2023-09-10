import os
import json
import numpy as np
from PIL import Image
import torch
import random
from configure.config import class_names, segmentation_class_names
from _utils.utils import get_random_data, BBoxUtility


class Generator:
    def __init__(self,
                 image_root: str,
                 anno_path: str,
                 input_size: tuple,
                 batch_size: int,
                 train_split: float,
                 priors: np.ndarray,
                 num_classes: int):
        '''
        prior boxes tuning method based on retinaFace
        :param image_root:
        :param anno_path:
        :param input_size:
        :param batch_size:
        :param train_split:
        :param priors:the total prior boxes under each receptive field
        :param num_classes:number of detection categories
        '''
        self.image_root = image_root
        self.anno_path = anno_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.split_train_val()
        self.box_util = BBoxUtility(priors=priors,
                                    num_classes=num_classes)

    def split_train_val(self):
        with open(self.anno_path, mode='r', encoding='utf-8') as f:
            self.total_files_info = json.load(f)

        self.train_files_len = int(self.total_files_info.__len__() * self.train_split)
        self.val_files_len = self.total_files_info.__len__() - self.train_files_len

    def get_train_len(self):
        if not self.train_files_len % self.batch_size:
            return self.train_files_len // self.batch_size
        else:
            return self.train_files_len // self.batch_size + 1

    def get_val_len(self):
        if not self.val_files_len % self.batch_size:
            return self.val_files_len // self.batch_size
        else:
            return self.val_files_len // self.batch_size + 1

    def generate(self, trianing: bool = True):
        np.random.seed(1)
        np.random.shuffle(self.total_files_info)
        if trianing:
            files_info = self.total_files_info[:self.train_files_len]
        else:
            files_info = self.total_files_info[self.train_files_len:]
        while True:
            sources, sg_sources, targets = [], [], []
            for i, file_info in enumerate(files_info):
                image_path = os.path.join(self.image_root, file_info['name'])
                boxes, sg_points, sg_names = [], [], []
                for label in file_info['labels']:
                    # file_info['labels']为dict,label为list
                    if label['category'] not in class_names + segmentation_class_names:
                        continue
                    if label['category'] in segmentation_class_names:
                        sg_points.append(label['poly2d'])
                        sg_names.append(label['category'])
                    if label['category'] in class_names:
                        dt_points = label['box2d']
                        dt_points.update({'category': class_names.index(label['category'])})
                        boxes.append(list(map(lambda x: int(x), dt_points.values())))
                # 如果有目标检测框，将list转为ndarray

                boxes = np.array(boxes)
                image, sg_image, boxes = get_random_data(image_path=image_path,
                                                         boxes=boxes,
                                                         sg_points=sg_points,
                                                         sg_names=sg_names,
                                                         seg_class_name=segmentation_class_names,
                                                         input_shape=self.input_size)
                # 对box的label进行one_hot编码
                try:
                    # 这里使用异常处理是因为可能boxes为空列表
                    one_hot_label = np.eye(len(class_names), dtype='int')[np.array(boxes)[:, -1].astype('int')]
                except IndexError:
                    pass
                if len(boxes):
                    # boxes包含有坐标和类别
                    boxes = np.concatenate([boxes[:, :4], one_hot_label], axis=-1)
                    del one_hot_label
                else:
                    pass

                assign_boxes = self.box_util.assign_boxes(boxes=boxes)

                sources.append(image)
                sg_sources.append(sg_image)
                targets.append(assign_boxes)

                if sources.__len__() == self.batch_size or i + 1 == files_info.__len__():
                    sources_ = np.array(sources.copy()).transpose((0, 3, 1, 2))  # 使用torch框架，CHW
                    sg_sources_ = np.eye(segmentation_class_names.__len__())[np.array(sg_sources.copy())]
                    targets_ = np.array(targets.copy())

                    sources.clear()
                    sg_sources.clear()
                    targets.clear()

                    yield sources_, sg_sources_, targets_


if __name__ == '__main__':
    import configure.config as cfg
    from _utils.anchors import Anchors

    priors = Anchors(scales=cfg.scales,
                     ratios=cfg.ratios)(cfg.input_size)
    data_gen = Generator(image_root=cfg.image_root,
                         anno_path=cfg.annotation_path,
                         input_size=cfg.input_size,
                         batch_size=cfg.batch_size,
                         train_split=cfg.train_split,
                         priors=priors,
                         num_classes=cfg.class_names.__len__())
    train_gen = data_gen.generate(trianing=True)
    sources, sg_sources, targets = next(train_gen)
    1
