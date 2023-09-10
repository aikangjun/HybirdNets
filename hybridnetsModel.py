import torch
import numpy as np
from torch import nn
from PIL import Image, ImageFont, ImageDraw
from _utils.utils import BBoxUtility, calculate_f1score
from network.Hybirdnets import Hybridnets
from custom.customlosses import ConfidenceLoss, BBOXL1Loss
import configure.config as cfg


class HybridnetsModel:
    def __init__(self,
                 fpn_cells: int,
                 num_layers: int,
                 num_anchors: int,
                 num_classes: int,
                 seg_classes: int,
                 num_features: int,
                 conv_channels: list,
                 out_indices: tuple,
                 up_scale: tuple,
                 backbone: str,
                 priors: np.ndarray,
                 learning_rate: float,
                 weight_decay: float,
                 iou_thresh: float,
                 nms_thresh: float,
                 resume_train: bool,
                 ckpt_path: str,
                 device: object = None):
        '''

        :param fpn_cells:
        :param num_layers: 分类头和回归头 网络的层数
        :param num_anchors:
        :param num_classes:
        :param seg_classes:
        :param num_features:
        :param conv_channels:
        :param out_indices:
        :param up_scale:
        :param backbone:
        :param priors:
        :param learning_rate:
        :param weight_decay:
        :param iou_thresh:
        :param nms_thresh:
        :param resume_train:
        :param ckpt_path:
        :param device:
        '''
        self.box_utils = BBoxUtility(priors=priors,
                                     num_classes=num_classes - 1,
                                     overlap_threshold=iou_thresh,
                                     nms_thresh=nms_thresh)
        self.network = Hybridnets(backbone=backbone,
                                  fpn_cells=fpn_cells,
                                  out_indices=out_indices,
                                  conv_channels=conv_channels,
                                  num_features=num_features,
                                  num_anchors=num_anchors,
                                  num_classes=num_classes,
                                  num_layers=num_layers,
                                  num_seg_classes=seg_classes,
                                  up_scale=up_scale)
        if device:
            self.device = device
            self.network.to(device)

        if resume_train:
            try:
                ckpt = torch.load(ckpt_path)
                self.network.load_state_dict(ckpt['state_dict'])
                print("model successfully loaded,loss is {:3f}".format(ckpt['loss']))
            except FileNotFoundError:
                raise ('please enter the right params path')

        self.conf_loss = ConfidenceLoss()
        self.bbox_loss = BBOXL1Loss()
        self.seg_loss = nn.BCELoss(reduction='mean')

        weights, bias = self.network.split_weights()
        self.optimizer = torch.optim.Adam(params=[{'params': weights, 'weight_decay': weight_decay},
                                                  {'params': bias}],
                                          lr=learning_rate)
        self.train_loss, self.val_loss = 0, 0
        self.train_acc, self.val_acc = 0, 0
        self.train_conf_acc, self.val_conf_acc = 0, 0
        self.train_f1_score, self.val_f1_score = 0, 0
        self.num_classes = num_classes - 1

    def train(self, sources, seg_sources, targets):
        sources = torch.tensor(sources).float()
        seg_sources = torch.tensor(seg_sources).float()
        targets = torch.tensor(targets).float()
        if self.device:
            sources = sources.to(self.device)
            seg_sources = seg_sources.to(self.device)
            targets = targets.to(self.device)
        self.optimizer.zero_grad()  #
        regressions, classifications, segmentations = self.network(sources)
        regressions = torch.reshape(input=regressions,
                                    shape=(regressions.size(0), -1, regressions.size(-1)))
        classifications = torch.reshape(input=classifications,
                                        shape=(classifications.size(0), -1, classifications.size(-1)))

        conf_loss = self.conf_loss(classifications, targets[..., 4:])
        bbox_loss = self.bbox_loss(regressions, targets[..., :5])
        seg_loss = self.seg_loss(torch.permute(input=segmentations, dims=(0, 2, 3, 1)), seg_sources)
        loss = conf_loss + bbox_loss + seg_loss
        loss.backward()  #
        self.optimizer.step()  #
        self.train_loss += loss.data.item()
        prob_confs = torch.where(condition=torch.ge(classifications[..., 0:1], .5),
                                 input=torch.ones_like(classifications[..., 0:1]),
                                 other=torch.zeros_like(classifications[..., 0:1]))
        # torch.prod() 求乘积
        # .data 得到一个新的tensor,但是没有梯度；新的tensor改变，旧的tensor也会改变，旧tensor依然可以求导，不安全
        # .detach() 得到一个新的tensor,但是没有梯度；新的tensor改变，旧的tensor也会改变，旧tensor不可以求导，安全
        # .item()返回一个标准的python数据类型，只能用于tensor有一个元素
        # .tolist() 返回一个标准的python数据list，可用于有多个元素的tensor
        total_num = torch.prod(torch.tensor(targets[..., 4:5].size())).float().detach().item()
        object_num = (1 - targets[..., 4:5]).cpu().sum().data.item()
        correct_conf_num = torch.eq(targets[..., 4:5], prob_confs).float().detach().cpu().sum().item()

        self.train_conf_acc += correct_conf_num / total_num  # 是否区别背景和目标的准确率

        object_mask = (1 - targets[..., 4:5]).squeeze(dim=-1).bool()
        prob_class = classifications[..., 1:][object_mask].argmax(dim=-1)
        real_class = targets[..., 5:][object_mask].argmax(dim=-1)

        correct_class_num = torch.eq(real_class, prob_class).float().detach().sum().cpu().item()
        self.train_acc += correct_class_num / object_num  # 目标的查全率
        self.train_f1_score += calculate_f1score(y_true=targets[..., 5:],
                                                 y_pred=classifications[..., 1:],
                                                 object_mask=object_mask,
                                                 num_classes=self.num_classes)

    def validate(self, sources, seg_sources, targets):
        sources = torch.tensor(sources).float()
        seg_sources = torch.tensor(seg_sources).float()
        targets = torch.tensor(targets).float()

        if self.device:
            sources = sources.to(self.device)
            seg_sources = seg_sources.to(self.device)
            targets = targets.to(self.device)

        regressions, classifications, segmentations = self.network(sources)
        regressions = torch.reshape(input=regressions,
                                    shape=(regressions.size(0), -1, regressions.size(-1)))
        classifications = torch.reshape(input=classifications,
                                        shape=(classifications.size(0), -1, classifications.size(-1)))

        conf_loss = self.conf_loss(classifications, targets[..., 4:])
        bbox_loss = self.bbox_loss(regressions, targets[..., :5])
        seg_loss = self.seg_loss(torch.permute(input=segmentations, dims=(0, 2, 3, 1)), seg_sources)
        loss = conf_loss + bbox_loss + seg_loss
        self.val_loss += loss.data.item()

        prob_confs = torch.where(torch.ge(classifications[..., 0:1], .5),
                                 torch.ones_like(classifications[..., 0:1]),
                                 torch.zeros_like(classifications[..., 0:1]))

        total_num = torch.prod(torch.tensor(targets[..., 4:5].size())).data.item()
        object_num = (1 - targets[..., 4:5]).cpu().sum().data.item()

        correct_conf_num = torch.eq(targets[..., 4:5], prob_confs).float().detach().cpu().sum().data.item()

        self.val_conf_acc += correct_conf_num / total_num

        object_mask = (1 - targets[..., 4:5]).squeeze(dim=-1).bool()

        prob_class = classifications[..., 1:][object_mask].argmax(dim=-1)
        real_class = targets[..., 5:][object_mask].argmax(dim=-1)

        correct_class_num = torch.eq(real_class, prob_class).float().detach().cpu().sum().data.item()

        self.val_acc += correct_class_num / object_num

        self.val_f1_score += calculate_f1score(targets[..., 5:], classifications[..., 1:],
                                               object_mask, self.num_classes)

    def generate_sample(self, sources, batch, sample_path):

        """
        Drawing and labeling
        """
        sources = torch.tensor(sources).float()
        if self.device:
            sources = sources.to(self.device)

        regressions, classifications, segmentations = self.network(sources)
        regressions = regressions.reshape(regressions.size(0), -1, regressions.size(-1))
        classifications = classifications.reshape(classifications.size(0), -1, classifications.size(-1))

        regressions = regressions.detach().cpu().numpy()
        classifications = classifications.detach().cpu().numpy()
        segmentations = segmentations.detach().cpu().numpy()

        index = np.random.choice(sources.size(0), 1)

        out_boxes, out_scores, out_classes = self.box_utils.detection_out([regressions[index],
                                                                           classifications[index]])

        out_boxes = np.array(out_boxes).squeeze(axis=0)
        out_scores = np.array(out_scores).squeeze(axis=0)
        out_classes = np.array(out_classes).squeeze(axis=0)

        source = sources[index[0]].cpu().numpy().transpose([1, 2, 0])
        segmentation = segmentations[index[0]].transpose([1, 2, 0])
        image = Image.fromarray(np.uint8(source * 255))

        if out_boxes.shape[0]:

            out_boxes = self.box_utils.correct_boxes(out_boxes, np.array(cfg.input_size),
                                                     np.array(cfg.input_size))

            out_boxes *= np.tile(np.array(cfg.input_size)[::-1], (2,))

            for coordinate, out_score, out_class in zip(out_boxes.astype('int'),
                                                        out_scores,
                                                        out_classes):

                left, top = coordinate[:2].tolist()
                right, bottom = coordinate[2:].tolist()

                font = ImageFont.truetype(font=cfg.font_path,
                                          size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))

                label = '{:s}: {:.2f}'.format(cfg.class_names[int(out_class)], out_score)

                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                draw.rectangle(coordinate[:2].tolist() + coordinate[2:].tolist(),
                               outline=cfg.rect_color, width=int(2 * cfg.thickness))

                draw.text(text_origin, str(label, 'UTF-8'),
                          fill=cfg.font_color, font=font)
                del draw

        image = np.array(image)
        segmentation = segmentation.argmax(axis=-1)
        for i in range(cfg.segmentation_class_names.__len__()):
            if not i:
                continue
            image[np.equal(segmentation, i)] = cfg.segmentation_colors[i]

        image = Image.fromarray(image)

        image.save(sample_path.format(batch), quality=95, subsampling=0)
