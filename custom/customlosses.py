import numpy as np
import torch
from torch import nn


class ConfidenceLoss(nn.Module):
    def __init__(self,
                 neg_pos_ratio: int = 4,
                 neg_for_hard: int = 100):
        super(ConfidenceLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio  # 正负样本比例 由正样本确定负样本数量
        self.neg_for_hard = neg_for_hard  # 如果正样本数量全部为零，则直接指定负样本个数为100
        # self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true):
        def softmax_loss(y_pred, y_true):
            '''
            This error is only applicable to the output activated by softmax
            :param y_pred:
            :param y_true: one hot label
            :return: softmax loss
            '''
            y_pred = torch.maximum(y_pred, torch.tensor(1e-7))
            softmax_loss = -torch.sum(y_true * torch.log(y_pred), dim=-1)
            return softmax_loss

        #  取出先验框的数量
        num_boxes = y_true.size(1)

        # softmax loss
        cls_loss = softmax_loss(y_pred=y_pred, y_true=y_true)
        # 也可以直接使用封装好的损失函数
        # cls_loss = self.loss_func(y_pred,y_true).sun(dim=-1)
        # 每张图的正样本个数 (batch_size,)
        num_pos = torch.sum(1 - y_true[..., 0], dim=-1)
        # 每张图片对应的正样本的分类置信度
        pos_conf_loss = torch.sum(cls_loss * (1 - y_true[..., 0]), dim=1)
        # 多数情况下大部分候选框都不包含检测物，导致负样本误差极大，易导致神经元死亡
        # 每张图片的负样本数量
        # (batch_size,)
        num_neg = torch.minimum(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        # 拿到每张图片是否存在负样本，得到bool mask
        # 如果num_pos均为0，则num_neg也均为0
        num_neg_by_pos_mask = torch.greater(num_neg, 0)
        # 如果num_pos均为0
        # 则默认选择100个先验框作为负样本
        has_min = torch.any(num_neg_by_pos_mask).float()  # 1.0或者0.0
        num_neg = torch.cat([num_neg, torch.tensor([(1 - has_min) * self.neg_for_hard]).to(torch.device('cuda'))],
                            dim=0)
        num_batch_neg = torch.sum(num_neg[torch.greater(num_neg, 0)]).int()

        # 把不是背景的概率求和，求和后的概率越大，代表越难分类(或者使用cls_loss也可以)
        # (batch_size,n)
        max_confs = torch.sum(y_pred[..., 1:], dim=-1)
        # 只有没有包含物体的先验框才得到保留
        # 我们在整个batch里面选取最难分类的num_batch_neg个先验框作为负样本
        max_confs = torch.reshape(max_confs * y_true[..., 0], shape=(-1,))
        indices = torch.topk(max_confs, k=num_batch_neg).indices
        # 根据索引，取出num_batch_neg个误差最大的负样本
        neg_conf_loss = torch.reshape(cls_loss, shape=(-1,))[indices]

        # 进行归一化
        # 避免num_pos全部为0，求和为0，导致数值溢出
        num_pos = torch.where(torch.not_equal(num_pos, 0), num_pos, torch.ones_like(num_pos))
        total_loss = (torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss)) / torch.sum(num_pos)
        return total_loss


class BBOXL1Loss(nn.Module):
    def __init__(self,
                 sigma: int = 1,
                 weights: int = 1):
        super(BBOXL1Loss, self).__init__()
        self.sigma_squared = sigma ** 2
        self.weights = weights  # BBOX loss在全部loss的权重

    def forward(self, y_pred, y_true):
        # y_pred shape: (batch_size,len(anchors),4)
        # y_true shape: (batch_size,len(anchors),5)
        regression_logits = y_pred[..., :4]
        regress_targets = y_true[..., :4]
        anchor_state = 1 - y_true[..., 4]  # y_true索引为4的位置包含是否为背景的状态
        # 取出作为正样本的先验框
        bool_mask = torch.eq(anchor_state, 1)  # torch.eq() 逐元素比较;torch.equal() 整体比较
        regression_logits = regression_logits[bool_mask]
        regress_targets = regress_targets[bool_mask]

        # 计算smooth L1 loss
        regression_diff = torch.abs(regression_logits - regress_targets)
        regression_loss = torch.where(torch.less(regression_diff, 1.0 / self.sigma_squared),
                                      0.5 * self.sigma_squared * torch.pow(regression_diff, 2),
                                      regression_diff - 0.5 / self.sigma_squared)

        normalizer = torch.maximum(torch.ones(size=()), torch.tensor(regression_diff.size(0)))
        normalizer = normalizer.float()
        loss = torch.sum(regression_loss) / normalizer

        return loss * self.weights


if __name__ == '__main__':
    mask = torch.tensor([True, True, False, False])
    min_mask = torch.any(mask).float()
    1
