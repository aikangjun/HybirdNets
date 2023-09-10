# 获取真实框处理后的数据,生成多尺度定位框
import itertools

import numpy as np


class Anchors:
    def __init__(self,
                 scales: np.ndarray,
                 ratios: list,
                 anchor_scale=4.,
                 pyramid_levels=None):
        '''

        :param scales:
        :param ratios:
        :param anchor_scale:float number representing the scale of size of the base
            anchor to the feature stride 2^level.
        :param pyramid_levels: 5 receptive fields by default
        '''
        self.anchor_scale = anchor_scale
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.scales = scales
        self.ratios = ratios
        self.strides = [2 ** x for x in self.pyramid_levels]

    def __call__(self, image_shape, *args, **kwargs):
        '''
        Generates multiscale anchor boxes.
        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
        Returns:
          anchor_boxes: a numpy array with shape [1, N, 4], which stacks anchors on all
            feature levels.
        :param image_shape:
        :param args:
        :param kwargs:
        :return:
        '''
        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                # itertools.product()根据可迭代对象生成笛卡尔积；与zip不同，zip只根据索引一一对应
                if image_shape[1] % stride != 0 and image_shape[0] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x = base_anchor_size * ratio[0] / 2.
                anchor_size_y = base_anchor_size * ratio[1] / 2.

                x = np.arange(start=stride // 2, stop=image_shape[1], step=stride)
                y = np.arange(start=stride // 2, stop=image_shape[0], step=stride)
                ct_x, ct_y = np.meshgrid(x, y, indexing='xy')

                ct_yx = np.stack([ct_y, ct_x], axis=-1)
                ct_yx = np.tile(ct_yx, (2,)).astype('float')

                anchor_size = np.tile(np.array([anchor_size_y, anchor_size_x]),
                                      reps=2).astype('float')
                anchor_size[:2] = -anchor_size[:2]

                boxes = ct_yx + anchor_size
                boxes = boxes.reshape((-1, 4))
                boxes_level.append(boxes[:, np.newaxis])  # (n,1,4)
            # 将3个scale个aspect ratio固定的anchor在axis=1进行叠加
            boxes_level = np.concatenate(boxes_level, axis=1)  # (n,3,4)
            boxes_all.append(boxes_level.reshape((-1, 4)))  # (n*3,4)
        # boxes_all 是长度为5的list,因为有五个感受野
        anchor_boxes = np.concatenate(boxes_all, axis=0)

        # 归一化操作
        anchor_boxes[:, 0::2] /= image_shape[1]  # x/w
        anchor_boxes[:, 1::2] /= image_shape[0]  # y/h
        return anchor_boxes


if __name__ == '__main__':
    priors = Anchors(scales=[1, 1.25, 1.60],
                     ratios=[[1, 1], [1.4, 0.7], [0.7, 1.4]])((384, 640))
    1
