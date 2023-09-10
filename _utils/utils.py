import numpy as np
import torch
from PIL import Image, ImageDraw


def get_classes_name(class_path):
    '''
    从class_path获取类别名称
    :param class_path:
    :return:
    '''
    with open(class_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_random_data(image_path, boxes, sg_points, sg_names, seg_class_name, input_shape):
    '''

    :param image_path:
    :param boxes:
    :param sg_points:
    :param sg_names:
    :param seg_class_name:
    :param input_shape:
    :return:
    '''
    image = Image.open(image_path)
    iw, ih = image.size
    h, w = input_shape

    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # 对语义分割需要的原始形状数据进行生成
    # 生成一个灰度图大小(384，640)，在灰度图上画sg_points区域
    sg_image = Image.new(mode='L', size=image.size)
    draw = ImageDraw.Draw(sg_image)
    for name, points in zip(sg_names, sg_points):
        # 取出的points是还未处理数据list,内部包含1个dict,关键字为vertices,types,closed
        # [{'vertices': [[398.03354, 439.209423], [406.767818, 469.15552], [544.020763, 439.209423], [678.778199, 422.98862], [811.040129, 413.006587], [677.530445, 415.502096], [536.534238, 426.731883], [398.03354, 439.209423]], 'types': 'LLCCCCCC', 'closed': True}]
        for point in points:
            # point为取出的字典
            # 通过数据可知，name为drivable area时closed为True,为lane时，closed为False
            closed = point['closed']
            vertices = np.array(point['vertices'], dtype='float')
            vertices = vertices.tolist()
            vertices = [tuple(pt) for pt in vertices]
            # 分别*10与*15的目的在于避免图像resize时, 闭环与非闭环分割区域的边界效应
            # uint8下值为1与2中间的像素将会被置为1, 导致图像分割时误判
            if closed:
                # 画出drivable area的封闭矩形区域
                draw.polygon(vertices, fill=seg_class_name.index(name) * 15)
            else:
                # 画出lane的线，宽度为13
                for i in range(len(vertices) - 1):
                    draw.line([vertices[i], vertices[i + 1]], fill=seg_class_name.index(name) * 10, width=13)
    del draw

    # 产生目标检测需要的数据
    image = image.resize(size=(nw, nh), resample=Image.BICUBIC)
    new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
    new_image.paste(image, box=(dx, dy))
    image = np.array(new_image, 'float') / 255.
    image = np.clip(image, 0., 1.)

    # 产生语义分割需要的数据
    sg_image = sg_image.resize(size=(nw, nh), resample=Image.BICUBIC)
    new_image = Image.new(mode='L', size=(w, h))
    new_image.paste(sg_image, box=(dx, dy))
    new_image = np.array(new_image, 'uint8')
    # 避免图片resize产生的边界效应，即在resample时在边界产生的不必要的值
    sg_image = np.zeros_like(new_image)
    bool_mask = np.logical_or(np.equal(new_image, seg_class_name.index('lane') * 10),
                              np.equal(new_image, seg_class_name.index('drivable area') * 15))
    sg_image[bool_mask] = new_image[bool_mask]
    sg_image[np.equal(sg_image, seg_class_name.index('lane') * 10)] = 1
    sg_image[np.equal(sg_image, seg_class_name.index('drivable area') * 15)] = 2

    # 产生目标检测需要的boxes
    # correct boxes
    if len(boxes) > 0:
        np.random.shuffle(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
        # boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dx
        # boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dy
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        # 剔除无效boxes
        boxes = boxes[np.logical_and(boxes_w > 1, boxes_h > 1)]

        boxes = boxes.astype('float')
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h

    return image, sg_image, boxes


class BBoxUtility(object):
    def __init__(self,
                 priors: np.ndarray,
                 num_classes: int,
                 overlap_threshold=0.5,
                 nms_thresh=0.5):
        self.priors = priors
        self.num_classes = num_classes
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.nms_thresh = nms_thresh

    @staticmethod
    def iou(box, priors):
        # 计算真实框与所有先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(priors[:, :2], box[:2])
        inter_botright = np.minimum(priors[:, 2:4], box[2:4])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框面积
        area_priors = (priors[:, 2] - priors[:, 0]) * (priors[:, 3] - priors[:, 1])
        union = area_true + area_priors - inter
        iou = inter / np.maximum(union, 1e-7)
        return iou

    def encode_box(self, box, reture_iou=True):
        iou = self.iou(box=box[:4], priors=self.priors)
        encoded_box = np.zeros(shape=(self.num_priors, 4 + reture_iou))
        # 找到每一个真实框，重合程度较高的先验框
        # 一个真实框可能对应多个先验框
        assign_mask = iou > self.overlap_threshold
        if not np.any(assign_mask):
            # 如果iou全部小于阈值，assign_mask全部为False
            # 取iou最大的先验框
            assign_mask[np.argmax(iou)] = True
        if reture_iou:
            # 将取出的iou存放在最后
            encoded_box[:, 4][assign_mask] = iou[assign_mask]
        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]

        # 逆向编码，将真实框转化为Retinaface预测结果的格式
        # 先计算真实框的中心和宽高
        box_center = 0.5 * (box[:2] + box[2:4])
        box_wh = box[2:4] - box[:2]

        # 再计算重合度较高的先验框的中心点和宽高
        assigned_priors_center = 0.5 * (assigned_priors[..., 0:2] + assigned_priors[..., 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] - assigned_priors[:, 0:2])

        # 逆向求取应该有的预测结果，根据decode_box函数过程相反
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= 0.1

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= 0.2
        # np.ravel()将多维数组变为一维
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        assignment = np.zeros(shape=(self.num_priors, 4 + 1 + self.num_classes))
        # 索引4位置代表是否为背景
        assignment[:, 4] = 1
        if len(boxes) == 0:
            return assignment
        # 每一个真实框的编码后的值，和iou
        # encoded_boxes : (n,num_priors,5+num_classes)
        # np.apply_along_axis()，将arr沿着axis应用func1d
        encoded_boxes = np.apply_along_axis(func1d=self.encode_box,
                                            axis=1,
                                            arr=boxes)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合度最大的先验框，并且获取这个先验框的index
        # (num_priors,)
        best_iou = encoded_boxes[:, :, 4].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, 4].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 依赖于numpy的双索引，得到唯一先验标签
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]

        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:][best_iou_mask] = boxes[best_iou_idx, 4:]
        return assignment

    @classmethod
    def correct_boxes(cls, boxes, input_shape: np.ndarray, image_shape: np.ndarray):
        '''
         When the input is added with letterboxes, restore the coordinates when outputting.
        Because the input size of each sample in a batch is different, only process single sample
        :param boxes:
        :param input_shape:
        :param image_shape:
        :return:
        '''
        new_shape = image_shape * np.min(input_shape / image_shape)

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        scale_for_boxs = np.tile(scale[::-1], (2,))  # [scale[1],scale[0],scale[1],scale[0]]
        offset_for_boxs = np.tile(offset[::-1], (2,))  # [offset[1],offset[0],offset[1].offset[0]]

        boxes = (boxes - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
        return boxes

    def decode_boxes(self, mbox_loc):
        # 获得先验框的宽高
        prior_wh = self.priors[:, 2:4] - self.priors[:, :2]

        # 获取先验框的中心点
        prior_center = 0.5 * (self.priors[:, :2] + self.priors[:, 2:4])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center = mbox_loc[:, 0:2] * prior_wh * 0.1
        decode_bbox_center += prior_center

        # 真实框的宽与高的求取
        decode_bbox_wh = np.exp(mbox_loc[:, 2:4] * 0.2)
        decode_bbox_wh *= prior_wh

        # 获取真实框的左上角和右下角
        decode_bbox_xy_min = decode_bbox_center - 0.5 * decode_bbox_wh
        decode_bbox_xy_max = decode_bbox_center + 0.5 * decode_bbox_wh

        # 真实框的左上角和右下角进行堆叠
        decode_bbox = np.concatenate([decode_bbox_xy_min, decode_bbox_xy_max], axis=-1)
        decode_bbox = np.clip(decode_bbox, 0., 1.)

        return decode_bbox

    def detection_out(self, predictions: list, conf_thresh=0.5, prob_thresh=0.4):
        # prediction为一个list,分为两部分,prediction[0]表示回归预测结果，prediction[1]表示分类预测结果
        mbox_loc = predictions[0]
        mbox_conf = predictions[1]
        total_boxes, total_scores, total_classes = [], [], []
        for i in range(mbox_loc.__len__()):
            # ===解码过程===
            decode_bbox = self.decode_boxes(mbox_loc[i])
            # mbox_conf 该矩阵包含两部分，索引为0的位置代表是否为背景，因此类别置信度从1开始取值
            total_class_conf = mbox_conf[i][..., 1:]

            class_conf = np.max(total_class_conf, axis=-1)[..., np.newaxis]
            class_pred = np.argmax(total_class_conf, axis=-1)[..., np.newaxis]
            # 判断置信度是否大于门限要求
            conf_mask = (class_conf >= conf_thresh)[:, 0]
            # 将预测结果进行堆叠
            detections = np.concatenate([decode_bbox[conf_mask],
                                         class_conf[conf_mask],
                                         class_pred[conf_mask]],
                                        axis=1)
            unique_class = np.unique(detections[:, -1])
            best_box, best_score, best_class = [], [], []
            if not unique_class.__len__():
                total_boxes.append(best_box)
                total_scores.append(best_score)
                total_classes.append(best_class)
                continue
            # =====
            # 对种类进行循环
            # 非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框
            # 对种类进行循环可以帮助我们对每一个类分别进行非极大抑制
            # =====
            for cls in unique_class:
                cls_mask = detections[:, -1] == cls
                detection = detections[cls_mask]
                scores = detection[:, 4]
                points = detection[:, :4]
                # 根据得分对该种类进行从大到小排序
                arg_sort = np.argsort(scores)[::-1]
                points = points[arg_sort]
                scores = scores[arg_sort]
                while np.shape(points)[0] > 0:
                    # =====
                    # 每次取出该类中得分最大的框，计算其与其它所有预测框的iou,iou过大则剔除
                    # =====
                    best_box.append(points[0])
                    best_score.append(scores[0])
                    best_class.append(cls)
                    if len(points) == 1:
                        break
                    ious = self.iou(best_box[-1], points[1:])
                    points = points[1:][ious < self.nms_thresh]
                    scores = scores[1:][ious < self.nms_thresh]

            total_boxes.append(best_box)
            total_scores.append(best_score)
            total_classes.append(best_class)
        return total_boxes, total_scores, total_classes


def calculate_f1score(y_true, y_pred, object_mask, num_classes: int):
    scores = []
    if object_mask.any():
        y_true = y_true[object_mask]  # 取出批量中所有包含目标的cells
        y_pred = y_pred[object_mask]
        true_class = y_true.argmax(dim=-1)
        pred_class = y_pred.argmax(dim=-1)
        # 遍历检测物体种类
        for i in range(num_classes):
            precision_num = torch.eq(pred_class, i).float().sum()
            # 如果预测包含某一类，计算分数
            if precision_num:
                # 某一类的掩码
                true_bool_mask = torch.eq(true_class, i)
                pred_bool_mask = torch.eq(pred_class, i)
                # 如果true_bool_mask和pred_bool_mask都为True，则bool_mask为True
                bool_mask = torch.stack([true_bool_mask, pred_bool_mask], dim=-1).all(dim=-1)
                true_positive_num = bool_mask.float().sum()
            else:
                continue
            false_negative_num = true_bool_mask.float().sum() - true_positive_num
            recall_num = true_positive_num + false_negative_num
            if recall_num:
                precision = true_positive_num / precision_num
                recall = true_positive_num / recall_num

            else:
                continue
            score = 2 * (precision * recall) / (precision + recall)
            scores.append(score)

    f1_score = torch.square(torch.mean(torch.tensor(scores)))
    return torch.zeros(size=()) if torch.isnan(f1_score) else f1_score
