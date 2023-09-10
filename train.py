import torch
from _utils.anchors import Anchors
import configure.config as cfg
from hybridnetsModel import HybridnetsModel
from _utils.generate import Generator

if __name__ == '__main__':
    priors = Anchors(scales=cfg.scales,
                     ratios=cfg.ratios)(cfg.input_size)

    model = HybridnetsModel(fpn_cells=cfg.fpn_cells,
                            num_layers=cfg.num_layers,
                            num_anchors=cfg.num_anchors,
                            num_classes=cfg.class_names.__len__() + 1,
                            seg_classes=cfg.segmentation_class_names.__len__(),
                            num_features=cfg.num_features,
                            conv_channels=cfg.conv_channels,
                            out_indices=cfg.out_indices,
                            up_scale=cfg.up_scale,
                            backbone=cfg.backbone,
                            priors=priors,
                            learning_rate=cfg.learning_rate,
                            weight_decay=cfg.weight_decay,
                            iou_thresh=cfg.iou_thresh,
                            nms_thresh=cfg.nms_thresh,
                            resume_train=cfg.resume_train,
                            ckpt_path=cfg.ckpt_path + '//Epoch006_train_loss4.372_train_acc89.564.pth.tar',
                            device=cfg.device)
    data_gen = Generator(image_root=cfg.image_root,
                         anno_path=cfg.annotation_path,
                         input_size=cfg.input_size,
                         batch_size=cfg.batch_size,
                         train_split=cfg.train_split,
                         priors=priors,
                         num_classes=cfg.class_names.__len__())
    train_gen = data_gen.generate(trianing=True)
    val_gen = data_gen.generate(trianing=False)
    for epoch in range(cfg.Epochs):
        for i in range(data_gen.get_train_len()):
            sources, seg_sources, targets = next(train_gen)
            model.train(sources, seg_sources, targets)
            if (i + 1) % cfg.per_sample_interval == 0:
                model.generate_sample(sources, i + 1, cfg.train_sample_path)

        print('Epoch:{:0>3d}\ttrain_loss:{:.3f}\ttrain_acc:{:.3%}\t'
              'train_conf:{:.3%}\ttrain_f1score:{:.3%}'
              .format(epoch + 1,
                      model.train_loss / (i + 1),
                      model.train_acc / (i + 1),
                      model.train_conf_acc / (i + 1),
                      model.train_f1_score / (i + 1)))
        torch.save(obj={'state_dict': model.network.state_dict(),
                        'loss': model.train_loss / (i + 1),
                        'acc': model.train_acc / (i + 1) * 100},
                   f=cfg.ckpt_path + '\\Epoch{:0>3d}_train_loss{:.3f}_train_acc{:.3f}.pth.tar'.format(
                       epoch + 1, model.train_loss / (i + 1), model.train_acc / (i + 1) * 100,
                       model.train_acc / (i + 1) * 100
                   ))
        model.train_loss = 0
        model.train_acc = 0
        model.train_conf_acc = 0
        model.train_f1_score = 0

        for i in range(data_gen.get_val_len()):
            sources, seg_sources, targets = next(val_gen)
            model.validate(sources, seg_sources, targets)
            if (i + 1) % cfg.per_sample_interval == 0:
                model.generate_sample(sources, i + 1, cfg.val_sample_path)
        print('Epoch{:0>3d} '
              'validate loss is {:.3f} '
              'validate acc is {:.3f}% '
              'validate conf acc is {:.3f}% '
              'validate f1 score is {:.3f}% '.format(epoch + 1,
                                                     model.val_loss / (i + 1),
                                                     model.val_acc / (i + 1) * 100,
                                                     model.val_conf_acc / (i + 1) * 100,
                                                     model.val_f1_score / (i + 1) * 100))
        model.val_loss = 0
        model.val_acc = 0
        model.val_conf_acc = 0
        model.val_f1_score = 0
