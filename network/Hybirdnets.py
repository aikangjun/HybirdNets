import timm
from torch import nn
from custom.customlayers import BiFPN, BiFPNDecoder, Classifer, Regressor, SegmentatiomHead


class Hybridnets(nn.Module):
    def __init__(self,
                 backbone: str,
                 fpn_cells: int,
                 out_indices: tuple,
                 conv_channels: list,
                 num_features: int,
                 num_anchors: int,
                 num_classes: int,
                 num_layers: int,
                 num_seg_classes: int,
                 up_scale: tuple):
        super(Hybridnets, self).__init__()
        assert out_indices.__len__() == conv_channels.__len__()
        self.backbone = timm.create_model(model_name=backbone,
                                          pretrained=True,
                                          features_only=True,
                                          out_indices=out_indices)  # p2,p3,p4,p5

        self.bifpn = nn.Sequential(*[BiFPN(num_features=num_features,
                                           conv_channels=conv_channels[1:] if not i else None,
                                           attention=True,
                                           first_time=True if not i else False)
                                     for i in range(fpn_cells)])

        self.bifpn_decoder = BiFPNDecoder(in_channels=num_features,
                                          out_channels=num_features,
                                          embed_channels=conv_channels[0])  # embed_channels表示p2的通道数

        self.classifer = Classifer(num_features=num_features,
                                   num_anchors=num_anchors,
                                   num_classes=num_classes,
                                   num_layers=num_layers)

        self.regessor = Regressor(num_features=num_features,
                                  num_anchors=num_anchors,
                                  num_layers=num_layers)

        self.segmentation_head = SegmentatiomHead(in_channels=num_features,
                                                  out_channels=num_seg_classes,
                                                  scale_facter=up_scale)

    def forward(self, input):
        p2, p3, p4, p5 = self.backbone(input)
        p3, p4, p5, p6, p7 = self.bifpn([p3, p4, p5])
        feats = self.bifpn_decoder([p2, p3, p4, p5, p6, p7])

        segmentations = self.segmentation_head(feats)

        classifications = self.classifer([p3, p4, p5, p6, p7])

        regressions = self.regessor([p3, p4, p5, p6, p7])
        return regressions, classifications, segmentations

    def initialize_decoder(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initialize_header(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def split_weights(self):
        weights, bias = [], []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    weights.append(param)
                elif name.split('.')[-1] == 'bias':
                    bias.append(param)
                else:
                    weights.append(param)
        return weights, bias


if __name__ == '__main__':
    import configure.config as cfg
    import torch

    hyb = Hybridnets(backbone=cfg.backbone,
                     fpn_cells=cfg.fpn_cells,
                     out_indices=cfg.out_indices,
                     conv_channels=cfg.conv_channels,
                     num_features=cfg.num_features,
                     num_anchors=cfg.num_anchors,
                     num_classes=cfg.class_names.__len__() + 1,
                     num_layers=cfg.num_layers,
                     num_seg_classes=cfg.segmentation_class_names.__len__(),
                     up_scale=cfg.up_scale)
    inp = torch.randn(size=(2, 3, 384, 640))
    regressions, classifications, segmentations = hyb(inp)
    1
