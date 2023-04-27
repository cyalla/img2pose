import torch
from torch.nn import DataParallel, Module
from torch.nn.parallel import DistributedDataParallel
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn as nn
from model_loader import load_model
from models import FasterDoFRCNN
from efficientnet_pytorch import EfficientNet

class WrappedModel(Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, images, targets=None):
        return self.module(images, targets)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class img2poseModel:
    def __init__(
        self,
        depth,
        min_size,
        max_size,
        model_path=None,
        device=None,
        pose_mean=None,
        pose_stddev=None,
        distributed=False,
        gpu=0,
        threed_68_points=None,
        threed_5_points=None,
        rpn_pre_nms_top_n_test=6000,
        rpn_post_nms_top_n_test=1000,
        bbox_x_factor=1.1,
        bbox_y_factor=1.1,
        expand_forehead=0.3,
    ):
        self.depth = depth
        self.min_size = min_size
        self.max_size = max_size
        self.model_path = model_path
        self.distributed = distributed
        self.gpu = gpu

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # create network backbone
        # backbone = resnet_fpn_backbone(f"resnet{self.depth}", pretrained=True)

        backbone = nn.Sequential(
                # [from, repeats, module, args]
                Conv(-1, 64, 3, 2),  # 0-P1/2
                Conv(-1, 128, 3, 2),  # 1-P2/4
                C2f(-1, 128, True, 3),  # 2
                C2f(-1, 128, True, 3),  # 3
                C2f(-1, 128, True, 3),  # 4
                Conv(-1, 256, 3, 2),  # 5-P3/8
                C2f(-1, 256, True, 3),  # 6
                C2f(-1, 256, True, 3),  # 7
                C2f(-1, 256, True, 3),  # 8
                C2f(-1, 256, True, 3),  # 9
                C2f(-1, 256, True, 3),  # 10
                C2f(-1, 256, True, 3),  # 11
                Conv(-1, 512, 3, 2),  # 12-P4/16
                C2f(-1, 512, True, 3),  # 13
                C2f(-1, 512, True, 3),  # 14
                C2f(-1, 512, True, 3),  # 15
                C2f(-1, 512, True, 3),  # 16
                C2f(-1, 512, True, 3),  # 17
                C2f(-1, 512, True, 3),  # 18
                Conv(-1, 1280, 3, 2),  # 19-P5/32
                C2f(-1, 1024, True, 3),  # 20
                C2f(-1, 1024, True, 3),  # 21
                C2f(-1, 1024, True, 3),  # 22
                SPPF(-1, 1024, 5),  # 23
            )

        backbone.out_channels = 1280
        
        # # Define the EfficientNet backbone
        # print("EfficientNet")
        # conv_stem = torch.nn.Sequential(EfficientNet.from_pretrained('efficientnet-b0')._conv_stem)
        # bn = torch.nn.Sequential(EfficientNet.from_pretrained('efficientnet-b0')._bn0)
        # blocks = torch.nn.Sequential(*EfficientNet.from_pretrained('efficientnet-b0')._blocks)
        # conv_head = torch.nn.Sequential(EfficientNet.from_pretrained('efficientnet-b0')._conv_head)
        # backbone = torch.nn.Sequential(conv_stem, bn, blocks, conv_head)
        # backbone.out_channels = 1280
        # print(backbone)

        if pose_mean is not None:
            pose_mean = torch.tensor(pose_mean)
            pose_stddev = torch.tensor(pose_stddev)

        if threed_68_points is not None:
            threed_68_points = torch.tensor(threed_68_points)

        if threed_5_points is not None:
            threed_5_points = torch.tensor(threed_5_points)

        # create the feature pyramid network
        self.fpn_model = FasterDoFRCNN(
            backbone,
            2,
            min_size=self.min_size,
            max_size=self.max_size,
            pose_mean=pose_mean,
            pose_stddev=pose_stddev,
            threed_68_points=threed_68_points,
            threed_5_points=threed_5_points,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            bbox_x_factor=bbox_x_factor,
            bbox_y_factor=bbox_y_factor,
            expand_forehead=expand_forehead,
        )

        # if using cpu, remove the parallel modules from the saved model
        self.fpn_model_without_ddp = self.fpn_model

        if self.distributed:
            self.fpn_model = self.fpn_model.to(self.device)
            self.fpn_model = DistributedDataParallel(
                self.fpn_model, device_ids=[self.gpu]
            )
            self.fpn_model_without_ddp = self.fpn_model.module

            print("Model will use distributed mode!")

        elif str(self.device) == "cpu":
            self.fpn_model = WrappedModel(self.fpn_model)
            self.fpn_model_without_ddp = self.fpn_model

            print("Model will run on CPU!")

        else:
            self.fpn_model = DataParallel(self.fpn_model)
            self.fpn_model = self.fpn_model.to(self.device)
            self.fpn_model_without_ddp = self.fpn_model

            print(f"Model will use {torch.cuda.device_count()} GPUs!")

        if self.model_path is not None:
            self.load_saved_model(self.model_path)
            self.evaluate()

    def load_saved_model(self, model_path):
        load_model(
            self.fpn_model_without_ddp, model_path, cpu_mode=str(self.device) == "cpu"
        )

    def evaluate(self):
        self.fpn_model.eval()

    def train(self):
        self.fpn_model.train()

    def run_model(self, imgs, targets=None):
        outputs = self.fpn_model(imgs, targets)

        return outputs

    def forward(self, imgs, targets):
        losses = self.run_model(imgs, targets)

        return losses

    def predict(self, imgs):
        assert self.fpn_model.training is False

        with torch.no_grad():
            predictions = self.run_model(imgs)

        return predictions
