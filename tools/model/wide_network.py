import torch.nn.functional
from torchvision import transforms

from model.mynn import *
from model.wide_resnet_base import WiderResNetA2

Norm2d = torch.nn.BatchNorm2d


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, norm_act=nn.BatchNorm2d):
        super(ProjectionHead, self).__init__()
        # tf zhou / shikun liu
        self.proj = nn.Sequential(
            # nn.Conv2d(dim_in, dim_in, kernel_size=1),
            # norm_act(dim_in),
            # norm_act(dim_in, affine=False),
            # nn.ReLU(),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        )
        # initialize_weights(self.proj)

    def forward(self, x):
        return self.proj(x)


class ResidualAnomalyBlock(nn.Module):
    def __init__(self):
        super(ResidualAnomalyBlock, self).__init__()
        self.atten_aspp = _AtrousSpatialPyramidPoolingModule(4096, 256, output_stride=8)  
        self.bot_atten_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.atten_aspp_final = nn.Conv2d(256, 304, kernel_size=1, bias=False)
        self.proj_head = ProjectionHead(dim_in=256, proj_dim=304)

        initialize_weights(self.atten_aspp)
        initialize_weights(self.bot_atten_aspp)
        initialize_weights(self.atten_aspp_final)

        self.fine_tune_layers = []
        self.fine_tune_layers.append(self.atten_aspp)
        self.fine_tune_layers.append(self.bot_atten_aspp)
        self.fine_tune_layers.append(self.proj_head)

        self.retrain_layers = []
        self.retrain_layers.append(self.atten_aspp_final)
        # self.retrain_layers.append(self.proj_head)

    def forward(self, x, upsample_size):
        x = self.atten_aspp(x)
        x = self.bot_atten_aspp(x)
        x = Upsample(x, upsample_size)
        res_ = self.atten_aspp_final(x)
        if not self.training:
            return res_, None
        proj_ = self.proj_head(x)
        return res_, proj_


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features
        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepWV3Plus(nn.Module):
    """
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, trunk='WideResnet38'):

        super(DeepWV3Plus, self).__init__()
        wide_resnet = WiderResNetA2(structure=[3, 3, 6, 3, 1, 1], classes=1000, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        wide_resnet = wide_resnet.module
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet
        self.residual_block = ResidualAnomalyBlock()

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.train_module_list = ["branch1.residual_block.atten_aspp", "branch1.residual_block.proj_head",
                                  "branch1.residual_block.bot_atten_aspp"]

    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = inp.unsqueeze(0).cuda()
        assert len(inp.shape) == 4  # (B, C, W, H)
        x_size = inp.size()
        x = self.mod1(inp)  
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x_in_ = self.mod7(x)
        ##wider_resnet above 
        res_, proj_ = self.residual_block(x_in_, upsample_size=m2.size()[2:])
        x_out_ = self.aspp(x_in_)
        dec0_up = self.bot_aspp(x_out_)

        dec0_fine = self.bot_fine(m2)
        dec0_up = Upsample(dec0_up, m2.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final(dec0)
        dec2 = self.final(dec0 + res_)

        out1 = Upsample(dec1, x_size[2:])
        out2 = Upsample(dec2, x_size[2:])

        return out1, out2, proj_

