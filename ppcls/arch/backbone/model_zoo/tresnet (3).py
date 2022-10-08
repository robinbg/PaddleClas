import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppcls.utils.initializer import kaiming_normal_, ones_, zeros_, normal_
from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

from typing import Optional
from functools import partial
from collections import OrderedDict

MODEL_URLS = {"TResNet-M":""}

        
class Downsample(nn.Layer):
    def __init__(self, filt_size=3, stride=2, channels=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels


        assert self.filt_size == 3
        a = paddle.to_tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :])
        self.filt = filt / paddle.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].tile(paddle.to_tensor([self.channels,1,1,1])))



    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


class SpaceToDepth(nn.Layer):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape((N, C, H // self.bs, self.bs, W // self.bs, self.bs))  # (N, C, H//bs, bs, W//bs, bs)
        x = x.transpose((0, 3, 5, 1, 2, 4))  # (N, bs, bs, C, H//bs, W//bs)
        x = x.reshape((N, C * (self.bs ** 2), H // self.bs, W // self.bs))  # (N, C*bs^2, H//bs, W//bs)
        return x


class DepthToSpace(nn.Layer):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape((N, self.bs, self.bs, C // (self.bs ** 2), H, W))  # (N, bs, bs, C//bs^2, H, W)
        x = x.transpose((0, 3, 4, 1, 5, 2))  # (N, C//bs^2, H, bs, W, bs)
        x = x.reshape((N, C // (self.bs ** 2), H * self.bs, W * self.bs))  # (N, C//bs^2, H * bs, W * bs)
        return x

class FastGlobalAvgPool2d(nn.Layer):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.shape
            return x.reshape((in_size[0], in_size[1], -1)).mean(axis=2)
        else:
            return x.reshape((x.shape[0], x.shape[1], -1)).mean(axis=-1).reshape((x.shape[0], x.shape[1], 1, 1))

class SEModule(nn.Layer):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastGlobalAvgPool2d()
        self.fc1 = nn.Conv2D(channels, reduction_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(reduction_channels, channels, kernel_size=1, padding=0)
        # self.activation = hard_sigmoid(inplace=inplace)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se



"""class Conv2DABN(nn.Layer):
    def __init__(self, ni, nf, stride, activation="leaky_relu", kernel_size = 3, activation_param=1e-2, groups=1):
        super(Conv2DABN, self).__init__()
        self.conv = nn.Conv2D(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=1, bias_attr=False)
        self.batch_norm = nn.BatchNorm(nf, act=None, param_attr=nn.initializer.Constant(), in_place=True)
        self.act = nn.LeakyReLU(activation_param)
        self.activation = activation
    def forward(self, input):
        input = self.batch_norm(self.conv(input))
        if self.activation == "identity":
            return input
        else:
            return self.act(input)"""

def Conv2DABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2D(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=1, bias_attr=False),
        nn.BatchNorm(nf, act=None, param_attr=nn.initializer.Constant(), in_place=True, momentum=0.9),
        nn.LeakyReLU(activation_param) if activation=="leaky_relu" else nn.Identity()    
    )
class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = Conv2DABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = Conv2DABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                aal = Downsample(channels=planes, filt_size=3, stride=2)
                self.conv1 = nn.Sequential(Conv2DABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           aal)

        self.conv2 = Conv2DABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2DABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = Conv2DABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = Conv2DABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                aal = Downsample(channels=planes, filt_size=3, stride=2)
                self.conv2 = nn.Sequential(Conv2DABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           aal)

        self.conv3 = Conv2DABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(nn.Layer):

    def __init__(self, layers, in_chans=3, class_num=1000, width_factor=1.0, remove_aa_jit=False):
        super(TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepth()
        anti_alias_layer = Downsample
        global_pool_layer = FastGlobalAvgPool2d(flatten=True)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = Conv2DABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        # body
        self.body = nn.Sequential(space_to_depth, conv1, layer1, layer2, layer3, layer4)
       
        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(global_pool_layer)
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        fc = nn.Linear(self.num_features, class_num)
        self.head = nn.Sequential(fc)

        self.apply(self._init_weight)
        self.apply(self._init_residual)

    def _init_weight(self, m):
        if isinstance(m, nn.Conv2D):
            kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2D) or isinstance(m, nn.BatchNorm):
            ones_(m.weight)
            zeros_(m.bias)
    
    def _init_residual(self, m):
        if isinstance(m, BasicBlock):
            zeros_(m.conv2[1].weight)
        if isinstance(m, Bottleneck):
            zeros_(m.conv3[1].weight)
        if isinstance(m, nn.Linear):
            normal_(m.weight, mean=0, std=0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2D(kernel_size=2, stride=2, ceil_mode=True))
            layers += [Conv2DABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
   #     print(type(x))
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits

def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def TResNet_M(pretrained=False, use_ssld=False, **kwargs):
    """ Constructs a medium TResnet model.
    """
    in_chans = 3
    model = TResNet(layers=[3, 4, 11, 3], in_chans=in_chans, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["TResNet-M"], use_ssld=use_ssld)
    return model


def TResNet_L(pretrained=False, use_ssld=False, **kwargs):
    """ Constructs a large TResnet model.
    """
    in_chans = 3
    model = TResNet(layers=[4, 5, 18, 3], in_chans=in_chans, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["TResNet-L"], use_ssld=use_ssld)
    return model


def TResNet_XL(pretrained=False, use_ssld=False, **kwargs):
    """ Constructs an extra-large TResnet model.
    """
    in_chans = 3
    model = TResNet(layers=[4, 5, 24, 3], in_chans=in_chans, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["TResNet-XL"], use_ssld=use_ssld)
    return model