import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution: keeps channel count unchanged (in_channels → in_channels)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   groups=in_channels, bias=False)
        # Pointwise convolution: maps from in_channels to out_channels (e.g. 14 → 64)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)

    
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=14, num_classes=2, head='linear',in_planes=64):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channel, self.in_planes, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1)
        )
        
        
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*self.in_planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*self.in_planes, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*self.in_planes, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if head == 'linear':
            self.head = nn.Linear(8*self.in_planes * block.expansion, num_classes)
        elif head == 'mlp':
            dim_in = 8*self.in_planes * block.expansion
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, num_classes)
            )
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.head(out)

# Define ResNet variants

def MSI_ResNet18(**kwargs):
    return ResNet(ResidualBlock, [2, 2, 2, 2], **kwargs)

def MSI_ResNet34(**kwargs):
    return ResNet(ResidualBlock, [3, 4, 6, 3], **kwargs)

def MSI_ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def MSI_ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
