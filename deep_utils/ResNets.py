import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)              
                      
        self.conv2 =  nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)                
        
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
                      
   
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
     
        return out
     
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
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
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out
     
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=14,num_classes=2,zero_init_residual=False,head_type='linear',in_planes = 64):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.initial_planes = in_planes  
        self.head_type=head_type
        self.num_classes= num_classes

        
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channel, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(self.in_planes),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1) )
        
        
        self.layer1 = self._make_layer(block, self.initial_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * self.initial_planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.initial_planes, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * self.initial_planes, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   
        
        self.head_dim = self.initial_planes * 8 * block.expansion
        if self.head_type == 'linear':
            self.head = nn.Linear(self.head_dim, self.num_classes)
        elif self.head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.ReLU(),
                nn.Linear(self.head_dim, self.num_classes)
            )
    
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                
       
       
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * block.expansion
        return nn.Sequential(*layers) 
        
     
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Final ResNet block
        x = self.avgpool(x)  # Apply Global Average Pooling
        x = torch.flatten(x, 1)  # Flatten before MLP head
        
        return self.head(x)
  
def ResNet18(**kwargs):
    return ResNet(ResidualBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(ResidualBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs) 
        