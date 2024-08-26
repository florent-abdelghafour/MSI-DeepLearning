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


def ResNet18(**kwargs):
    return ResNet(ResidualBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(ResidualBlock, [3, 4, 6, 3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
     
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=14,num_classes=2,zero_init_residual=False,head='linear'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channel, self.in_planes, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(self.in_planes ),
                        nn.ReLU())
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))        
        
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
                
        
        if head=='linear':
            self.head = nn.Linear(512*block.expansion, num_classes)
            
        elif head == 'mlp':
            dim_in =512*block.expansion
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, num_classes)
            )
        
        
       
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * block.expansion
        return nn.Sequential(*layers) 
    
    model_dict = {
    'ResNet18': [ResNet18, 512],
    'ResNet34': [ResNet34, 512],
    'ResNet50': [ResNet50, 2048],
    'ResNet101': [ResNet101, 2048]}

        
     
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.head(out)
        return out    
       
    def predict(self, images):
            # Convert the images to PyTorch tensors and move them to the correct device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            image_tensors = (images).to(device)
    
            # Pass the images through the model to get predictions
            with torch.no_grad():
                self.eval()  # Set the model to evaluation mode
                outputs = self(image_tensors)
                probabilities = F.softmax(outputs, dim=1)  # Convert raw scores to probabilities
            return probabilities.cpu().numpy()  # Move predictions to CPU and convert to numpy array 
        
        