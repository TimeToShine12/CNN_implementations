import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(ConvBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=intermediate_channels)
        self.conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=intermediate_channels)
        self.conv3 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels * self.expansion,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(num_features=intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)

        # skip connection
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ConvBlock, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layers1 = self._make_layer(ConvBlock, layers[0], intermediate_channels=64, stride=1)
        self.layers2 = self._make_layer(ConvBlock, layers[1], intermediate_channels=128, stride=2)
        self.layers3 = self._make_layer(ConvBlock, layers[2], intermediate_channels=256, stride=2)
        self.layers4 = self._make_layer(ConvBlock, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512*4, out_features=num_classes)

    def _make_layer(self, ConvBlock, num_res_blocks, intermediate_channels, stride):
        layers = []
        identity_downsample = None

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                                          out_channels=intermediate_channels * 4,
                                                          kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(num_features=intermediate_channels * 4))

        layers.append(ConvBlock(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4

        for i in range(num_res_blocks - 1):
            layers.append(ConvBlock(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)

        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x

# testing resnet50 -> output should be [2, 1000]
def resnet50(img_channels=3, num_classes=1000):
    return ResNet(ConvBlock, layers=[3, 4, 6, 3], image_channels=img_channels, num_classes=num_classes)

model = resnet50()
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(y.shape)
