import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, in_channels=1):
        super(LeNet, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.avg_pool(x)
        x = self.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = self.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# output -> [64, 10]
x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)
