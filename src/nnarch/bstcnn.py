import torch
import torch.nn as nn

class BstCnn(nn.Module):
    def __init__(self, num_classes=100):
        super(BstCnn, self).__init__()

        self.num_classes = num_classes

        self.conv5x5_1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )

        self.conv3x3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )

        self.conv3x3_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(35, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(35, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )

        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((50, 50))

        self.fc = nn.Sequential(
            nn.Linear(16*50*50, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        out_5x5 = self.conv5x5_1(x)
        out_3x3 = self.conv3x3_1(x)
        output = torch.cat([out_5x5, out_3x3, x], 1)
        output = self.maxpool(output)
        
        out_3x3 = self.conv3x3_2(output)
        out_1x1 = self.conv1x1_2(output)
        output = torch.cat([out_3x3, out_1x1], 1)
        output = self.maxpool(output)

        out_1x1 = self.conv1x1_3(output)
        output = self.avgpool(out_1x1)

        output = output.reshape(output.size(0), -1)
        output = self.fc(output)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
