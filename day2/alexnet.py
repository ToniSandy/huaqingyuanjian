import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):  # CIFAR10是10分类任务
        super().__init__()
        self.features = nn.Sequential(
            # 输入: [3, 224, 224]
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # [64, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                 # [64, 27, 27]
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # [192, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # [192, 13, 13]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),          # [384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # [256, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平为 [batch_size, 256*6*6]
        x = self.classifier(x)
        return x

# 测试代码
if __name__ == "__main__":
    model = AlexNet()
    dummy_input = torch.randn(1, 3, 224, 224)  # 标准AlexNet输入尺寸
    output = model(dummy_input)
    print(f"模型输出尺寸: {output.shape}")  # 应为 [1, 10]