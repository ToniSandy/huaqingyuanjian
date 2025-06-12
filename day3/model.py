import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        # 使用预训练的ResNet18作为基础模型
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False

        # 替换最后的全连接层
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_model(num_classes, device='cuda'):
    model = ImageClassifier(num_classes=num_classes)
    return model.to(device)