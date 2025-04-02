import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


model = resnet50(weights=ResNet50_Weights.DEFAULT)
# model.fc = nn.Linear(2048,20)
for name,param in model.named_parameters():
    if "fc." in name or "layer4." in name:
        pass
    else:
        param.requires_grad = False
# print(model.fc)
# image = torch.rand(2,3,224,224)
# output = model(image)
# print(output.shape)

class MyResNet(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        del self.model.fc
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

if __name__ == '__main__':
    model = MyResNet()
    image = torch.rand(2,3,224,224)
    output = model(image)
    print(output.shape)