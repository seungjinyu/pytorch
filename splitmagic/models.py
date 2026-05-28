from torchvision.models import resnet18
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LongCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu4 = nn.ReLU()

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.relu6 = nn.ReLU()

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.relu4(self.conv4(x))

        x = x.flatten(1)

        x = self.relu5(self.fc1(x))
        x = self.relu6(self.fc2(x))
        x = self.fc3(x)

        return x
    
class TinyBasicBlockNoBN(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        out = out + identity
        out = self.relu2(out)

        return out


class TinyResNetNoBN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.relu0 = nn.ReLU()

        self.block1 = TinyBasicBlockNoBN(16)
        self.block2 = TinyBasicBlockNoBN(16)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.relu0(self.stem(x))

        x = self.block1(x)
        x = self.block2(x)

        x = self.pool(x)
        x = x.flatten(1)

        x = self.fc(x)

        return x
    
    
import torch.nn as nn


def make_resnet18_cifar10():
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    # inplace ReLU 끄기
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    return model