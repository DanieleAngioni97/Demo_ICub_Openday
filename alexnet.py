import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Flatten(nn.Module):
    """Layer custom per reshape del tensore
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), 256 * 6 * 6)
        return x


class AlexNet(nn.Module):
    """Modello Pytorch di Alexnet pre-addestrato su ImageNet. Rispetto all'originale i layer sono contenuti
       in attributi separati, in modo da poter essere richiamati facilmente
    """

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = Flatten()
        self.dropout1 = nn.Dropout()
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(get_state_dict())
    return model


def get_state_dict():
    """Carica il modello pre-trained di imagenet cambiando i nomi dei layer
    """
    sub = {'features.0.weight': 'conv1.weight',
           'features.0.bias': 'conv1.bias',
           'features.3.weight': 'conv2.weight',
           'features.3.bias': 'conv2.bias',
           'features.6.weight': 'conv3.weight',
           'features.6.bias': 'conv3.bias',
           'features.8.weight': 'conv4.weight',
           'features.8.bias': 'conv4.bias',
           'features.10.weight': 'conv5.weight',
           'features.10.bias': 'conv5.bias',
           'classifier.1.weight': 'fc6.weight',
           'classifier.1.bias': 'fc6.bias',
           'classifier.4.weight': 'fc7.weight',
           'classifier.4.bias': 'fc7.bias',
           'classifier.6.weight': 'fc8.weight',
           'classifier.6.bias': 'fc8.bias'
           }
    old_state_dict = model_zoo.load_url(model_urls['alexnet'])
    new_state_dict = OrderedDict()
    for k in list(old_state_dict.keys()):
        new_state_dict[sub[k]] = old_state_dict.pop(k)
    return new_state_dict
