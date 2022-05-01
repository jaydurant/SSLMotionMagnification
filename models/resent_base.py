import torchvision.models as models
import torch
from models import MLPHead


class ResNet50(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__()

        resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)