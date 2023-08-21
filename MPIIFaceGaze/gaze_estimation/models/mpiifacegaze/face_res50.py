import torch.nn as nn
from .modules import resnet50

class gaze_network(nn.Module):
    def __init__(self, in_stride=2):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet50(pretrained=True, in_stride=in_stride)
        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2)
        )

    def forward(self, face):
        feature = self.gaze_network(face)
        feature = feature.view(feature.size(0), -1)

        gaze = self.gaze_fc(feature)
        return gaze
