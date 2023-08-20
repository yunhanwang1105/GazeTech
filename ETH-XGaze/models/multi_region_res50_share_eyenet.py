import torch.nn as nn
import torch
from .resnet import resnet50

class gaze_network(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1, in_stride=2):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet50(pretrained=True, in_stride=in_stride)
        self.gaze_network_two_eye = resnet50(pretrained=True, in_stride=in_stride)

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048 * 3, 2),
        )

    def forward(self, left_eye, right_eye, face):

        feature_left_eye = self.gaze_network_two_eye(left_eye) # share resnet
        feature_right_eye = self.gaze_network_two_eye(right_eye) # share resnet

        feature_face = self.gaze_network(face)
        feature = torch.cat((feature_left_eye, feature_right_eye, feature_face), 1)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze