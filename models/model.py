import torch
import torch.nn as nn


class PoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseNet, self).__init__()
        # 기본적인 CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 224x224 이미지 기준 feature map 크기 계산 (예: 56x56) → 실제 이미지 크기에 따라 수정 필요
        self.regressor = nn.Sequential(
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 3)  # 각 keypoint에 대해 x, y 좌표 예측
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.regressor(x)
        return x
