import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class IntroDetectionModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )

        self.fc_out = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        x, _ = self.lstm(x)
        return self.fc_out(x)