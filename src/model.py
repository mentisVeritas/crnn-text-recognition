import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # H/2
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # H/2
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # H/2
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # RNN
        # After 3 MaxPool2d: H = 32/2/2/2 = 4, so features = 256 * 4 = 1024
        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )

        # FC
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        # x: [B, 1, 32, 128]
        x = self.cnn(x)  # [B, 256, H, W]

        b, c, h, w = x.size()

        # превращаем в последовательность
        x = x.permute(0, 3, 1, 2)   # [B, W, C, H]
        x = x.reshape(b, w, c * h)  # [B, W, features]

        # RNN
        x, _ = self.rnn(x)

        # классификация
        x = self.fc(x)  # [B, W, num_classes]

        return x