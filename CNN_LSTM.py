import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=3, sequence_length=10, num_classes=2):
        super(CNNLSTM, self).__init__()
        
        # CNN layers with Batch Normalization
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Dynamically calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size(input_channels, 224, 224)
        
        # LSTM layers
        self.lstm = nn.LSTM(self.cnn_output_size, 128, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),  # Bidirectional LSTM outputs 128 * 2
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def _get_cnn_output_size(self, channels, height, width):
        with torch.no_grad():
            dummy_input = torch.randn(1, channels, height, width)
            cnn_output = self.cnn(dummy_input)
            return cnn_output.view(-1).size(0)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(-1, C, H, W)  # Reshape for CNN
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape for LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the last time-step output
        x = self.fc(x)
        return x

# Initialize the model
model = CNNLSTM(input_channels=3, sequence_length=10, num_classes=2)
