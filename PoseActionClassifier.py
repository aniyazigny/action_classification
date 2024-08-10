import torch.nn as nn
import torch.nn.functional as F
from PoseActionDataset import PoseActionDataset


class PoseActionClassifierLSTM(nn.Module):
    def __init__(self, num_classes, input_size=34, hidden_size=128, num_layers=2):
        super(PoseActionClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x is of shape (batch_size, seq_len, 17, 2)
        batch_size, seq_len, _, _ = x.shape
        x = x.view(batch_size, seq_len, -1)  # Flatten the last two dimensions (17, 2) into 34
        h_lstm, _ = self.lstm(x)  # LSTM output
        h_lstm_last = h_lstm[:, -1, :]  # Take the last output of the LSTM
        output = self.fc(h_lstm_last)
        return output

if __name__ == "__main__":
    # Example usage
    poses_dir = '/home/aniyazi/masters_degree/action_recognition/data/train/poses/'
    dataset = PoseActionDataset(poses_dir)
    num_classes = len(dataset.class_labels)
    model = PoseActionClassifierLSTM(num_classes)
