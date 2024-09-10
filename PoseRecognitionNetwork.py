import torch
import torch.nn as nn

class PoseRecognitionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size, pretrained_lstm_path=None):
        super(PoseRecognitionNetwork, self).__init__()
        
        # LSTM layer - same structure as in the classifier network
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Load pretrained LSTM weights if a path is provided
        if pretrained_lstm_path is not None:
            self._load_pretrained_lstm(pretrained_lstm_path)
        
        # Linear layer for mapping LSTM output to the embedding vector
        self.fc = nn.Linear(hidden_size, embedding_size)

    def _load_pretrained_lstm(self, weights_path):
        # Load the entire model's state dict
        state_dict = torch.load(weights_path)
        
        # Extract only the LSTM part and remove the "lstm." prefix
        lstm_weights = {key.replace('lstm.', ''): value for key, value in state_dict.items() if 'lstm.' in key}
        
        # Load these weights into the LSTM layer
        self.lstm.load_state_dict(lstm_weights)
        print(f"Pretrained LSTM weights loaded from {weights_path}")


    def forward(self, x):
        # LSTM layer - output features and hidden state
        lstm_out, _ = self.lstm(x)  # LSTM output (batch_size, seq_len, hidden_size)
        
        # Take the last output of the LSTM (seq_len -> -1 for last)
        lstm_out_last = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layer to produce the embedding
        embedding = self.fc(lstm_out_last)  # (batch_size, embedding_size)
        
        return embedding

# Example usage:
if __name__ == "__main__":
    # Define hyperparameters
    input_size = 34  # For example, 17 keypoints * 2 (x, y)
    hidden_size = 128
    num_layers = 2
    embedding_size = 64  # Embedding vector size

    # Path to the pre-trained classifier weights
    pretrained_lstm_path = '/home/aniyazi/masters_degree/action_classification/runs/train/weights/best.pt'

    # Create the pose recognition network with pretrained LSTM weights
    model = PoseRecognitionNetwork(input_size, hidden_size, num_layers, embedding_size, pretrained_lstm_path=pretrained_lstm_path)

    # Example input: batch_size = 16, sequence_length = 30, input_size = 34
    example_input = torch.randn(16, 30, input_size)
    
    # Get the embedding output
    embedding_output = model(example_input)
    print("Embedding shape:", embedding_output.shape)  # Expected shape: (16, embedding_size)
