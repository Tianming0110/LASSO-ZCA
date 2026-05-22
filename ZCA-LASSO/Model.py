"""
Model -

Author: Worromot
Date: 2026/5/8
"""

import torch
import torch.nn as nn


class TE_MLP(nn.Module):
    """
    Fully Connected Neural Network (Multi-Layer Perceptron)
    Most suitable for current static features without time dependencies (B, 52)
    """

    def __init__(self, input_dim=52, num_classes=22):
        super(TE_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 52)
        return self.net(x)


class TE_1DCNN(nn.Module):
    """
    One-Dimensional Convolutional Neural Network (1D CNN)
    Treats the 52 sensor features as a spatial sequence of length 52 for convolutional feature extraction
    """

    def __init__(self, input_dim=52, num_classes=22):
        super(TE_1DCNN, self).__init__()
        # Input shape requirement: (B, Channels, Length)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),  # Add lightweight Dropout after convolution layer

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)  # Add lightweight Dropout after convolution layer
        )
        # After two MaxPools, length 52 -> 26 -> 13. Feature dimension: 32 * 13 = 416
        flatten_dim = 64 * (input_dim // 4)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 64),
            nn.BatchNorm1d(64),  # Addition: Batch Normalization for fully connected layer
            nn.ReLU(),
            nn.Dropout(0.6),  # Addition: Dropout for fully connected layer
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x initial shape: (B, 52)
        # Add channel dimension to become (B, 1, 52) to fit Conv1d
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten (B, 416)
        x = self.classifier(x)
        return x


class TE_LSTM(nn.Module):
    """
    Long Short-Term Memory Network (LSTM)
    The current sequence length is 1 (treating the 52-dimensional data at a single time step as a sequence of time length 1)
    Reserves a perfect interface for introducing time windows in the future
    """

    def __init__(self, input_dim=52, hidden_dim=64, num_layers=2, num_classes=22):
        super(TE_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5  # Note: dropout here only acts between layers of multi-layer LSTM
        )

        # Addition: Normalization and regularization between LSTM output and classifier
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x initial shape: (B, 52)
        # Add time sequence dimension to become (B, 1, 52)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out, (h_n, c_n) = self.lstm(x)
        # Extract the output of the last time step (B, 1, hidden_dim) -> (B, hidden_dim)
        last_out = out[:, -1, :]

        # Output to the fully connected layer after BN and Dropout
        last_out = self.bn(last_out)
        last_out = self.dropout(last_out)

        return self.classifier(last_out)


class TE_Transformer(nn.Module):
    """
    Transformer Encoder Model
    Uses self-attention mechanism to find potential correlations among 52 features
    """

    def __init__(self, input_dim=52, d_model=64, nhead=4, num_layers=2, num_classes=22):
        super(TE_Transformer, self).__init__()
        # First map 52 dimensions to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)

        # Addition: Standard feature preprocessing for Transformer
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(0.5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.5
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Addition: Classifier normalization and Dropout
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x initial shape: (B, 52)
        # Add sequence dimension (B, Sequence_Length=1, 52)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.embedding(x)  # (B, 1, d_model)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        x = self.transformer_encoder(x)
        # Extract pooled results (since the sequence length is 1, just take the 0th position directly)
        x = x[:, 0, :]  # (B, d_model)

        return self.classifier(x)


# ================= Test Code (ensure data dimensions flow smoothly) =================
if __name__ == "__main__":
    # Simulate a Batch of data given by DataLoader (Batch Size=64, Features=52)
    dummy_x = torch.randn(64, 52)

    print("--- Dimension Flow Test ---")
    mlp = TE_MLP()
    print(f"MLP output dimension: {mlp(dummy_x).shape}")  # Expected: [64, 22]

    cnn = TE_1DCNN()
    print(f"CNN output dimension: {cnn(dummy_x).shape}")  # Expected: [64, 22]

    lstm = TE_LSTM()
    print(f"LSTM output dimension: {lstm(dummy_x).shape}")  # Expected: [64, 22]

    transformer = TE_Transformer()
    print(f"Transformer output dimension: {transformer(dummy_x).shape}")  # Expected: [64, 22]