import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_TBNN(nn.Module):
    def __init__(
        self,
        n_features,
        n_lstm_hidden,
        n_hidden,
        hidden_size=100,
        num_lstm_layers=2,
        bidirectional=False,
        n_post_lstm_layers=2,
        output_size=10,
        dropout=0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(n_features, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_hidden)]
        )
        self.dropout = nn.Dropout(dropout)
        h_in = hidden_size
        self.lstm = nn.LSTM(
            h_in,
            n_lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_output_size = n_lstm_hidden * (2 if bidirectional else 1)
        self.post_lstm_layers = nn.ModuleList(
            [nn.Linear(lstm_output_size, lstm_output_size) for _ in range(n_post_lstm_layers)]
        )
        self.output_layer = nn.Linear(lstm_output_size, output_size)

    def forward(self, x, tb):
        batch_size, seq_length, _ = x.shape
        x = F.selu(self.linear(x.reshape(-1, x.shape[2])))
        x = self.dropout(x)  # Apply dropout after the linear layer
        for hidden_layer in self.hidden_layers:
            x = F.selu(hidden_layer(x))
            x = self.dropout(x)  # Apply dropout after each hidden layer
        x = x.reshape(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the output of the last time step
        for post_layer in self.post_lstm_layers:
            x = F.selu(post_layer(x))
        x = F.selu(self.output_layer(x))
        # Select only indices 0, 1, 4, and 8 from x
        selected_indices = [0, 1, 4, 8]
        #print(f"x_selected shape: {x_selected.shape}")
        tb = tb.detach()
        # Select matching indices from tb (along the second dimension, matching x indices)
        #print(f"tb shape: {tb.shape}")
        tb_selected = tb[:, selected_indices, :]
        #print(f"tb_selected shape: {tb_selected.shape}")
        tb_layer = torch.einsum('ijk,ik->ij', tb_selected, x)
        return tb_layer

class LSTM_TBNN_R(nn.Module):
    def __init__(
        self,
        n_features=10,
        pre_lstm_layers=1,
        n_lstm_hidden=32,
        hidden_size=100,
        num_lstm_layers=2,
        bidirectional=True,
        n_post_lstm_layers=2,
        output_size=10,  # 10 tensor basis coefficients
        dropout=0.2,     # Standard dropout rate
        recurrent_dropout=0.2,  # Dropout rate for recurrent connections
    ):
        super().__init__()
        self.linear = nn.Linear(n_features, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(pre_lstm_layers)]  # single pre-LSTM layer
        )
        self.dropout = nn.Dropout(dropout)
        
        # In PyTorch, 'dropout' parameter is for input->hidden connections
        # 'recurrent_dropout' concept is implemented via hidden->hidden dropout (second parameter)
        self.lstm = nn.LSTM(
            hidden_size,
            n_lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_lstm_layers > 1 else 0,  # Only applies when num_layers > 1
        )
        
        lstm_output_size = n_lstm_hidden * (2 if bidirectional else 1)
        self.post_lstm_layers = nn.ModuleList(
            [nn.Linear(lstm_output_size, lstm_output_size) for _ in range(n_post_lstm_layers)]
        )
        self.output_layer = nn.Linear(lstm_output_size, output_size)

    def forward(self, X, TB, gradU):
        """
        Inputs:
        - X: (700, 250, 11) → 10 features + 1 k
        - TB: (700, 160, 9, 10) → Tensor basis
        - gradU: (700, 160, 1) → Only 1 non-zero component
        Output:
        - R_pred: (700, 160, 1)
        """
        k = X[:, -1, -1]  # (700, 160)
        features = X[:, :, :10]  # (700, 160, 10)

        batch_size, seq_len, _ = features.shape

        x = features.reshape(-1, features.shape[2])  # (700*160, 10)
        x = F.selu(self.linear(x))  # (700*160, hidden)
        x = self.dropout(x)  # Apply dropout after activation

        for layer in self.hidden_layers:
            x = F.selu(layer(x))
            x = self.dropout(x)  # Apply dropout after each hidden layer

        x = x.reshape(batch_size, seq_len, -1)  # (700, 160, hidden)

        # Apply LSTM (already has dropout configured in __init__)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # (700, lstm_output_size)

        for layer in self.post_lstm_layers:
            x = F.selu(layer(x))
            x = self.dropout(x)  # Apply dropout after each post-LSTM layer

        alpha = self.output_layer(x)  # (700, 10)
        
        # Contract with TB[:, :, 1, :] to get bR_flat[1]
        tb_reduced = TB[:, 1, :]  # (700, 160, 10)
        alpha_expanded = alpha # (700, 1, 10)
        # Compute bR_flat[1] = sum(alpha * tb_reduced)
        bR_1 = torch.sum(tb_reduced * alpha_expanded, dim=1)  # (700, 160)
        # gradU: (700, 160, 1) → squeeze to (700, 160)
        gradU_flat = gradU.squeeze(-1)
        # Calculate R_pred
        R_pred = 2 * k * bR_1 * gradU_flat  # (700, 160)

        return R_pred.unsqueeze(-1) 
