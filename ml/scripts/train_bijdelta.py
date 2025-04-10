#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural network training script for tensor basis neural networks.
"""

import sys
import os
import random
import torch
import numpy as np

# Add the project root to the path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.utils.openFoamUtils import *
from src.utils.data_utils import *
from src.models.tbnn import *


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """Get the device to use for training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

sequence_length = 25
stride = 10
invariants = ['I1_1', 'I1_3', 'I1_5', 'I1_7', 'I1_13', 'I1_29', 'I1_33', 'q2', 'q3', 'q4']
target = 'bij'

#Model parameters
n_features = 10  
n_lstm_hidden = 128      # Increased hidden units in LSTM
n_hidden = 3
hidden_size = 20        # Increased hidden size in linear layers
num_lstm_layers = 2      # Increased number of LSTM layers
bidirectional = False     # Use bidirectional LSTM
n_post_lstm_layers = 2   # Additional fully connected layers after LSTM
output_size = 10
learning_rate = 1e-4
batch_size = 32
dropout = 0.1

num_epochs = 20
batch_size = 32
lr = 1e-4





def main():
    """Main training function."""
    # Set the seed for reproducibility
    set_seed(42) 
    
    # Get the device
    device = get_device()
    
    
    # 1. Load data
    X_stacked_846, Y_stacked_846, TB_stacked_846,_, gradU_stacked_846,y_values_846 = read_dataset(846, sequence_length,stride,invariants=invariants,target=target,filter=5,gradU_column= 'gradU_0_1')
    X_stacked_1155, Y_stacked_1155, TB_stacked_1155,_, gradU_stacked_1155,y_values_1155 = read_dataset(1155,sequence_length,stride,invariants=invariants,target=target,filter=5,gradU_column= 'gradU_0_1')
    X_stacked_1475, Y_stacked_1475, TB_stacked_1475,_, gradU_stacked_1475,y_values_1475 = read_dataset(1475,sequence_length,stride,invariants=invariants,target=target,filter=5,gradU_column= 'gradU_0_1')

    
    X_training = torch.cat([X_stacked_846, X_stacked_1475])[:,:,:,:].to(device)
    Y_training = torch.cat([Y_stacked_846, Y_stacked_1475])[:,:,:].to(device)
    TB_training = torch.cat([TB_stacked_846, TB_stacked_1475])[:,:,:,:].to(device)
    gradU_training = torch.cat([gradU_stacked_846, gradU_stacked_1475])[:,:,:].to(device)

    X_validation = X_stacked_1155[:,:,:,:].to(device)
    Y_validation = Y_stacked_1155[:,:,:].to(device)
    TB_validation = TB_stacked_1155[:,:,:,:].to(device)
    gradU_validation = gradU_stacked_1155[:,:,:].to(device)
    # 2. Preprocess data
    T_total, Y_total = X_training.shape[0], X_training.shape[1]
    num_points = T_total * Y_total
    all_pairs = np.array(np.meshgrid(np.arange(T_total), np.arange(Y_total))).T.reshape(-1, 2)

    val_indices = np.random.choice(len(all_pairs), size=int(0.2 * len(all_pairs)), replace=False)
    train_indices = np.array([i for i in range(len(all_pairs)) if i not in val_indices])

    val_pairs = all_pairs[val_indices]
    train_pairs = all_pairs[train_indices]
    #Get more weight for the yix
    y_indices = train_pairs[:, 1]  # Extract y indices
    weights = 1 / (y_indices + 1)  # Higher weight for smaller yix
    weights /= weights.sum()  # Normalize weights to sum to 1
    # 3. Create model
    model = LSTM_TBNN(n_features=n_features,
        n_lstm_hidden=n_lstm_hidden,
        n_hidden=n_hidden,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        bidirectional=bidirectional,
        n_post_lstm_layers=n_post_lstm_layers,
        output_size=output_size,
        dropout=dropout,
    ).to(device)
    # 4. Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()
    # 5. Train model
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Sample from training pairs with the biased weights
        batch_indices = np.random.choice(len(train_pairs), size=len(train_pairs), replace=False, p=weights)
        train_pairs_shuffled = train_pairs[batch_indices]

        # Batch loop
        for i in range(0, len(train_pairs), batch_size):
            batch_pairs = train_pairs_shuffled[i:i+batch_size]

            # Extract batch data
            tix_batch = batch_pairs[:, 0]
            yix_batch = batch_pairs[:, 1]

            x_batch = X_training[tix_batch, yix_batch]         # [batch_size, seq_len, features]
            y_batch = Y_training[tix_batch, yix_batch]        # [batch_size, 1]
            TB_batch = TB_training[tix_batch, yix_batch]      # [batch_size, 9, 10]
    
            optimizer.zero_grad()

            y_pred = model(x_batch, TB_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            optimizer.step()

            total_loss += loss.item() * batch_pairs.shape[0]  # Weighted loss

        average_loss = total_loss / len(train_pairs)

        # Validation on the held-out pairs
        model.eval()
        val_losses = []
        with torch.no_grad():
            # Validation
            for i in range(0, len(val_pairs), batch_size*2):
                batch_val_pairs = val_pairs[i:i+batch_size*2]
                
                tix_val = batch_val_pairs[:, 0]
                yix_val = batch_val_pairs[:, 1]
                
                x_val = X_training[tix_val, yix_val]
                y_val = Y_training[tix_val, yix_val]
                TB_val = TB_training[tix_val, yix_val]
                
                ypred_val = model(x_val, TB_val)
                val_loss = criterion(ypred_val, y_val)
                val_losses.append(val_loss.item())
                
        val_loss = np.mean(val_losses)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}, Val Loss: {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the weights
            print(f'>>> New best model saved at epoch {epoch+1} with Val Loss: {val_loss:.6f}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': {
                    'n_features': n_features,
                    'n_lstm_hidden': n_lstm_hidden,
                    'n_hidden': n_hidden,
                    'hidden_size': hidden_size,
                    'num_lstm_layers': num_lstm_layers,
                    'bidirectional': bidirectional,
                    'n_post_lstm_layers': n_post_lstm_layers,
                    'output_size': output_size,
                    'dropout': dropout
                }
            }, '../models/best_model_tbnn_w_hparm.pth')
            print(f'Best model saved with Val Loss: {best_val_loss:.6f}')


    # 6. Validate model
    model.eval()
    val_losses = []
    y_pred_list = []
    rel_errors = []
    with torch.no_grad():
        for t in range(X_validation.shape[0]):
            x_batch = X_validation[t]
            y_batch = Y_validation[t]
            TB_batch = TB_validation[t]
            
            # Prediction step
            y_pred = model(x_batch, TB_batch)
            
            # Store predictions
            y_pred_list.append(y_pred.cpu().detach().numpy())
            
            # Calculate loss
            loss = criterion(y_pred, y_batch)
            val_losses.append(loss.item())
            
            # Calculate relative error
            relative_error = np.abs(y_pred.cpu().detach().numpy() - y_batch.cpu().detach().numpy()) / np.abs(y_batch.cpu().detach().numpy())
            rel_errors.append(np.mean(relative_error))
            
    # Calculate average loss
    avg_loss = np.mean(val_losses)
    print(f"Average validation loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main()