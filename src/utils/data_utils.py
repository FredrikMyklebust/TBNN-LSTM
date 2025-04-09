import os
import sys
import requests
import zipfile
from tqdm import tqdm  
import torch
import pandas as pd
import numpy as np
from src.config import BF_DATASET_PATH
from scipy.ndimage import gaussian_filter1d



def precompute_stacked_batches(X, Y, TB, additional_inputs=None, gradU=None, sequence_length=3, stride=1):
    X_batches = []
    Y_batches = []
    TB_batches = []
    additional_inputs_batches = []
    gradU_batches = []

    T = X.shape[0]  # Total time steps

    for t in range(T):
        # Compute time indices with spacing, wrapping around using modulo
        indices = [(t - i * stride) % T for i in reversed(range(sequence_length))]  # e.g., t-6, t-3, t
        X_batch = X[indices].transpose(1, 0, 2)  # Shape: (1, sequence_length, 1)
        Y_batch = Y[t]
        TB_batch = TB[t]
        
        X_batches.append(torch.tensor(X_batch, dtype=torch.float32))
        Y_batches.append(torch.tensor(Y_batch, dtype=torch.float32))
        TB_batches.append(torch.tensor(TB_batch, dtype=torch.float32))
        
        if additional_inputs is not None:
            additional_inputs_batch = additional_inputs[t]
            additional_inputs_batches.append(torch.tensor(additional_inputs_batch, dtype=torch.float32))
        
        if gradU is not None:
            gradU_batch = gradU[t]
            gradU_batches.append(torch.tensor(gradU_batch, dtype=torch.float32))

    X_stacked = torch.stack(X_batches)
    Y_stacked = torch.stack(Y_batches)
    TB_stacked = torch.stack(TB_batches)
    
    additional_inputs_stacked = None
    if additional_inputs is not None:
        additional_inputs_stacked = torch.stack(additional_inputs_batches)
    
    gradU_stacked = None
    if gradU is not None:
        gradU_stacked = torch.stack(gradU_batches)

    return X_stacked, Y_stacked, TB_stacked, additional_inputs_stacked, gradU_stacked

def read_dataset(Re, sequence_length, stride, 
                 invariants=None,target='bij', 
                 additional_input_columns=None, 
                 gradU_column=None,filter=None):
    # Default values if not provided
    if invariants is None:
        invariants = ['I1_1', 'I1_3', 'I1_5', 'I1_7', 'I1_13', 'I1_29', 'I1_33', 'q2', 'q3', 'q4', 'krans']
    
    if additional_input_columns is None:
        additional_input_columns = ['ddt', 'div', 'laplacian', 'prod', 'sp']
    
    flow_case = 'Re'+str(Re)
    dataset = pd.read_csv(BF_DATASET_PATH)
    grouped_data = dataset.groupby('case_name')
    data = grouped_data.get_group(flow_case)
    
    X = data[invariants].values
    y_values = data['y'].values
    kDeficit = data.filter(regex='kDeficit').values
    bijDelta = data.filter(regex='bijDelta').values

    TB = data.filter(regex='T').values.reshape(-1, 9, 10)
    
    # Handle optional inputs
    gradU = None
    if gradU_column is not None:
        gradU = data[gradU_column].values
    
    additional_inputs = None
    if additional_input_columns and len(additional_input_columns) > 0:
        additional_inputs = data[additional_input_columns].values
    
    # Reshape the tensors to have the correct shape
    X = X.reshape(700, -1, len(invariants))
    if target == 'bij':
        Y = bijDelta.reshape(700, -1, 9)
        Y = Y[:,:,[0, 1, 4, 8]]
    elif target == 'kDeficit':
        Y = kDeficit.reshape(700, -1, 1)
    else:
        raise ValueError(f"Target {target} not supported")
    TB = TB.reshape(700, -1, 9, 10)
    #If filtering is needed
    
    if filter is not None:
        tb_filtered = np.zeros_like(TB)
        for i in range(TB.shape[0]):
            for j in range(TB.shape[2]):
                tb_filtered[i,:,j,:] = gaussian_filter1d(
                    TB[i,:,j,:],
                    sigma=filter,
                    axis=0)
        TB = tb_filtered
    if gradU is not None:
        gradU = gradU.reshape(700, -1, 1)
    
    if additional_inputs is not None:
        additional_inputs = additional_inputs.reshape(700, -1, len(additional_input_columns))
    
    X_stacked, Y_stacked, TB_stacked, additional_inputs_stacked, gradU_stacked = precompute_stacked_batches(
        X, Y, TB, additional_inputs, gradU, sequence_length, stride
    )
    
    return X_stacked, Y_stacked, TB_stacked, additional_inputs_stacked, gradU_stacked, y_values

def download_and_extract_data(url, download_dir):
    """
    Downloads and extracts a zip file from the specified URL
    """
    # Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Get the zip filename from the URL
    zip_filename = "turbulence_statistics.zip"
    zip_path = os.path.join(download_dir, zip_filename)
    
    # Download the file if it doesn't exist
    if not os.path.exists(zip_path):
        print(f"Downloading ZIP file from {url}...")
        
        # Stream the download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Download complete: {zip_path}")
    else:
        print(f"ZIP file already exists at {zip_path}")
    
    # Extract the ZIP file
    extract_dir = os.path.join(download_dir, "vanDerA2018")
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
        print(f"Extracting ZIP file to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the total number of files for progress tracking
            total_files = len(zip_ref.namelist())
            
            # Extract with progress tracking
            for i, file in enumerate(zip_ref.namelist()):
                zip_ref.extract(file, extract_dir)
                if i % 10 == 0 or i == total_files - 1:  # Update progress periodically
                    print(f"Extracted {i+1}/{total_files} files ({(i+1)/total_files*100:.1f}%)", end="\r")
        print(f"\nExtraction complete: {extract_dir}")
    else:
        print(f"Files already extracted at {extract_dir}")
    
    return extract_dir