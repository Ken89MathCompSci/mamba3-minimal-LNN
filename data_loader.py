import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

class UKDaleDataset(Dataset):
    """
    Dataset class for UK-DALE data
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_ukdale(file_path, appliance_index, window_size=100, target_size=1, 
                             train_ratio=0.7, val_ratio=0.15, normalize=True):
    """
    Load and preprocess UK-DALE data from BaseNILM format
    
    Args:
        file_path: Path to the .mat file
        appliance_index: Index of the appliance to disaggregate (0-based, add 2 for actual column)
        window_size: Size of the input sequence window
        target_size: Size of target output (1 for sequence-to-point)
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        normalize: Whether to normalize the data
        
    Returns:
        Dictionary containing DataLoaders and metadata
    """
    # Load the data
    data = scipy.io.loadmat(file_path)
    
    # Extract data based on the structure we identified
    # From the exploration, we know:
    # - 'input' contains the aggregate data with shape (9481946, 3)
    # - 'output' contains the appliance data with shape (9481946, 7)
    
    # The input structure is [time, id, power]
    mains_data = data['input'][:, 2]  # Third column is the power data
    
    # For the appliance data, we need to add 2 to the index
    # First 2 columns may be time and ID, then appliance data starts
    actual_appliance_index = appliance_index + 2
    
    # Make sure the index is within bounds
    if actual_appliance_index >= data['output'].shape[1]:
        # If out of bounds, provide error with more context
        raise ValueError(f"Appliance index {appliance_index} (column {actual_appliance_index}) is out of bounds. "
                         f"Available appliances are 0 to {data['output'].shape[1] - 3} (columns 2 to {data['output'].shape[1] - 1})")
    
    appliance_data = data['output'][:, actual_appliance_index]
    
    # Print information about the data
    print(f"Mains data shape: {mains_data.shape}")
    print(f"Appliance data shape: {appliance_data.shape}")
    
    # Get appliance name if available
    if 'labelOut' in data and isinstance(data['labelOut'], np.ndarray):
        label_out = data['labelOut']
        if label_out.size > actual_appliance_index:
            # Handle different labelOut structures
            if label_out.ndim == 2 and label_out.shape[0] == 1:
                # Shape is (1, N) - access as [0, index]
                if isinstance(label_out[0, actual_appliance_index], np.ndarray) and label_out[0, actual_appliance_index].size > 0:
                    appliance_name = str(label_out[0, actual_appliance_index].item()).strip()
                else:
                    appliance_name = str(label_out[0, actual_appliance_index]).strip()
            else:
                # 1D array
                if isinstance(label_out[actual_appliance_index], np.ndarray) and label_out[actual_appliance_index].size > 0:
                    appliance_name = str(label_out[actual_appliance_index].item()).strip()
                else:
                    appliance_name = str(label_out[actual_appliance_index]).strip()
            print(f"Appliance name: {appliance_name}")
        else:
            appliance_name = f"Appliance {appliance_index}"
    else:
        appliance_name = f"Appliance {appliance_index}"
    
    # Plot a sample of the data to verify
    plt.figure(figsize=(12, 6))
    sample_size = min(1000, len(mains_data))
    plt.plot(mains_data[:sample_size], label='Aggregate')
    plt.plot(appliance_data[:sample_size], label=appliance_name)
    plt.legend()
    plt.title(f'Sample Data from UK-DALE - {appliance_name}')
    plt.savefig(f'ukdale_sample_appliance_{appliance_index}.png')
    plt.close()
    
    # Normalize data if required
    if normalize:
        mains_scaler = StandardScaler()
        appliance_scaler = StandardScaler()
        
        mains_data = mains_scaler.fit_transform(mains_data.reshape(-1, 1)).flatten()
        appliance_data = appliance_scaler.fit_transform(appliance_data.reshape(-1, 1)).flatten()
    else:
        mains_scaler = None
        appliance_scaler = None
    
    # Create sequences
    X, y = create_sequences(mains_data, appliance_data, window_size, target_size)
    
    # Split data
    total_samples = len(X)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    
    # Use sequential split rather than random permutation
    # This is better for time series data
    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, total_samples)
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = UKDaleDataset(X_train, y_train)
    val_dataset = UKDaleDataset(X_val, y_val)
    test_dataset = UKDaleDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'mains_scaler': mains_scaler,
        'appliance_scaler': appliance_scaler,
        'appliance_index': appliance_index,
        'appliance_name': appliance_name,
        'window_size': window_size,
        'target_size': target_size,
        'input_size': 1,  # Single feature (power)
        'output_size': target_size
    }

def create_sequences(mains, appliance, window_size, target_size=1):
    """
    Create sequences for sequence-to-sequence or sequence-to-point prediction
    
    Args:
        mains: The aggregate power data
        appliance: The target appliance power data
        window_size: Size of the input window
        target_size: Size of the target window (1 for sequence-to-point)
        
    Returns:
        X: Input sequences
        y: Target sequences or points
    """
    X, y = [], []
    
    # Use stride to reduce the number of highly correlated samples
    stride = 5
    
    for i in range(0, len(mains) - window_size - target_size + 1, stride):
        X.append(mains[i:i+window_size])
        
        if target_size == 1:
            # Sequence-to-point: predict the middle point
            midpoint = i + window_size // 2
            y.append(appliance[midpoint:midpoint+1])
        else:
            # Sequence-to-sequence
            y.append(appliance[i+window_size:i+window_size+target_size])
    
    return np.array(X).reshape(-1, window_size, 1), np.array(y)

def explore_available_appliances(file_path):
    """
    Explore available appliances in a UK-DALE .mat file

    Args:
        file_path: Path to the .mat file

    Returns:
        Dictionary mapping appliance indices to their names
    """
    data = scipy.io.loadmat(file_path)

    appliance_names = {}

    # Check if labelOut is available
    if 'labelOut' in data and isinstance(data['labelOut'], np.ndarray):
        label_out = data['labelOut']

        # Handle different labelOut structures
        if label_out.ndim == 2 and label_out.shape[0] == 1:
            # Shape is (1, N) - flatten to get individual labels
            label_out = label_out.flatten()

        # Skip the first 2 columns (time and ID)
        for i in range(2, len(label_out)):
            # Extract the appliance name from the array
            if isinstance(label_out[i], np.ndarray) and label_out[i].size > 0:
                appliance_name = str(label_out[i].item()).strip()
            else:
                appliance_name = str(label_out[i]).strip()
            appliance_names[i-2] = appliance_name

    # If no names found, create generic ones
    if not appliance_names and 'output' in data:
        output_shape = data['output'].shape
        if len(output_shape) == 2:
            # Skip the first 2 columns (time and ID)
            for i in range(output_shape[1] - 2):
                appliance_names[i] = f"Appliance_{i}"

    return appliance_names

if __name__ == "__main__":
    folder_path = "preprocessed_datasets/ukdale"
    file_path = os.path.join(folder_path, "ukdale5.mat")
    
    # Explore available appliances
    appliances = explore_available_appliances(file_path)
    print("Available appliances:")
    for idx, name in appliances.items():
        print(f"  Index {idx}: {name}")
    
    # Test loading data for the first appliance
    appliance_index = 0  # First appliance (adjust based on available appliances)
    
    print(f"\nLoading data for appliance index {appliance_index}...")
    data_dict = load_and_preprocess_ukdale(
        file_path,
        appliance_index,
        window_size=100,
        target_size=1
    )
    
    print("\nData loaded successfully!")
    print(f"Appliance: {data_dict['appliance_name']}")
    print(f"Training batches: {len(data_dict['train_loader'])}")
    print(f"Validation batches: {len(data_dict['val_loader'])}")
    print(f"Test batches: {len(data_dict['test_loader'])}")
