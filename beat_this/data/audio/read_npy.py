import numpy as np
import torch
def read_npy(path):
    return np.load(path)
def read_npy_torch(path):
    """
    Read a numpy file and convert it to a PyTorch tensor.
    
    Args:
        path (str or Path): Path to the numpy file
        
    Returns:
        torch.Tensor: The loaded data as a PyTorch tensor
    """
    return torch.from_numpy(np.load(path))


mel = read_npy_torch("/home/julio.hsu/beat_this/beat_this/data/audio/spectrograms/gtzan/gtzan_blues_00000/track.npy")
# Unpack the GTZAN dataset from npz file
def unpack_gtzan_npz(npz_path):
    """
    Unpacks a GTZAN dataset from an npz file.
    
    Args:
        npz_path (str): Path to the npz file containing GTZAN dataset
        
    Returns:
        dict: Dictionary containing the unpacked GTZAN data
    """
    print(f"Unpacking GTZAN dataset from {npz_path}")
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            gtzan_data = dict(data)
            print(f"Successfully unpacked GTZAN dataset with {len(gtzan_data)} entries")
            return gtzan_data
    except Exception as e:
        print(f"Error unpacking GTZAN dataset: {e}")
        return None

# Unpack the GTZAN dataset
gtzan_data = unpack_gtzan_npz("/home/julio.hsu/beat_this/gtzan.npz")
if gtzan_data:
    # Print some information about the dataset
    print(f"GTZAN dataset keys: {list(gtzan_data.keys())}")
    # Print the shape of the first item if available

author_mel = gtzan_data["gtzan_blues_00000/track"]
author_mel = torch.from_numpy(author_mel)
print(torch.allclose(mel, author_mel))
print("mel", mel.shape)
print("author_mel", author_mel.shape)

# Save the mel and author_mel tensors to files
def save_tensor(tensor, filename):
    """
    Save a PyTorch tensor to a numpy file.
    
    Args:
        tensor (torch.Tensor): The tensor to save
        filename (str): The filename to save to
    """
    np.save(filename, tensor.numpy())
    print(f"Saved tensor with shape {tensor.shape} to {filename}")

# Save the mel and author_mel tensors
save_tensor(mel, "gtzan_blues_00000_mel.npy")
save_tensor(author_mel, "gtzan_blues_00000_author_mel.npy")

print("Both tensors have been saved successfully.")
