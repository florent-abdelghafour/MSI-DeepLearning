import os
import h5py
from torch.utils.data import Dataset, DataLoader
import torch

import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class MSI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Path to the root directory containing the dataset (organized by class folders).
            transform (callable, optional): Transform to apply to the data. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.metadata = self.build_metadata()
        
    def build_metadata(self):
        """
        Build metadata for the dataset by traversing the directory structure.
        
        Returns:
            list: List of dictionaries, each containing:
                  - 'file_path': Path to the .h5 file.
                  - 'label': Class label extracted from the folder name.
        """
        metadata = []
        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            for sample_folder in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample_folder)
                if not os.path.isdir(sample_path):
                    continue
                
                h5_file = os.path.join(sample_path, f"{sample_folder}.h5")
                if os.path.isfile(h5_file):
                    metadata.append({
                        "file_path": h5_file,
                        "label": class_folder  # Folder name as the label
                    })
        
        return metadata
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (datacube, label) where datacube is a PyTorch tensor and label is the class label.
        """
        sample_metadata = self.metadata[idx]
        file_path = sample_metadata["file_path"]
        label = sample_metadata["label"]
        
        # Load the .h5 file
        with h5py.File(file_path, "r") as f:
            datacube = torch.tensor(f["datacube"][:], dtype=torch.float32)
        
        # Apply transformation if specified
        if self.transform:
            datacube = self.transform(datacube)
        
        # Convert label to a numerical value if needed
        return datacube, label
    
    
    
if __name__ == "__main__":
    dataset_root = "D:\\data_citrus\\data_cube"
    dataset = MSI_Dataset(root_dir=dataset_root)
    # Set the maximum number of batches to load
    max_batches = 2

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate through DataLoader
for i, batch in enumerate(dataloader):
    datacubes, labels = batch
    print(f"Batch {i + 1}:")
    print(f"  Datacubes shape: {datacubes.shape}")
    print(f"  Labels: {labels}")
    
    if i + 1 == max_batches:  # Stop after max_batches
        break

