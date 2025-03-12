import os
import h5py
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np


class MSI_Dataset2(Dataset):
    def __init__(self, root_dir, transform=None, transform_args=None):
        """
        Initialize the dataset.
        
      Args:
            root_dir (str): Path to the root directory containing the dataset.
            transforms (list or str, optional): A single transform or a list of transform names to apply sequentially.
            transform_args (dict, optional): Dictionary of arguments for each transform.
                Example:
                    {
                        "resize": {"size": (800, 800)},
                        "vegetation_index_transform": {"target_channels": [2, 6, 9]}
                    }
        """
        self.root_dir = root_dir
        self.classes=[]
        self.labels = []  
        self.label_encoder = LabelEncoder()
        self.class_counts={}
        self.data_info = self.build_data_info()
        self.h5_files = {} 
        
        if transform is None:
            self.transforms = []
        elif isinstance(transform, list):
            self.transforms = transform
        else:
            self.transforms = [transform]

        self.transforms_dict = {
            'resize': self.resize_transform ,
            'v index': self.vegetation_index_transform,        
        }
        
         # Initialize transform_args
        self.transform_args = transform_args if transform_args is not None else {}

        
    def build_data_info(self):
        """
        Build metadata for the dataset by traversing the directory structure.
        
        Returns:
            list: List of dictionaries, each containing:
                  - 'file_path': Path to the .h5 file.
                  - 'label': Class label extracted from the folder name.
        """
        data_info = []
        unique_classes = set()
        labels = []
        class_counts = {}
        
        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            unique_classes.add(class_folder)
            class_folder = class_folder.strip()
            
            if class_folder not in class_counts:
                class_counts[class_folder] = 0 
            
            for sample_folder in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample_folder)
                if not os.path.isdir(sample_path):
                    continue
                
                h5_file = os.path.join(sample_path, f"{sample_folder}.h5")
                if os.path.isfile(h5_file):
                    data_info.append({
                        "file_path": h5_file,
                        "label": class_folder  # Folder name as the label
                    })
                    
                    labels.append(class_folder)
                    class_counts[class_folder] += 1
                    
        self.classes = sorted(unique_classes)
        self.labels = np.array(labels)
        self.class_counts = [class_counts[cls] for cls in self.classes]
        
        if len(self.labels) > 0:
            self.label_encoder.fit(self.labels)
            for sample in data_info:
                sample["encoded_label"] = self.label_encoder.transform([sample["label"]])[0]
                    
        return data_info
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.data_info)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (datacube, label) where datacube is a PyTorch tensor and label is the class label.
        """
        sample_data_info = self.data_info[idx]
        file_path = sample_data_info["file_path"]
        label = torch.tensor(sample_data_info["encoded_label"]).to(torch.long)
        
        if file_path not in self.h5_files:
            self.h5_files[file_path] = h5py.File(file_path, "r")
        
        datacube = self.h5_files[file_path]["datacube"][:]
        datacube = torch.tensor(datacube, dtype=torch.float32)

            
        
        if self.transforms:
            datacube = self.apply_transforms(datacube)
            
        # metadata_group = f['metadata']

            # # Initialize an empty dictionary to store metadata
            # metadata = {}

            # # Loop through attributes and read them
            # for key, value in metadata_group.attrs.items():
            #     # Decode byte arrays to strings if necessary
            #     if isinstance(value, bytes):
            #         metadata[key] = value.decode('utf-8')
            #     else:
            #         metadata[key] = value

            # # Loop through datasets within the metadata group
            # for key in metadata_group.keys():
            #     dataset = metadata_group[key]
            #     if dataset.dtype.char == 'S':  # String array
            #         metadata[key] = dataset[()].astype(str).tolist()
            #     else:
            #         metadata[key] = dataset[()]
        
        return datacube, label     
    
    
    # def apply_transform(self, data):
    #     if self.transform in self.transforms_dict:
    #         transform_function = self.transforms_dict[self.transform]
    #         return transform_function(data)
    #     else:
    #         raise ValueError(f"Transform '{self.transform}' is not supported.")
        
    def apply_transforms(self, data):
        """
        Apply each transform in the self.transforms list in order.
        
        Args:
            data (Tensor): The input tensor to be transformed.
            
        Returns:
            Tensor: The transformed tensor.
        """
        for transform in self.transforms:
            if transform in self.transforms_dict:
                # Retrieve any additional arguments for this transform, if provided.
                args = self.transform_args.get(transform, {})
                transform_function = self.transforms_dict[transform]
                data = transform_function(data, **args)
            else:
                raise ValueError(f"Transform '{transform}' is not supported.")
        return data
    
            
    def resize_transform(self, data, size=(1200, 1200)):
        """
        Resize transform using PyTorch's interpolate function.
        
        Args:
            data (Tensor): Input tensor to resize.
            size (tuple, optional): The target size (height, width). Default is (1200, 1200).
            
        Returns:
            Tensor: The resized tensor.
        """
        # Add a batch dimension required for interpolation.
        data = data.unsqueeze(0)
        resized_data = torch.nn.functional.interpolate(data, size=size, mode='bilinear', align_corners=False)
        resized_data = resized_data.squeeze(0)  # Remove the batch dimension.
        return resized_data
    
    def vegetation_index_transform(self, data, target_channels):
        """
        Select specific channels (bands) for computing vegetation indices from a 4D tensor.
        
        Args:
            data (Tensor): Input tensor with shape (B, C, H, W).
            target_channels (list): A list of channel indices to select (e.g., [2, 6, 9]).
            
        Returns:
            Tensor: A 4D tensor with the selected channels.
            
        Raises:
            ValueError: If any channel in target_channels is out of range.
        """

        if not isinstance(target_channels, list):
            raise ValueError("target_channels must be provided as a list of channel indices.")
        
        num_channels = data.shape[1]
        # Check that each requested channel index is within the valid range.
        for idx in target_channels:
            if idx < 0 or idx >= num_channels:
                raise ValueError(f"Channel index {idx} is out of bounds for input with {num_channels} channels.")

        # Select and return only the specified channels.
        return data[:, target_channels, :, :]

    
    
    
# if __name__ == "__main__":
#     dataset_root = "D:\\data_citrus\\data_cube"
#     dataset = MSI_Dataset(root_dir=dataset_root)
#     # Set the maximum number of batches to load
#     max_batches = 2

#     # Create DataLoader
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Iterate through DataLoader
# for i, batch in enumerate(dataloader):
#     datacubes, labels = batch
#     print(f"Batch {i + 1}:")
#     print(f"  Datacubes shape: {datacubes.shape}")
#     print(f"  Labels: {labels}")
    
#     if i + 1 == max_batches:  # Stop after max_batches
#         break

