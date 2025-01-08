import os
import h5py
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch


class MSI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_args=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Path to the root directory containing the dataset (organized by class folders).
            transform (callable, optional): Transform to apply to the data. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.transform_args = transform_args if isinstance(transform_args, dict) else {}
        self.classes=[]
        self.labels = []  
        self.label_encoder = LabelEncoder()
        self.data_info = self.build_data_info()
     
        
    def build_data_info(self):
        """
        Build metadata for the dataset by traversing the directory structure.
        
        Returns:
            list: List of dictionaries, each containing:
                  - 'file_path': Path to the .h5 file.
                  - 'label': Class label extracted from the folder name.
        """
        data_info = []
        for class_folder in os.listdir(self.root_dir):
            self.classes.append(class_folder)
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            
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
                    
                    self.labels.append(class_folder)
        self.label_encoder.fit(self.labels)                 
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
        label_str = sample_data_info["label"]
        label = torch.tensor(self.label_encoder.transform([label_str])[0], dtype=torch.long)
        
        # Load the .h5 file
        with h5py.File(file_path, "r") as f:
            datacube = torch.tensor(f["datacube"][:], dtype=torch.float32)
        
        if self.transform:
            datacube = self.apply_transform(datacube)
            
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
            #     # Apply transformation if specified
            #     if self.transform:
            #         datacube = self.transform(datacube)
        
        return datacube, label     
    
    
    def apply_transform(self, data):
        if self.transform in self.transforms_dict:
            transform_function = self.transforms_dict[self.transform]
            return transform_function(data)
        else:
            raise ValueError(f"Transform '{self.transform}' is not supported.")
            
    def resize_transform(self,data,size=None):
        if size is None:
            size = self.transform_args.get('size', (1200,1200))   
        
        # Convert numpy array to tensor

        # Apply resize transformation
        resized_data = torch.nn.functional.interpolate(data.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
        resized_data = resized_data.squeeze(0)
             
        return resized_data
    
    
    
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

