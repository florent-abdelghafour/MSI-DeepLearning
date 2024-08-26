import torch
from torch.utils.data import Dataset, DataLoader    
from torchvision.transforms import v2
import os
import h5py
from sklearn.preprocessing import LabelEncoder



# ###############################################################################
# ###############################################################################
class HyperspectralDataset(Dataset):
    def __init__(self, hdf5_directory,transform=None, transform_args=None):
        self.hdf5_directory = hdf5_directory
        self.file_paths = []
        self.transform = transform      
        self.transform_args = transform_args if isinstance(transform_args, dict) else {}
       
        self.classes=[]
        self.labels = []  
        self.label_encoder = LabelEncoder()
      
                    
        self.transforms_dict = {
            'resize': self.resize_transform          
        }

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        hdf5_path = self.file_paths[idx]
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            data=hdf5_file['dataset'][:]
            data = torch.tensor(data)
       
        label_str = self.labels[idx]
        label = torch.tensor(self.label_encoder.transform([label_str])[0], dtype=torch.long)


        
        if self.transform:
            data = self.apply_transform(data)
        
        return data.float(), label

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
    
    def populate_file_paths_and_classes(self):
            for class_folder in os.listdir(self.hdf5_directory):    
                self.classes.append(class_folder)
    
            for root, _, files in os.walk(self.hdf5_directory):
                for file in files:
                    if file.endswith('.h5'):
                        file_path =os.path.join(root, file)
                        self.file_paths.append(file_path)
              
                        class_label=file_path.split(os.sep)[-3]
                        self.labels.append(class_label)
                           
            self.label_encoder.fit(self.labels)
                            
    