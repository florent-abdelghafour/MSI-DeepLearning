from deep_utils import HyperspectralDataset
from ms_utils import *
from torch.utils.data import DataLoader
import torch

data_path = "F:\\Leticia\\citrus_data\\data_loader\hdf5"
citrus_data  = HyperspectralDataset(data_path, transform='resize', transform_args={'size': (1200, 1200)})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
num_workers=0



dataloader = DataLoader(citrus_data, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=num_workers)

batch_counter = 0
for data, label ,metadata in dataloader:
    batch_counter += 1
    msi=data.cpu().detach()
    metadata=metadata

    if batch_counter>1:
        break
    
