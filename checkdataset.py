from torch.utils.data import DataLoader
from deep_utils import MSI_Dataset
import torch

dataset_root = "D:\\data_citrus\\data_cube"

max_batches=3
batch_size = 1
citrus_data  = MSI_Dataset(root_dir=dataset_root,transform='resize', transform_args={"resize": {"size": (800, 800)}})
dataloader = DataLoader(citrus_data,batch_size=batch_size, shuffle=True)

class_counts = torch.tensor(citrus_data.class_counts, dtype=torch.float32)
class_weights = class_counts / class_counts.sum()



for i, batch in enumerate(dataloader):
    datacubes, labels = batch
    print(f"Batch {i + 1}:")
    print(f"  Datacubes shape: {datacubes.shape}")
    print(f"  Labels: {labels}")
    print(f" Datacubes type: {datacubes.dtype}")
    
    if i + 1 == max_batches:  # Stop after max_batches
        break

