import os
from torch.utils.data import  DataLoader
from deep_utils import MSI_Dataset
import numpy as np
import matplotlib.pyplot as plt


dataset_root = "D:\\data_citrus\\registered_ecc\\data_cube"

output_directory =   os.path.join(os.path.dirname(dataset_root), "data_mean")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
dataset = MSI_Dataset(root_dir=dataset_root)
metadata=dataset.metadata

# Set the maximum number of batches to load
max_batches = 25

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate through DataLoader
for i, batch in enumerate(dataloader):
    datacubes, labels = batch
    print(f"Batch {i + 1}:")
    print(f"  Datacubes shape: {datacubes.shape}")
    print(f"  Labels: {labels}")
    
    file_path = dataset.metadata[i]["file_path"]
    image_name = os.path.splitext(os.path.basename(file_path))[0]
    print(image_name)
    
    # Convert datacubes Tensor to NumPy (Remove batch dimension)
    datacube_np = datacubes.squeeze(0).numpy()  # Shape: (C, H, W)

    # Compute mean channel image
    mean_image = np.mean(datacube_np, axis=0)  # Shape: (H, W)
    
    # Save the image
    output_path = os.path.join(output_directory, f"{image_name}_mean.png")
    plt.imsave(output_path,mean_image,cmap='hot')
    print(f"Saved: {output_path}")

    if i + 1 == max_batches:  # Stop after max_batches
            break
        
