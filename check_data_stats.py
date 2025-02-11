from torch.utils.data import  DataLoader
from deep_utils import MSI_Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch

wavelengths= [405, 430, 450, 490, 525, 550, 560, 570, 630, 650, 685, 710, 735, 850]

dataset_root = "D:\\data_citrus\\data_cube_2"

dataset = MSI_Dataset(root_dir=dataset_root,transform='resize', transform_args={"size": (800, 800)})
dataloader = DataLoader(dataset)

batch_size=1
num_samples = 100

for idx, batch in enumerate(dataloader):
    if idx<5:
        datacubes, labels = batch
        rows, cols = datacubes.shape[2], datacubes.shape[3]
        random_pixel_indices = torch.randint(0, rows, (num_samples,)), torch.randint(0, cols, (num_samples,))
        spectral_samples = datacubes[0, :, random_pixel_indices[0], random_pixel_indices[1]]  
        spectral_samples_np = spectral_samples.numpy()

        # Plot spectral samples as curves
        plt.figure(figsize=(8, 5))
        for i in range(num_samples):
            plt.plot(range(len(wavelengths)), spectral_samples_np[:, i], label=f"Pixel {i+1}")

        plt.title("Spectral samples")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.grid(True)
        plt.xticks(ticks=range(len(wavelengths)), labels=wavelengths)
        plt.show()