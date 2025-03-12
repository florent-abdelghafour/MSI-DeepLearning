
from torch.utils.data import DataLoader
from deep_utils import MSI_Dataset
import torch
import torchvision.transforms as T
import random
import torch.nn.functional as F  


class MSI_aug:
    def __init__(self, noise_std=0.02, scale_range=(0.9, 1.1), shift_range=(-0.05, 0.05), 
                 flip=True, rotate=True, brightness=0.2, contrast=0.2, blur=True, elastic=True):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.flip = flip
        self.rotate = rotate
        self.brightness = brightness
        self.contrast = contrast
        self.blur = blur
        self.elastic = elastic

    def __call__(self, img_tensor):
        """
        Apply augmentation to the multispectral image tensor.
        :param img_tensor: Tensor of shape (C, H, W)
        :return: Augmented tensor of shape (C, H, W)
        """
        C, H, W = img_tensor.shape
        
        # Apply random flipping
        if self.flip:
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])  # Horizontal flip
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])  # Vertical flip
        
        # Apply random rotation
        if self.rotate:
            angle = random.choice([0, 90, 180, 270])
            img_tensor = T.functional.rotate(img_tensor, angle)

        # Apply spectral shifting and scaling per channel
        scale_factors = torch.empty(C).uniform_(*self.scale_range)
        shift_values = torch.empty(C).uniform_(*self.shift_range)
        img_tensor = img_tensor * scale_factors.view(C, 1, 1) + shift_values.view(C, 1, 1)

        # Add Gaussian noise
        noise = torch.randn_like(img_tensor) * self.noise_std
        img_tensor = img_tensor + noise
        
        # Apply random brightness adjustment
        if self.brightness > 0:
            factor = 1.0 + (random.uniform(-self.brightness, self.brightness))
            img_tensor = img_tensor * factor
        
        # Apply random contrast adjustment
        if self.contrast > 0:
            mean_val = img_tensor.mean(dim=[1, 2], keepdim=True)
            contrast_factor = 1.0 + (random.uniform(-self.contrast, self.contrast))
            img_tensor = (img_tensor - mean_val) * contrast_factor + mean_val
        
        # Apply random Gaussian blur
        if self.blur and random.random() > 0.5:
            kernel_size = random.choice([3, 5])  # Random kernel size
            img_tensor = T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))(img_tensor)
        
        # Apply random elastic distortion (spatial transformation) 
        if self.elastic and random.random() > 0.5:
            # Generate a random displacement field
            displacement = torch.randn(2, H, W) * 0.03  # Small perturbations
            base_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0), (1, C, H, W), align_corners=False)  # Identity grid
            new_grid = base_grid + displacement.permute(1, 2, 0).unsqueeze(0)  # Add displacement
            
            # Apply the grid warp
            img_tensor = F.grid_sample(img_tensor.unsqueeze(0), new_grid, mode='bilinear', padding_mode='reflection', align_corners=False).squeeze(0)

        return img_tensor




dataset_root = "D:\\data_citrus\\data_cube"

NB_CH =14
size =(800,800)
batch_size =1
citrus_data  = MSI_Dataset(root_dir=dataset_root,transform='resize', transform_args={"resize": {"size": size}})
dataloader = DataLoader(citrus_data,batch_size=batch_size, shuffle=True)

msi_augmentation = MSI_aug()

for i, batch in enumerate(dataloader):
    if i < 3:
        datacubes, labels = batch
        augmented_datacubes = torch.stack([msi_augmentation(img) for img in datacubes])