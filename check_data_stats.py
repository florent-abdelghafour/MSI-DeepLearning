from torch.utils.data import  DataLoader
from deep_utils import MSI_Dataset



dataset_root = "D:\\data_citrus\\registered_ecc\\data_cube_2"

dataset = MSI_Dataset(root_dir=dataset_root)
dataloader = DataLoader(dataset)

batch_size=1

for idx, batch in enumerate(dataloader):
    if idx<2:
        datacubes, labels = batch
        print(datacubes.size())
        print(labels)