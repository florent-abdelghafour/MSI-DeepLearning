from torch.utils.data import  DataLoader,random_split
from deep_utils import MSI_Dataset,ResNet18, ResNet34
from deep_utils import train
import torch
import torch.optim as optim
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt

dataset_root = "D:\\data_citrus\\data_cube"

citrus_data  = MSI_Dataset(root_dir=dataset_root,transform='resize', transform_args={"size": (640,640)})
dataloader = DataLoader(citrus_data )

NB_CH =14
batch_size = 12
EPOCHS=20
NW=0
LR = 0.01
WD = 0.015
model_type ='ResNet18_all_ch'

seed=42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


nb_samples = len(citrus_data)
classes= citrus_data.classes
num_classes = len(classes)

dataset_size = len(citrus_data)
test_size = int(0.2 * dataset_size)
train_size = dataset_size - test_size
cal_size = int(0.75 * train_size)
val_size = train_size - cal_size      


train_dataset, test_dataset = random_split(citrus_data, [train_size, test_size], 
                                            generator=torch.Generator().manual_seed(seed))
cal_dataset, val_dataset = random_split(train_dataset, [cal_size, val_size], 
                                        generator=torch.Generator().manual_seed(seed))

# augmentation = data_augmentation(slope=slope, offset=offset, noise=noise, shift=shift)
#  cal_dataset.dataset.preprocessing=augmentation

# Create data loaders
cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=True, num_workers=NW)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NW)
test_loader= DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=NW)

labs= '_'.join(classes) 

save_path = os.path.dirname(dataset_root) + f"/models/{model_type}/{labs}"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
base_path = os.path.dirname(dataset_root) + f"/figures/{model_type}/{labs}"
if not os.path.exists(base_path):
    os.makedirs(base_path)

model = ResNet18(in_channel=NB_CH,num_classes=num_classes,head='mlp')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

results= train(model, optimizer, criterion, cal_loader, val_loader, 
                                        num_epochs=EPOCHS, save_path=save_path)

train_losses = results['train_losses']
val_losses = results['val_losses']
val_f1 = results['val_metrics']
accuracies = results['accuracies']
best_epoch = results['best_epoch']

if 'best_model_path' in results:
    best_model_path = results['best_model_path']

tl = torch.stack(train_losses).numpy()
vl = torch.stack(val_losses).numpy()
f1= np.array(val_f1)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.plot(tl, label=f'Training Loss',color='tab:blue')
ax1.plot(vl, label=f'Validation Loss',color='tab:orange')
ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fancybox=True, shadow=True, fontsize=12)

ax2 = ax1.twinx()
ax2.tick_params(axis='y', labelcolor='tab:green')
ax2.set_ylabel('R2 Score', color='tab:green')
ax2.plot(f1, label=f'R2 Score', linestyle='--',color='tab:green')
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.75), fancybox=True, shadow=True, fontsize=12)

ax1.grid(True)
plt.title(f'Training performances')
# plt.show(block=False)

pdf_path =base_path+ f"/cross_entropy_{labs}.pdf"
plt.savefig(pdf_path, format='pdf')

# pickle_path = base_path + f"/RMSE_{['dataset_type']}_{labs}.pkl"
# with open(pickle_path, 'wb') as f:
#     pickle.dump(fig, f)

