import os
import matplotlib.pyplot as plt
from deep_utils import test 
from torch.utils.data import  DataLoader,random_split
from deep_utils import MSI_Dataset,ResNet18
import torch
import numpy as np

model_path  = "D:\\data_citrus\\models\\ResNet18_all_ch_4\\hlb_non_hlb_best.pth"
dataset_root = "D:\\data_citrus\\data_cube"


citrus_data  = MSI_Dataset(root_dir=dataset_root,transform='resize', transform_args={"resize": {"size": (1200, 1200)}})
dataloader = DataLoader(citrus_data )
    
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
test_size = int(0.25 * dataset_size)
train_size = dataset_size - test_size
train_dataset, test_dataset = random_split(citrus_data, [train_size, test_size], 
                                            generator=torch.Generator().manual_seed(seed))


NB_CH =14
batch_size = 4 
IP=16 # MAX = 64 
# NW=0 #   -----------0 for windows ~20 ore more on Linux depending on CPU - GPU I/O-----------------------
NW = min(4, os.cpu_count() - 1) if os.name != 'nt' else 0
num_classes=2

test_loader= DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=NW)
model = ResNet18(in_channel=NB_CH,num_classes=num_classes,head_type='mlp',in_planes =IP,zero_init_residual=False)

model.load_state_dict(torch.load(model_path))

Y_pred, Y = test(model,model_path,test_loader)


num_classes = Y_pred.shape[1]  # Number of classes
y_true = (Y.numpy()).astype(int)  # Convert tensors to NumPy for easier indexing
Y_pred =(Y_pred.numpy()).astype(int)

conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)

# Fill the confusion matrix
for true, pred in zip(Y_pred, y_true):
    conf_matrix[true, pred] += 1
    
    
precision, recall, f1_score = [], [], []

for c in range(num_classes):
    TP = conf_matrix[c, c].item()
    FP = conf_matrix[:, c].sum().item() - TP  # All predicted as class c but not actual class c
    FN = conf_matrix[c, :].sum().item() - TP  # All actual class c but not predicted as class c

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

print("Precision per class:", precision)
print("Recall per class:", recall)
print("F1-score per class:", f1_score)