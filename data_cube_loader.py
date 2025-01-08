import h5py
import os
import matplotlib.pyplot as plt

dataset_path_ecc ="D:\\data_citrus\\registered_ecc\\data_cube_2"
dataset_path_raw ="D:\\data_citrus\\data_cube_2"

list_roots=[dataset_path_raw,dataset_path_ecc]

dataset={}

for root in list_roots: 
    class_type = os.listdir(root)
    for c in class_type:
        class_path = os.path.join(root,c)
        for im_folder in os.listdir(class_path):
            im_path = os.path.join(class_path,im_folder)
            if os.path.isdir(im_path):
                    # Use the im_folder as the key and store path and class
                    dataset[im_folder] = {
                        "path": im_path,
                        "class": c
                    }
           

for key in list(dataset.keys())[:5]:  
    print(f"Key: {key}")
    print(f"Value: {dataset[key]}")
    print("-" * 40)  


# Get the first 5 keys
first_keys = list(dataset.keys())[:5]

# Dictionary to store matching keys and their corresponding datacubes
matching_datacubes = {}

# Iterate over the first 5 keys
for key in first_keys:
    current_path = dataset[key]['path']
    print(f"Checking for key: {key} with path: {current_path}")
    
    # Find other keys where the current path is contained within the 'path'
    matching_keys = [k for k in dataset.keys() if current_path in dataset[k]['path'] and k != key]
    
    if matching_keys:
        print(f"Found matching keys for {key}: {matching_keys}")
        for match_key in matching_keys:
            try:
                # Read 'datacube' from the current key
                with h5py.File(current_path, 'r') as h5_file:
                    if 'datacube' in h5_file:
                        current_datacube = h5_file['datacube'][:]
                        print(f"Datacube shape for key '{key}': {current_datacube.shape}")
                    else:
                        print(f"Key 'datacube' not found in {current_path}")
                
                # Read 'datacube' from the matching key
                match_path = dataset[match_key]['path']
                with h5py.File(match_path, 'r') as h5_file:
                    if 'datacube' in h5_file:
                        match_datacube = h5_file['datacube'][:]
                        print(f"Datacube shape for key '{match_key}': {match_datacube.shape}")
                        
                        # Compare the datacubes
                        if current_datacube.shape == match_datacube.shape:
                            comparison_result = (current_datacube == match_datacube).all()
                            print(f"Datacubes are identical: {comparison_result}")
                        else:
                            print(f"Datacubes have different shapes: {current_datacube.shape} vs {match_datacube.shape}")
                    else:
                        print(f"Key 'datacube' not found in {match_path}")
            except Exception as e:
                print(f"Error processing files: {e}")
    print("-" * 40)

