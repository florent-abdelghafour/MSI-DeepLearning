import os
from PIL import Image
from ms_utils.msi_image_utils import *
import matplotlib.pyplot as plt
import h5py

dataset_path_ecc ="D:\\data_citrus\\registered_ecc\\data_raw"
dataset_path_raw ="D:\\data_citrus\\data_raw"

list_path =[dataset_path_raw,dataset_path_ecc]
list_im=[]

for path in list_path:
    if  check_dir(path):
        num_classes= count_folders(path)
        print(f'{num_classes} classes in {path}')
        
        dataset_info = {}     
        
        for class_folder in os.listdir(path):                              # iterate over classes folder (e.g. [HLB, Healthy])
            class_path = os.path.join(path,class_folder)
                
            num_bands = len(os.listdir(class_path))
            for band in os.listdir(class_path):                                    # iterate over spectral bands folder (e.g. [405 nm, ....., 850 nm])
                band_path = os.path.join(class_path,band)
                if os.path.isdir(band_path):
                    band_wavelength =extract_numeric_part(band_path)              # extract numeric part of band folder name -> find wavelengh       
                    
                    for i,filename in enumerate(os.listdir(band_path)): 
                       if i<100:                 
                        name, ext = os.path.splitext(filename)                     # Split the filename into name and extension parts
                        base_name = name.split('.')[0]                   
                        image_path = os.path.join(band_path, filename)
                        
                        if is_supported_image(image_path):
                            image = Image.open(image_path)
                        else:
                            continue
                        
                        if base_name not in dataset_info:
                            # print(base_name)
                            width, height = image.size
                            image.close()
                            
                            dataset_info[base_name] = {
                                                    'name': base_name,
                                                    'wavelengths': [],
                                                    'width': width,  # 
                                                    'height': height,
                                                    'image_paths': [],  # Store band image paths
                                                    'nb_bands': num_bands,
                                                    'img type': pil_mode_to_np_type.get(image.mode),
                                                    'class' : class_folder    ,
                                                    'default_bands': [685, 525, 430]                                               
                                                }
                            
                        dataset_info[base_name]['wavelengths'].append(band_wavelength)
                        dataset_info[base_name]['image_paths'].append(image_path)
        
        for name,image_info in zip(dataset_info.keys(),dataset_info.values()):
            image_info['image_paths'] = sorted(image_info['image_paths'], key=lambda path: extract_numeric_part(path))  
       
        list_im.append(dataset_info) 
    
     
outputs_path=[]     
for path in  list_path :
    output_directory =   os.path.join(os.path.dirname(path), "data_cube_2")
    outputs_path.append(output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)    


dict1, dict2 = list_im[0], list_im[1]

for  (image_info_1, image_info_2) in zip(dict1.values(), dict2.values()):
    image_info_list = [image_info_1, image_info_2]  # Combine both dictionaries into one list
    for idx,image_info in enumerate(image_info_list):
        
        output_directory=outputs_path[idx]
        output_class_directory = os.path.join(output_directory, image_info['class'])
        os.makedirs(output_class_directory, exist_ok=True)

        output_img_directory = os.path.join(output_class_directory, image_info['name'])
        os.makedirs(output_img_directory, exist_ok=True)

        output_file = os.path.join(output_img_directory, f"{image_info['name']}.h5")
        print(f"Output File: {output_file}")

        # Create the datacube
        datacube = create_multispectral_array(image_info)
        datacube = np.transpose(datacube, (2, 0, 1))

        # Save the HDF5 file
        with h5py.File(output_file, 'w') as hdf5_file:
            hdf5_file.create_dataset('datacube', data=datacube)
            metadata_group = hdf5_file.create_group('metadata')
            for key, value in image_info.items():
                if isinstance(value, list):
                    metadata_group.create_dataset(key, data=np.array(value, dtype='S'))
                elif isinstance(value, str):
                    metadata_group.attrs[key] = value.encode('utf-8')
                elif isinstance(value, (int, float)):
                    metadata_group.attrs[key] = value
                else:
                    metadata_group.attrs[key] = str(value).encode('utf-8')

        print(f"HDF5 file for {image_info['name']} created successfully.")
        
        mean_image = np.sum(datacube, axis=0)  # Shape: (H, W)
    
        # Save the image
        output_path =   os.path.join(os.path.dirname(output_directory), "data_mean")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = os.path.join(output_path, f"{image_info['name']}_mean.png")
        plt.imsave(save_path,mean_image,cmap='grey')
        print(f"Saved: {save_path}")
    
    