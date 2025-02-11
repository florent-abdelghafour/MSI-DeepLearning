import os
from PIL import Image

from ms_utils.msi_image_utils import *
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


if __name__ == "__main__":
    dataset_path ="D:\\data_citrus\\data_raw"
    # '/home/localadmin/data_citrus/data_raw'

    dataset_info = {}                                                              # Initialize an empty dictionary to store information

    if  check_dir(dataset_path):
        num_classes= count_folders(dataset_path)
        print(f'{num_classes} classes in root')


        for class_folder in os.listdir(dataset_path):                              # iterate over classes folder (e.g. [HLB, Healthy])
            class_path = os.path.join(dataset_path,class_folder)
                
            num_bands = len(os.listdir(class_path))
            for band in os.listdir(class_path):                                    # iterate over spectral bands folder (e.g. [405 nm, ....., 850 nm])
                band_path = os.path.join(class_path,band)
                if os.path.isdir(band_path):
                    band_wavelength =extract_numeric_part(band_path)              # extract numeric part of band folder name -> find wavelengh       
                    
                    for filename in os.listdir(band_path):                 
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
                                                    'width': width,  # Store dimensions only once per set of wavelengths
                                                    'height': height,
                                                    'image_paths': [],  # Store image paths
                                                    'nb_bands': num_bands,
                                                    'img type': pil_mode_to_np_type.get(image.mode),
                                                    'class' : class_folder    ,
                                                    'default_bands': [685, 525, 430]                                               
                                                }
                            
                        dataset_info[base_name]['wavelengths'].append(band_wavelength)
                        dataset_info[base_name]['image_paths'].append(image_path)
        
        for name,image_info in zip(dataset_info.keys(),dataset_info.values()):
            image_info['image_paths'] = sorted(image_info['image_paths'], key=lambda path: extract_numeric_part(path))  
        

    output_directory =   os.path.join(os.path.dirname(dataset_path), "data_cube")
        
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
              
        
    with ProcessPoolExecutor() as executor:
        # Initialize tqdm progress bar with total number of tasks
        futures = {executor.submit(save_datacube, image_info,output_directory): image_info for image_info in dataset_info.values()}
        # Update progress bar as tasks complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving Datacubes", unit="datacube"):
            future.result()  # Retrieve result to ensure any exceptions are raised