
import numpy as np
import os
import re
from PIL import Image

import h5py 

###############################################################################
def extract_numeric_part(text):
    numeric_parts = re.findall(r'\d+', text)
    return int(numeric_parts[0]) if numeric_parts else None


def check_dir(directory):
    """Checks if a directory exists and contains files."""
    if os.path.exists(directory) and os.path.isdir(directory):
        for item in os.scandir(directory):
            if item.is_dir() and any(os.scandir(item.path)):
                return True
        print(f"{directory} exists but does not contain any non-empty folders.")
        return False
    else:
        print(f"{directory} does not exist or is not a directory.")
        return False

def count_folders(directory):
    return sum(os.path.isdir(os.path.join(directory, item)) for item in os.listdir(directory))


image_formats = {
    'JPEG': ('.jpg', '.jpeg'),
    'PNG': ('.png',),
    'TIFF': ('.tif', '.tiff'),
    
}

def is_supported_image(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    for format_name, extensions in image_formats.items():
        if ext in extensions:
            return True
    return False



###############################################################################
###############################################################################

def create_multispectral_array(image_info):
    if not image_info:
        return None

    # Get dimensions from the first image

    height, width = image_info['height'], image_info['width']

    # Get the number of channels (wavelengths)
    num_channels = len(image_info['wavelengths'])

    # Initialize the NumPy array
    multispectral_array = np.zeros(
        (height, width, num_channels), dtype=np.uint8)

    # Loop through the wavelengths for this image
    for j, wavelength in enumerate(image_info['wavelengths']):
        # Open the image and read the data
        image_path = image_info['image_paths'][j]
        image = Image.open(image_path)
        image_data = np.array(image)
        image.close()

        # Assign the image data to the corresponding channel in the array
        multispectral_array[:, :, j] = image_data

    return multispectral_array
###############################################################################

###############################################################################
# Mapping between PIL modes and NumPy data types
pil_mode_to_np_type = {
    '1': np.uint8,  # 1-bit pixels, black and white
    'L': np.uint8,  # 8-bit pixels, grayscale
    'P': np.uint8,  # 8-bit pixels, mapped to any other mode using a color palette
    'RGB': np.uint8,  # 3x8-bit pixels, true color
    'RGBA': np.uint8,  # 4x8-bit pixels, true color with transparency mask
    'CMYK': np.uint8,  # 4x8-bit pixels, color separation
    'YCbCr': np.uint8,  # 3x8-bit pixels, color video format
    'LAB': np.uint8,  # 3x8-bit pixels, the L*a*b color space
    'HSV': np.uint8,  # 3x8-bit pixels, Hue, Saturation, Value color space
    'I': np.int32,  # 32-bit signed integer pixels
    'F': np.float32  # 32-bit floating point pixels
}
###############################################################################


def save_datacube(image_info,output_directory):
    output_class_directory = os.path.join(output_directory,image_info['class'])
    if not os.path.exists(output_class_directory):
            os.makedirs(output_class_directory)
                 
    output_img_directory = os.path.join(output_class_directory,image_info['name'])
    if not os.path.exists(output_img_directory):
        os.makedirs(output_img_directory)
    
    output_file = os.path.join(output_img_directory, f"{image_info['name']}.h5")
    print(output_file)   
        
    datacube=create_multispectral_array(image_info)
    datacube = np.transpose(datacube, (2, 0, 1))
    
        
    with h5py.File(output_file, 'w') as hdf5_file:
        hdf5_file.create_dataset('datacube', data=datacube)
        metadata_group = hdf5_file.create_group('metadata')
        for key, value in image_info.items():
            if isinstance(value, list):
                # Convert lists to string arrays
                metadata_group.create_dataset(key, data=np.array(value, dtype='S'))
            elif isinstance(value, str):
                # Convert strings to byte arrays
                metadata_group.attrs[key] = value.encode('utf-8')
            elif isinstance(value, (int, float)):
                # Save numbers directly as attributes
                metadata_group.attrs[key] = value
            else:
                # For any other types, convert to string
                metadata_group.attrs[key] = str(value).encode('utf-8')

    print(f"HDF5 file {image_info['name']} created successfully.")