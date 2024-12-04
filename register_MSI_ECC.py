import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from os.path import join
from ms_utils import *


def float64_to_uint8(input_array):
    # Step 1: Scale the values to [0, 255]
    min_val = np.min(input_array)
    max_val = np.max(input_array)

    scaled_array = ((input_array - min_val) / (max_val - min_val)) * 255.0
    # Step 2: Convert to np.uint8
    uint8_array = np.uint8(scaled_array)  
    return uint8_array  

lens_channel = {
     'Q1': {
         'ch': [0, 7, 11],       # Channel indices associated with lens Q1
         'wv': [405, 570, 710]      # Wavelengths associated with lens Q1
     },
     'Q2': {
         'ch': [4,8],       
         'wv': [525, 630]      
     },
     'Q3': {
         'ch': [13],      
         'wv': [850]      
     },
     'Q4': {
         'ch': [1,5,9],     
         'wv': [430,550,650]      
     },
     'Q5': {
         'ch': [3,12],            
         'wv': [490,735]                
     },
     'Q6': {
         'ch': [2,6,10],            
         'wv': [450,560,685]               
     },
     
     'default': {
        'wavelengths': [710,630,850,650,735,685],  # Wavelengths for default channels
        'default_channels': [11, 8, 13, 9, 12, 10]       # Default channels for all lenses
    }
 }

base_directory = "/home/localadmin/data_citrus/data_raw"
class_image_info = {}
class_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]
for class_folder in class_folders:
    # Directly call make_ms_dataset with the class folder path
    image_info_list = make_ms_dataset(os.path.join(base_directory, class_folder), ext='.TIFF')
    
    # Store the image_info_list in the dictionary
    class_image_info[class_folder] = image_info_list
    

bands_nm =[710,630,850,650,735,685]
master_band  = 630
master_band_index = bands_nm.index(master_band)

##############################################################################
# Identify complementary bands for the master band's lens
complementary_positions = {}
complementary_band= {}
for lens, info in lens_channel.items():
    if lens != 'default':
        wavelengths = info['wv']
        complementary_positions[lens] = []
        complementary_band[lens]=[]
        if master_band not in  wavelengths:
            for i, wv in enumerate(wavelengths):
                    if wv not in bands_nm:
                        complementary_positions[lens].append(info['ch'][i])
                        complementary_band[lens].append(info['wv'][i])

##############################################################################

warp_mode = cv.MOTION_HOMOGRAPHY
warp_matrix = np.eye(3, 3, dtype=np.float32)
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1000, 1e-5)



for class_folder, image_info_list in class_image_info.items():
    for i, image_data in enumerate(image_info_list):
        try:
                metadata = image_info_list[i]
                print(metadata['name'])
                multispectral_array = create_multispectral_array(image_data)

                wv = metadata['wavelengths']
                lenses_indices = []
                for band in bands_nm:
                    index = wv.index(band)
                    lenses_indices.append(index)

                lenses_array = multispectral_array[:, :, lenses_indices]
                registered_bands = []
                list_bands = []
                registered_lens = []

                for j, band in enumerate(bands_nm):
                    if band == master_band:
                        print(f'master band {band}nm')

                        master_channel = lenses_array[:, :, master_band_index]
                        registered_bands.append(master_channel)
                        list_bands.append(band)
                        registered_lens.append(master_channel)

                        for key, lens_data in lens_channel.items():
                            wv_list = lens_data.get('wv', [])
                            ch_list = lens_data.get('ch', [])

                            if master_band in wv_list:
                                matching_wavelengths = [wv for wv in wv_list if wv != master_band]

                                matching_channels = [ch for ch_index, ch in enumerate(ch_list) if ch_index != wv_list.index(master_band)]
                                break

                        for ch, wv in zip(matching_channels, matching_wavelengths):
                            list_bands.append(wv)
                            registered_bands.append(multispectral_array[:, :, ch])

                        break

                for j, band in enumerate(bands_nm):
                    if band != master_band:
                        print(f'Q{j+1}, band {band}nm')
                        channel = lenses_array[:, :, j]
                        (cc, warp_matrix) = cv.findTransformECC(master_channel, channel, warp_matrix, warp_mode, criteria)
                        registered_band = cv.warpPerspective(channel, warp_matrix, (master_channel.shape[1], master_channel.shape[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

                        print(f'initial cc = {cc}')
                        registered_bands.append(registered_band)
                        list_bands.append(band)
                        registered_lens.append(registered_band)

                        cc_after_warping = cv.matchTemplate(master_channel, registered_band, cv.TM_CCORR_NORMED)[0]
                        print(f' registered cc = {cc_after_warping}')

                        complementary_indices = complementary_positions.get(f'Q{j + 1}', [])
                        complementary_nm = complementary_band.get(f'Q{j + 1}', [])

                        for complement_index, nm in zip(complementary_indices, complementary_nm):
                            complement_channel = multispectral_array[:, :, complement_index]

                            registered_band = cv.warpPerspective(complement_channel, warp_matrix, (master_channel.shape[1], master_channel.shape[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
                            registered_bands.append(registered_band)
                            list_bands.append(nm)

          

                sorted_indices = np.argsort(list_bands)
                sorted_bands = np.sort(list_bands)
                sorted_registered_bands = [registered_bands[i] for i in sorted_indices]
                registered_image = np.dstack(sorted_registered_bands)
                
        
            
                image_name = image_data['name']
                par_fold, org = os.path.split(base_directory)
                for j, band in enumerate(sorted_bands):
                    regis_band = sorted_registered_bands[j]

                    output_folder = join(par_fold, 'registered_ecc',org,class_folder, f"{band}NM")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    output_path = join(output_folder, f"{image_name}_REG_{band}nm.tiff")
                    cv.imwrite(output_path, regis_band)
                    
                    print(output_folder)
                    print(output_path)

        except:
            im_id = metadata['name']
            print(f'image {im_id} registration failed')
            continue