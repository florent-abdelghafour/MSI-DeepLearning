import os
import numpy as np
import cv2 as cv
from os.path import join
from ms_utils import make_ms_dataset, create_multispectral_array
from concurrent.futures import ThreadPoolExecutor


# Parameters independent of threads
generic_params = {
    "bands_nm": [710, 630, 850, 650, 735, 685],
    "master_band": 630,
    "lens_channel": {
        'Q1': {'ch': [0, 7, 11], 'wv': [405, 570, 710]},
        'Q2': {'ch': [4, 8], 'wv': [525, 630]},
        'Q3': {'ch': [13], 'wv': [850]},
        'Q4': {'ch': [1, 5, 9], 'wv': [430, 550, 650]},
        'Q5': {'ch': [3, 12], 'wv': [490, 735]},
        'Q6': {'ch': [2, 6, 10], 'wv': [450, 560, 685]},
        'default': {'wavelengths': [710, 630, 850, 650, 735, 685],
                    'default_channels': [11, 8, 13, 9, 12, 10]},
    },
    "warp_mode": cv.MOTION_HOMOGRAPHY,
    "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1000, 1e-5),
    "base_directory": "D:\\data_citrus\\data_raw",
    "output_folder_base": "registered_ecc",
}

# Identify complementary bands for the master band's lens
complementary_positions = {}
complementary_band = {}

bands_nm = generic_params["bands_nm"]
master_band = generic_params["master_band"]
lens_channel = generic_params["lens_channel"]

# Loop to populate complementary_positions and complementary_band
for lens, info in lens_channel.items():
    if lens != 'default':
        wavelengths = info['wv']
        complementary_positions[lens] = []
        complementary_band[lens] = []
        if master_band not in wavelengths:
            for i, wv in enumerate(wavelengths):
                if wv not in bands_nm:
                    complementary_positions[lens].append(info['ch'][i])
                    complementary_band[lens].append(info['wv'][i])

# Add complementary information to the generic params
generic_params['complementary_positions'] = complementary_positions
generic_params['complementary_band'] = complementary_band


# Function to process a single image
def process_image_data(image_data, generic_params):
    try:
        bands_nm = generic_params['bands_nm']
        master_band = generic_params['master_band']
        lens_channel = generic_params['lens_channel']
        warp_mode = generic_params['warp_mode']
        criteria = generic_params['criteria']
        base_directory = generic_params['base_directory']
        output_folder_base = generic_params['output_folder_base']
        
        print(image_data['name'])
        # Load multispectral array
        multispectral_array = create_multispectral_array(image_data)
        wv = image_data['wavelengths']
        master_band_index = bands_nm.index(master_band)
        
        # Get the lenses indices
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
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        for j, band in enumerate(bands_nm):
            if band != master_band:
                channel = lenses_array[:, :, j]
                try:
                    # Attempt ECC alignment
                    (cc, warp_matrix) = cv.findTransformECC(master_channel, channel, warp_matrix, warp_mode, criteria)
                except cv.error:
                    # Mark as failed and reset warp_matrix
                    print(f"ECC alignment failed for band {band}. Using the original band.")
                    warp_matrix = np.eye(3, 3, dtype=np.float32)
                    
                registered_band = cv.warpPerspective(channel, warp_matrix, (master_channel.shape[1], master_channel.shape[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

                
                registered_bands.append(registered_band)
                list_bands.append(band)
                registered_lens.append(registered_band)

                # cc_after_warping = cv.matchTemplate(master_channel, registered_band, cv.TM_CCORR_NORMED)[0]

                complementary_indices = complementary_positions.get(f'Q{j + 1}', [])
                complementary_nm = complementary_band.get(f'Q{j + 1}', [])

                for complement_index, nm in zip(complementary_indices, complementary_nm):
                    complement_channel = multispectral_array[:, :, complement_index]

                    registered_band = cv.warpPerspective(complement_channel, warp_matrix, (master_channel.shape[1], master_channel.shape[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
                    registered_bands.append(registered_band)
                    list_bands.append(nm)

        # Sort and save bands
        sorted_indices = np.argsort(list_bands)
        sorted_bands = np.sort(list_bands)
        sorted_registered_bands = [registered_bands[i] for i in sorted_indices]

        # Save registered bands
        image_name = image_data['name']
        par_fold, org = os.path.split(base_directory)
        for j, band in enumerate(sorted_bands):
            output_folder = join(par_fold, output_folder_base, org, class_folder, f"{band}NM")
            os.makedirs(output_folder, exist_ok=True)
            output_path = join(output_folder, f"{image_name}_REG.ARW_{band}nm.tiff")
            cv.imwrite(output_path, sorted_registered_bands[j])

        print(f"Processed image: {image_name}")
        return True

    except Exception as e:
        print(f"Image {image_data['name']} failed: {e}")
        return False


# Main parallel processing
if __name__ == "__main__":
    base_directory = generic_params['base_directory']
    
    class_image_info = {}
    class_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]
    for class_folder in class_folders:
        # Directly call make_ms_dataset with the class folder path
        image_info_list = make_ms_dataset(os.path.join(base_directory, class_folder), ext='.TIFF')
        # Store the image_info_list in the dictionary
        class_image_info[class_folder] = image_info_list
    
    for class_folder in class_folders:
        image_info_list = make_ms_dataset(os.path.join(base_directory, class_folder), ext='.TIFF')
        class_image_info[class_folder] = image_info_list

    failed_images = []

    for class_folder, image_info_list in class_image_info.items():
        print(f"Processing class folder: {class_folder}")

        # Parallel processing using ThreadPoolExecutor for the images in the current class folder
        with ThreadPoolExecutor() as executor:
            futures = []

            for image_data in image_info_list:
                futures.append(executor.submit(process_image_data, image_data, generic_params))

            for future in futures:
                success = future.result()
                if not success:
                    # If registered_image is None, add the failure details to the failed_images list
                    failed_image_info = {
                        'image_name': image_data['name'],
                        'image_path': join(base_directory, class_folder, image_data['name']),
                    }
                    failed_images.append(failed_image_info)
                else:
                    # Optionally, you can print success
                    print(f"Successfully processed {image_data['name']}.")

    # After all threads are done, log failed images to a file
    if failed_images:
        log_path = os.path.join(base_directory, "failed_images_reg_ecc_log.txt")
        with open(log_path, "w") as log_file:
            for failed_image in failed_images:
                log_file.write(f"Failed image: {failed_image['image_name']}, Path: {failed_image['image_path']}\n")
    
    
    
