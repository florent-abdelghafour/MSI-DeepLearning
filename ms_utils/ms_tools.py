import numpy as np
import os
from PIL import Image
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle


def extract_numeric_part(s):
    # Use regular expressions to extract numeric part from a string
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    else:
        return None


def make_ms_dataset(base_directory, ext='.jpg'):
    image_info_dict = {}

    # Loop through the folders
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract the numeric part from the folder name as an integer
            wavelength = extract_numeric_part(folder_name)

        # Loop through the files in the folder
        for filename in os.listdir(folder_path):
            # Split the filename into name and extension parts
            name, ext = os.path.splitext(filename)
            

            # Check if the file is a JPEG image
            if ext.lower() == ext.lower():
                # Extract the image name (before the first period in the base name)
                base_name = name.split('.')[0]

                # Get the image file path
                image_path = os.path.join(folder_path, filename)

                # If dimensions are not yet determined, read them from the first image
                image = Image.open(image_path)

                # Create a dictionary for metadata if it doesn't exist
                if base_name not in image_info_dict:

                    width, height = image.size
                    image.close()

                    image_info_dict[base_name] = {
                        'name': base_name,
                        'wavelengths': [],
                        'width': width,  # Store dimensions only once per set of wavelengths
                        'height': height,
                        'image_paths': [],  # Store image paths
                        'nb_bands': len(os.listdir(base_directory))
                    }

                # Append the wavelength as an integer
                image_info_dict[base_name]['wavelengths'].append(wavelength)
                image_info_dict[base_name]['image_paths'].append(image_path)

    return list(image_info_dict.values())


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



def project_scores(ms_array, scores, mask=None):

    score_map = np.zeros(
        (ms_array.shape[0], ms_array.shape[1], scores.shape[1]))
    if mask == None:
        mask = np.ones((ms_array.shape[0], ms_array.shape[1]))
    count = 0

    for i, j in np.ndindex((ms_array.shape[0], ms_array.shape[1])):
        if np.all(mask[i, j] != 0):
            score_map[i, j, :] = scores[count, :]
            count += 1

    return score_map


def imadjust(x, a, b, c, d, gamma=1):
    # Converts an image range from [a,b] to [c,d].
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def pseudo_rgb(ms_array, metadata, default_bands=None):
    wv = metadata['wavelengths']
    if default_bands == None:
        default_bands = [685, 525, 430]

    rgb_indices = []
    for band in default_bands:
        index = wv.index(band)
        rgb_indices.append(index)

    rgb_im = ms_array[:, :, rgb_indices]

    stretched_truecolor = imadjust(
        rgb_im, rgb_im.min(), rgb_im.max(), 0, 1, gamma=0.5)

    return stretched_truecolor


def get_spectrum(ms_array, metadata):

    wv = np.array(metadata['wavelengths'])
    rgb_im = pseudo_rgb(ms_array, metadata)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(rgb_im, cmap='gray')
    plt.show()

    all_spec = []
    all_pos = []
    colors = iter(plt.cm.tab20(np.linspace(0, 1, len(wv))))

    legends = []

    while True:
        pts = []
        pts = list(plt.ginput(-1, timeout=-1))
        if plt.waitforbuttonpress():
            break

    fig, ax = plt.subplots()

    fig2, ax2 = plt.subplots(constrained_layout=True)
    ax2.imshow(rgb_im, cmap='gray')
    for pos in pts:
        x = int(pos[0])
        y = int(pos[1])

        # Get the next color from the color map
        color = next(colors)

        legend_marker = ax.scatter(
            [], [], marker='x', s=50, color=color, label=f'x={x}, y={y}')
        legends.append(legend_marker)

        text_x = x
        text_y = y - 100
        ax2.plot(x, y, 'x', color=color, markersize=8)
        ax2.text(text_x, text_y, f'({x}, {y})', color=color, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='center')

        spectrum = ms_array[y, x, :]
        all_spec.append(spectrum)
        all_pos.append((x, y))
        ax.plot(wv, spectrum, label=f'({x}, {y})', color=color)

        ax.legend(handles=legends, loc='center left',
                  bbox_to_anchor=(1, 0.5), title='samples')

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.grid()
    plt.show()

    return all_spec, all_pos


def extract_from_polygon(polygon, ms_array, stride):

    all_spec = []
    all_pos = []

    polygon_path = polygon.get_path()

    for y in range(0, ms_array.shape[0], stride):
        for x in range(0, ms_array.shape[1], stride):
            if polygon_path.contains_point((x, y)):
                spectrum = ms_array[y, x, :]
                all_spec.append(spectrum)
                all_pos.append((x, y))

    return all_spec, all_pos


def get_spectra_polygon(ms_array, metadata, stride=10):
    wv = np.array(metadata['wavelengths'])
    rgb_im = pseudo_rgb(ms_array, metadata)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(rgb_im, cmap='gray')
    plt.show()

    polygons = []  # List to store polygons
    all_spectra = []  # List to store spectra from all polygons
    labels = []  # List to store labels for each polygon
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    while True:
        pts = []
        pts = list(plt.ginput(-1, timeout=-1))
        if len(pts) < 3:
            print("Please select at least 3 points to define a polygon")
            continue

        color = colors[len(polygons) % len(colors)]

        polygon = Polygon(pts, closed=True, fill=False,
                          edgecolor=color, linewidth=2)
        ax.add_patch(polygon)
        plt.draw()

        # Prompt user for label
        label = input("Enter label for this polygon: ")
        labels.append(label)

        spec, all_pos = extract_from_polygon(polygon, ms_array, stride)
        all_spectra.append(spec)
        polygons.append(all_pos)

        if plt.waitforbuttonpress():
            break  # Finish the figure

    for idx, spec in enumerate(all_spectra):
        fig2, ax2 = plt.subplots(constrained_layout=True)
        for spectrum in spec:
            ax2.plot(wv, spectrum)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.grid()
        plt.title(f"Label: {labels[idx]}")
        plt.show()

    # Prompt the user for the database filename
    database_filename = input("Enter the database filename (e.g., data.pkl): ")

    # Save data as a database
    database = {
        'spectra': all_spectra,
        'positions': polygons,
        'labels': labels
    }

    with open(database_filename, 'wb') as file:
        pickle.dump(database, file)

    return all_spectra, polygons, labels
