"""
This script processes four BMP images representing Mg/Si, Ca/Si, Al/Si ratios and region masks 
on Mercury’s surface. It keeps only the northern hemisphere, applies scaling factors, 
computes new ratio maps (Ca/Mg, Al/Mg, Ca/Al), and extracts pixel values for each predefined region.

The final output is a 2D NumPy array of shape (6, 7), where:
  • First dimension = 6 different chemical ratios:
        0: Mg/Si
        1: Ca/Si
        2: Al/Si
        3: Ca/Mg
        4: Al/Mg
        5: Ca/Al
  • Second dimension = 7 region types:
        0–5 = masked regions (in order): ["High-Mg", "Al-rich", "Caloris", "Rach", "High-al NVP", "Low-al NVP"]
        6    = full map (no masking)

Each element of the array is a 2D array (image) of the selected ratio, masked for the corresponding region.
"""

from PIL import Image
import numpy as np
import os

def regions_array():
    """
    Loads 3 ratio maps and 1 region mask map, all in BMP format,
    and returns a structured array (6, 7) containing masked pixel values 
    for each ratio and region (plus full maps).
    """

    # List of image file names and matching variable names
    file_names = ['mgsi.bmp', 'casi.bmp', 'alsi.bmp', 'regions.bmp']
    variables = ['mg_si', 'ca_si', 'al_si', 'regions']
    region_values = (1, 2, 3, 4, 5, 6)
    region_names = ["high-Mg", "Al-rich", "Caloris", "Rach", "high-al NVP", "low-al NVP"]

    # Folder containing the .bmp files
    DATA_FOLDER = "data"

    # Load all images as grayscale arrays
    images = {var: Image.open(os.path.join(DATA_FOLDER, file)) for var, file in zip(variables, file_names)}

    # Extract the region mask and crop to northern hemisphere (upper half)
    regions_mask_full = np.array(images['regions'], dtype=float)
    half_height = regions_mask_full.shape[0] // 2
    regions_mask = regions_mask_full[:half_height, :]

    # Scaling factors (from original data to real values)
    scale_factors = {'mg_si': 0.860023, 'ca_si': 0.318000, 'al_si': 0.402477}

    # Convert, scale, crop and replace 0 with NaN
    arr = {}
    for k in scale_factors:
        data_full = np.array(images[k], dtype=float)
        data = data_full[:half_height, :] * (scale_factors[k] / 255.0)
        data[data == 0] = np.nan
        arr[k] = data

    # Only keep pixels valid in all 3 source maps
    valid_mask = ~np.isnan(arr['mg_si']) & ~np.isnan(arr['ca_si']) & ~np.isnan(arr['al_si'])
    for k in arr:
        arr[k] = np.where(valid_mask, arr[k], np.nan)

    # Compute extra chemical ratios
    arr['ca_mg'] = arr['ca_si'] / arr['mg_si']
    arr['al_mg'] = arr['al_si'] / arr['mg_si']
    arr['ca_al'] = arr['ca_si'] / arr['al_si']

    # Define the order of channels
    channels_order = ['mg_si', 'ca_si', 'al_si', 'ca_mg', 'al_mg', 'ca_al']

    # Stack into one 3D array: (height, width, 6)
    combined = np.stack([arr[ch] for ch in channels_order], axis=-1)

    # Masked regions
    region_matrices = {}
    for r, name in zip(region_values, region_names):
        r_mask = (regions_mask == r) & valid_mask
        region_matrices[name] = np.where(r_mask[..., None], combined, np.nan)

    # Full maps without masking
    for i, ch in enumerate(channels_order):
        region_matrices["full_" + ch] = combined[..., i]

    # Final array: shape (6, 7)
    num_channels = len(channels_order)
    num_regions = len(region_names) + 1  # +1 for full maps
    giant_array = np.empty((num_channels, num_regions), dtype=object)

    for i, ch in enumerate(channels_order):
        # Fill masked regions
        for j, rname in enumerate(region_names):
            giant_array[i, j] = region_matrices[rname][..., i]
        # Fill full map
        giant_array[i, num_regions - 1] = region_matrices["full_" + ch]

    return giant_array


# ─────────────────────────────────────────────────────────────
# EXAMPLE USAGE: Previewing a full map
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Load the array (only upper hemisphere)
    lasagne = regions_array()

    # Show array structure
    print("lasagne array shape:", lasagne.shape)  # Should be (6, 7)

    # Example: show the full (non-masked) Ca/Mg ratio map
    full_ca_mg = lasagne[3, 6]  # Index 3 = Ca/Mg, column 6 = full map
    print("Shape de la carte complète Ca/Mg (moitié supérieure):", full_ca_mg.shape)

    # Display it
    plt.figure(figsize=(10, 8))
    plt.imshow(full_ca_mg, cmap='viridis')
    plt.colorbar(label='Ca/Mg Ratio')
    plt.title('Carte complète du rapport Ca/Mg (moitié supérieure)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()