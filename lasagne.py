from PIL import Image
import numpy as np
import os


def regions_array():
    """
    Charge 4 images (3 images de rapports et 1 image de masquage des régions),
    conserve uniquement la moitié supérieure (hémisphère nord) des cartes,
    applique une mise à l'échelle, remplace les 0 par NaN, calcule plusieurs
    rapports, et retourne un unique array 2D (6 x 7) de type object.

    Première dimension (x) : 6 rapports dans l'ordre suivant :
        0 : mg_si
        1 : ca_si
        2 : al_si
        3 : ca_mg (rapport Ca/Mg)
        4 : al_mg (rapport Al/Mg)
        5 : ca_al (rapport Ca/Al)

    Deuxième dimension (y) : 7 ensembles de données :
        0 à 5 : régions masquées selon la carte des régions (dans l'ordre) :
                ["high-Mg", "Al-rich", "Caloris", "Rach", "high-al NVP", "low-al NVP"]
        6     : carte complète (sans masque) pour le rapport considéré.

    Chaque élément giant_array[x, y] est un tableau 2D (dimensions = moitié
    supérieure de l'image) contenant les valeurs du rapport correspondant pour
    les pixels de la zone considérée.
    """

    # Noms des fichiers et correspondance des variables
    file_names   = ['mgsi.bmp', 'casi.bmp', 'alsi.bmp', 'regions.bmp']
    variables    = ['mg_si', 'ca_si', 'al_si', 'regions']
    region_values = (1, 2, 3, 4, 5, 6)
    region_names  = ["high-Mg", "Al-rich", "Caloris", "Rach", "high-al NVP", "low-al NVP"]


    # Dossier où sont stockés les données
    DATA_FOLDER = "data"


    # Chargement des images en ajoutant "data/" devant les noms de fichiers
    images = {var: Image.open(os.path.join(DATA_FOLDER, file)) for var, file in zip(variables, file_names)}

    # Création du masque des régions et découpe de la moitié supérieure
    regions_mask_full = np.array(images['regions'], dtype=float)
    half_height = regions_mask_full.shape[0] // 2  # hauteur / 2
    regions_mask = regions_mask_full[:half_height, :]

    # Facteurs d'échelle pour convertir en valeurs réelles
    scale_factors = {'mg_si': 0.860023, 'ca_si': 0.318000, 'al_si': 0.402477}

    # Conversion en tableaux NumPy, mise à l'échelle, découpe et remplacement de 0 par NaN
    arr = {}
    for k in scale_factors:
        data_full = np.array(images[k], dtype=float)
        data = data_full[:half_height, :] * (scale_factors[k] / 255.0)
        data[data == 0] = np.nan
        arr[k] = data

    # Création d'un masque commun pour les pixels valides dans Mg, Ca et Al
    valid_mask = ~np.isnan(arr['mg_si']) & ~np.isnan(arr['ca_si']) & ~np.isnan(arr['al_si'])
    for k in arr:
        arr[k] = np.where(valid_mask, arr[k], np.nan)

    # Calcul des rapports supplémentaires
    arr['ca_mg'] = arr['ca_si'] / arr['mg_si']
    arr['al_mg'] = arr['al_si'] / arr['mg_si']
    arr['ca_al'] = arr['ca_si'] / arr['al_si']

    # Empilement des 6 couches en un tableau 3D (hauteur x largeur x 6)
    channels_order = ['mg_si', 'ca_si', 'al_si', 'ca_mg', 'al_mg', 'ca_al']
    combined = np.stack([arr[ch] for ch in channels_order], axis=-1)

    # Extraction des régions masquées selon la carte des régions
    region_matrices = {}
    for r, name in zip(region_values, region_names):
        r_mask = (regions_mask == r) & valid_mask
        region_matrices[name] = np.where(r_mask[..., None], combined, np.nan)

    # Création des cartes complètes (sans masque) pour chaque canal
    for i, ch in enumerate(channels_order):
        region_matrices["full_" + ch] = combined[..., i]

    # Création du "giant array" final de dimensions (6, 7)
    num_channels = len(channels_order)       # 6
    num_regions  = len(region_names) + 1     # 6 régions + 1 (complete)
    giant_array = np.empty((num_channels, num_regions), dtype=object)

    # Remplissage pour chaque rapport et pour chaque région
    for i, ch in enumerate(channels_order):
        # Pour chaque région masquée
        for j, rname in enumerate(region_names):
            giant_array[i, j] = region_matrices[rname][..., i]
        # Pour la carte complète, on utilise la clé "full_" + ch
        giant_array[i, num_regions - 1] = region_matrices["full_" + ch]

    return giant_array


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Récupération du giant array (moitié supérieure uniquement)
    lasagne = regions_array()

    # Affichage des dimensions
    print("lasagne array shape:", lasagne.shape)

    # Exemple : carte complète du rapport Ca/Mg (indice 3, colonne 6)
    full_ca_mg = lasagne[3, 6]
    print("Shape de la carte complète Ca/Mg (moitié supérieure):", full_ca_mg.shape)

    # Visualisation
    plt.figure(figsize=(10, 8))
    plt.imshow(full_ca_mg, cmap='viridis')
    plt.colorbar(label='Ca/Mg Ratio')
    plt.title('Carte complète du rapport Ca/Mg (moitié supérieure)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()