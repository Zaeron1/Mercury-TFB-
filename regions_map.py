from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import os

# ————————————————————————————————————————————————————————————————
#  Paramètres d’affichage : agrandissement global de toutes les polices
# ————————————————————————————————————————————————————————————————
plt.rcParams.update({
    "font.size": 14,          # taille de base du texte
    "axes.titlesize": 18,     # titres des sous-graphiques
    "axes.labelsize": 16,     # libellés des axes
    "legend.fontsize": 14,    # légendes
    "xtick.labelsize": 12,    # graduations X
    "ytick.labelsize": 12,    # graduations Y
    "figure.titlesize": 20,   # titres de figure au niveau supérieur
})


# Permet de charger les images même si elles sont tronquées
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Désactiver la limite de pixels pour les images volumineuses (à utiliser avec précaution)
Image.MAX_IMAGE_PIXELS = None

# Charger le DEM (TIFF)
DATA_folder = "data"
DEM_path = os.path.join(DATA_folder, 'DEM.tif')
DEM = Image.open(DEM_path)

# Charger le masque (BMP)
region_path = os.path.join(DATA_folder, 'regions.bmp')
mask_image = Image.open(region_path)
mask_array = np.array(mask_image)

# Vérifier que les deux images ont la même taille, sinon redimensionner le DEM à la taille du masque
if DEM.size != mask_image.size:
    print("Redimensionnement du DEM à la taille du masque.")
    DEM = DEM.resize(mask_image.size, Image.BILINEAR)

# Convertir le DEM redimensionné en tableau numpy
DEM_array = np.array(DEM)

# Créer une colormap pour le masque
# Pour les valeurs 0 à 6 : 0 sera transparent, 1 à 6 auront des couleurs semi-transparentes.
mask_colors = [
    (0, 0, 0, 0),       # 0 : transparent
    (1, 0, 0, 0.5),     # 1 : rouge semi-transparent
    (0, 1, 0, 0.5),     # 2 : vert semi-transparent
    (0, 0, 1, 0.5),     # 3 : bleu semi-transparent
    (1, 1, 0, 0.5),     # 4 : jaune semi-transparent
    (1, 0, 1, 0.5),     # 5 : magenta semi-transparent
    (0, 1, 1, 0.5)      # 6 : cyan semi-transparent
]
mask_cmap = ListedColormap(mask_colors)
# Pour que chaque valeur entière soit correctement associée à une couleur,
# on définit les limites entre les classes.
norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], ncolors=mask_cmap.N)

# Définir l'étendue géographique et les graduations des axes
extent = [-180, 180, -90, 90]
longitude_ticks = np.arange(-180, 181, 60)
latitude_ticks = np.arange(-90, 91, 30)

# Créer la figure et l'axe
fig, axs = plt.subplots(figsize=(10, 8))

# Afficher le DEM en nuances de gris
axs.imshow(DEM_array, extent=extent, origin='upper', interpolation='none', cmap='gray')
axs.set_title("")
axs.set_xlabel("Longitude (°)")
axs.set_ylabel("Latitude (°)")
axs.set_xticks(longitude_ticks)
axs.set_yticks(latitude_ticks)
axs.grid(True)

# Superposer le masque avec le colormap défini
axs.imshow(mask_array, extent=extent, origin='upper', interpolation='none', cmap=mask_cmap, norm=norm)

# Ajouter une légende avec les noms des régions
legend_elements = [
    Patch(facecolor=(1, 0, 0, 0.5), edgecolor='r', label='High-Mg'),
    Patch(facecolor=(0, 1, 0, 0.5), edgecolor='g', label='Al-rich'),
    Patch(facecolor=(0, 0, 1, 0.5), edgecolor='b', label='Caloris'),
    Patch(facecolor=(1, 1, 0, 0.5), edgecolor='y', label='Rach'),
    Patch(facecolor=(1, 0, 1, 0.5), edgecolor='m', label='High-al NVP'),
    Patch(facecolor=(0, 1, 1, 0.5), edgecolor='c', label='Low-al NVP')
]
axs.legend(handles=legend_elements, loc='lower right', fontsize='small', title='Régions', title_fontsize='medium')

plt.show()