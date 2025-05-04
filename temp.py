#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation cubique de la température (Temp) en fonction de F (%) et Pression (GPa),
pour les missions Mer8 et Mer15 – affichage horizontal côte à côte.

Colormap personnalisée : noir → rouge → jaune.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 20,
})

# ---------------------------------------------------------------------------
# PARAMÈTRES -----------------------------------------------------------------
# ---------------------------------------------------------------------------
CSV_FILES: Dict[str, str] = {
    "8": "data/data_Mer8.csv",
    "15": "data/data_Mer15.csv",
}
INTERP_METHOD: str = "cubic"  # méthode unique
CMAP = LinearSegmentedColormap.from_list("black_red_yellow", ["black", "red", "yellow"])
FIGSIZE: tuple = (12, 5)
GRID_RES_F: int = 250
GRID_RES_P: int = 250

# ---------------------------------------------------------------------------
# LECTURE DES CSV ------------------------------------------------------------
# ---------------------------------------------------------------------------
required_cols = {"Pression", "F", "Temp"}
data: Dict[str, pd.DataFrame] = {}
for mission, file in CSV_FILES.items():
    path = Path(file)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {file}")
    df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans « {file} » : {', '.join(missing)}")
    data[mission] = df

# ---------------------------------------------------------------------------
# GRILLE DE RÉFÉRENCE COMMUNE ------------------------------------------------
# ---------------------------------------------------------------------------
all_F = np.concatenate([d["F"].values for d in data.values()])
all_P = np.concatenate([d["Pression"].values for d in data.values()])
F_grid = np.linspace(all_F.min(), 100.0, GRID_RES_F)
P_grid = np.linspace(all_P.min(), all_P.max(), GRID_RES_P)
F_mesh, P_mesh = np.meshgrid(F_grid, P_grid)

# ---------------------------------------------------------------------------
# FIGURE 1 × 2 : CUBIC uniquement --------------------------------------------
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2, figsize=FIGSIZE, sharex=True, sharey=True, constrained_layout=True
)

vmin_global, vmax_global = None, None  # pour gamme commune

# Calcul des valeurs interpolées et extrema
Ti_all = {}
for mission in ["8", "15"]:
    df = data[mission]
    points = df[["F", "Pression"]].values
    values = df["Temp"].values
    Ti = griddata(points, values, (F_mesh, P_mesh), method=INTERP_METHOD)
    Ti_all[mission] = Ti
    vmin = np.nanmin(Ti)
    vmax = np.nanmax(Ti)
    vmin_global = vmin if vmin_global is None else min(vmin_global, vmin)
    vmax_global = vmax if vmax_global is None else max(vmax_global, vmax)

# Tracés
for i, mission in enumerate(["8", "15"]):
    ax = axes[i]
    Ti = Ti_all[mission]
    im = ax.imshow(
        Ti,
        origin="lower",
        extent=[F_grid.min(), F_grid.max(), P_grid.min(), P_grid.max()],
        aspect="auto",
        cmap=CMAP,
        vmin=vmin_global,
        vmax=vmax_global,
    )
    ax.set_ylim(P_grid.max(), P_grid.min())  # pression croissante vers le bas
    ax.set_xlabel("F (%)")
    if i == 0:
        ax.set_ylabel("Pression (GPa)")
    label = "(a)" if mission == "8" else "(b)"
    ax.set_title(f"{label}")

# ---------------------------------------------------------------------------
# Barre de couleur commune
# ---------------------------------------------------------------------------
cbar_ax = fig.add_axes([0.15, -0.1, 0.7, 0.05])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Température interpolée (°C)")

plt.show()