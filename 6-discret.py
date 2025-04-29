"""
Analyse géochimique Mercurienne : interpolation des données expérimentales Mer8/Mer15,
correspondance avec les cartes régionales, visualisations cartographiques et statistiques.

"""

# – Les données expérimentales sont interpolées par pression et rapport élémentaire
# – Les cartes régionales sont fournies par *lasagne.regions_array()*
# – Les cartes régionales sont comparées aux données expérimentales interpolées  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from tqdm import tqdm
from matplotlib.patches import Patch
from typing import Dict, List
import lasagne  # module hypothétique fournissant les cartes régionale
import os

DATA_FOLDER = "data"

# ————————————————————————————————————————————————————————————————
#  Paramètres d’affichage : agrandissement global de toutes les polices
# ————————————————————————————————————————————————————————————————
plt.rcParams.update({
    "font.size": 14,          # taille de base du texte
    "axes.titlesize": 18,     # titres des sous-graphiques
    "axes.labelsize": 16,     # libellés des axes
    "legend.fontsize": 20,    # légendes
    "xtick.labelsize": 12,    # graduations X
    "ytick.labelsize": 12,    # graduations Y
    "figure.titlesize": 20,   # titres de figure au niveau supérieur
})

###############################################################################
# PRÉ‑TRAITEMENT DES DONNÉES EXPÉRIMENTALES
###############################################################################

def process_dataset(df: pd.DataFrame, deg: int = 1) -> Dict[str, np.ndarray]:
    """Interpolation polynomiale des rapports élémentaires pour chaque pression,
    puis appariement pixel‑par‑pixel avec les cartes régionales fournies par
    *lasagne.regions_array()*.
    """

    # Interpolation 1D des jeux de données expérimentaux ----------------------
    pressures = np.sort(df["Pression"].unique())
    records: List[Dict[str, float]] = []
    for p in pressures:
        sub = df[df["Pression"] == p].sort_values("F")
        if sub.empty:
            continue
        Fv = np.linspace(sub["F"].min(), max(100, sub["F"].max()), 50)
        polys = {r: Polynomial.fit(sub["F"], sub[r], deg) for r in ["Mg/Si", "Ca/Si", "Al/Si"]}
        records.extend({"Pression": p, "F": fv, **{r: polys[r](fv) for r in polys}} for fv in Fv)
    df_int = pd.DataFrame(records)

    # Récupération des cartes régionales (Mg, Ca, Al) --------------------------
    g = lasagne.regions_array()  # shape = (3, 6, Ny, Nx)
    maps = {r: np.asarray(g[i, 6], float) for i, r in enumerate(["Mg", "Ca", "Al"])}
    shape = maps["Mg"].shape

    exp = {r: df_int[r].values for r in ["Mg/Si", "Ca/Si", "Al/Si"]}
    best_res = np.full(shape, np.nan)
    best_idx = np.full(shape, -1, int)

    # Recherche du meilleur enregistrement expérimental pour chaque pixel ------
    for i in tqdm(range(shape[0]), desc="Pixels"):
        for j in range(shape[1]):
            obs = {r: maps[r][i, j] for r in maps}
            if any(np.isnan(v) for v in obs.values()):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                res = (
                    np.abs(exp["Mg/Si"] - obs["Mg"]) / obs["Mg"]
                    + np.abs(exp["Ca/Si"] - obs["Ca"]) / obs["Ca"]
                    + np.abs(exp["Al/Si"] - obs["Al"]) / obs["Al"]
                )
            idx = int(np.nanargmin(res))
            best_res[i, j] = res[idx]
            best_idx[i, j] = idx

    pressure_map = np.where(best_idx >= 0, df_int["Pression"].values[best_idx], np.nan)
    F_map = np.where(best_idx >= 0, df_int["F"].values[best_idx], np.nan)

    return {
        "df_int": df_int,
        "best_res": best_res,
        "best_idx": best_idx,
        "pressure_map": pressure_map,
        "F_map": F_map,
        "pressures": pressures,
    }

###############################################################################
# VISUALISATIONS INTERPOLATION & CARTES – (identiques à l’origine)
###############################################################################

def plot_interp(results: Dict[str, dict], data: Dict[str, pd.DataFrame], ratios: List[str]):
    """Compare les données brutes et interpolées pour chaque ratio."""
    for ratio in ratios:
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 8))
        for a, k in zip(ax, results):
            df_i, raw = results[k]["df_int"], data[k]
            for p in results[k]["pressures"]:
                sr = raw[raw["Pression"] == p]
                si = df_i[df_i["Pression"] == p]
                a.scatter(sr["F"], sr[ratio], s=10)
                a.plot(si["F"], si[ratio], lw=1)
            a.set_title(f"Mer{k}")
        ax[1].set_xlabel("F (%)")
        ax[0].set_ylabel(ratio)
        ax[1].set_ylabel(ratio)
        plt.tight_layout()
        plt.show()


def plot_map(results: Dict[str, dict], key: str, title: str, cmap: str = "viridis", unit: str = ""):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    if key == "F_map":
        vmin = 0
        vmax = 60
    else:
        vmin = min(np.nanmin(results["8"][key]), np.nanmin(results["15"][key]))
        vmax = max(np.nanmax(results["8"][key]), np.nanmax(results["15"][key]))

    for a, k in zip(ax, results):
        im = a.imshow(results[k][key], origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
        # Graduations latitude/longitude --------------------------------------
        lon_ticks = np.linspace(0, results[k][key].shape[1] - 1, 9)
        lat_ticks = np.linspace(0, results[k][key].shape[0] - 1, 3)
        lon_labels = [f"{x:.0f}°" for x in np.linspace(-180, 180, len(lon_ticks))]
        lat_labels = [f"{y:.0f}°" for y in np.linspace(90, 0, len(lat_ticks))]
        a.set_xticks(lon_ticks)
        a.set_xticklabels(lon_labels)
        a.set_yticks(lat_ticks)
        a.set_yticklabels(lat_labels)
        a.set_title(f"Mer{k} – {title}")

    # Barre de couleurs commune ----------------------------------------------
    fig.subplots_adjust(top=0.92, bottom=0.07, hspace=0.15)
    cax = fig.add_axes([0.2, 0.02, 0.6, 0.02])
    plt.colorbar(im, cax=cax, orientation="horizontal").set_label(f"{title} {unit}")
    plt.show()

###############################################################################
# HISTOGRAMMES & TABLEAUX – (identiques sauf plot_region_counts)
###############################################################################

def plot_hist_res(results):
    plt.figure()
    for k, style in zip(["8", "15"], [{"alpha": 0.5}, {"histtype": "step", "linestyle": "--"}]):
        res = results[k]["best_res"].ravel()
        plt.hist(res[~np.isnan(res)], bins=30, label=f"Mer{k}", **style)
    plt.xlabel("Somme des résidus relatifs")
    plt.ylabel("Nombre de pixels")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_frac_table(results):
    thr = [0.20, 0.30, 0.50]
    fracs = {
        k: [
            100
            * np.sum(results[k]["best_res"][~np.isnan(results[k]["best_res"])] <= t)
            / np.sum(~np.isnan(results[k]["best_res"]))
            for t in thr
        ]
        for k in ["8", "15"]
    }
    df = pd.DataFrame({"Threshold": thr, "Mer8 (%)": fracs["8"], "Mer15 (%)": fracs["15"]})
    print("\nTable des fractions de surface sous seuils (%):\n", df.to_string(index=False))
    x = np.arange(len(thr))
    plt.bar(x - 0.2, fracs["8"], 0.4, label="Mer8")
    plt.bar(x + 0.2, fracs["15"], 0.4, label="Mer15", hatch="///")
    plt.xticks(x, thr)
    plt.xlabel("Seuil de résidu")
    plt.ylabel("% surface")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_region_counts(results):
    """
    Histogramme RELATIF des pixels par région et par pression (Mer8/Mer15).

    – Mer8 : barres pleines semi-transparentes ;
    – Mer15 : contours pointillés.
    – Chaque barre est normalisée par le nombre total de pixels valides de la région.
    """

    names = ["High-Mg", "Al-rich", "Caloris", "Rach", "High-al NVP"]
    ps = np.unique(np.concatenate([results["8"]["pressures"], results["15"]["pressures"]]))
    ga = lasagne.regions_array()
    cmap = plt.cm.get_cmap("tab20", len(names))
    fig, ax = plt.subplots(figsize=(9, 6))
    bw = 0.8 / len(names)

    for i, reg in enumerate(names):
        mask = ~np.isnan(np.asarray(ga[0, i], float))
        total_pixels = np.sum(mask)
        if total_pixels == 0:
            continue

        for k in ["8", "15"]:
            idxs = [
                results[k]["df_int"].at[idx, "Pression"]
                for (ii, jj), idx in np.ndenumerate(results[k]["best_idx"])
                if idx >= 0 and mask[ii, jj]
            ]
            counts = np.bincount([np.where(ps == p)[0][0] for p in idxs], minlength=len(ps))
            # normalisation relative
            counts = counts / total_pixels

            x = np.arange(len(ps)) + (i - len(names) / 2 + 0.5) * bw
            ax.bar(
                x,
                counts,
                bw,
                label=f"{reg} (Mer{k})",
                facecolor=cmap(i) if k == "8" else "none",
                edgecolor=cmap(i),
                alpha=0.6 if k == "8" else 1.0,
                linestyle="--",
                linewidth=1.2,
            )

    ax.set_xticks(np.arange(len(ps)))
    ax.set_xticklabels([f"{p} GPa" for p in ps])
    ax.set_ylabel("Fraction relative des pixels")
    ax.set_title("")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    plt.tight_layout()
    plt.show()


def plot_Fopt(results):
    bins = np.linspace(0, 100, 21)
    F8 = results["8"]["F_map"].ravel()[~np.isnan(results["8"]["F_map"].ravel())]
    F15 = results["15"]["F_map"].ravel()[~np.isnan(results["15"]["F_map"].ravel())]
    df = pd.DataFrame(
        {
            "F_bin_start": bins[:-1],
            "Mer8_count": np.histogram(F8, bins)[0],
            "Mer15_count": np.histogram(F15, bins)[0],
        }
    )
    print("\nTable des nombres de pixels par bin de F optimal :\n", df.to_string(index=False))
    plt.hist(F8, bins, alpha=0.7, label="Mer8")
    plt.hist(F15, bins, histtype="step", linestyle="--", label="Mer15")
    plt.xlabel("F optimal (%)")
    plt.ylabel("Nombre de pixels")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.show()

###############################################################################
# COMPARAISON RÉGIONALE – VERSION ZOOMÉE
###############################################################################

def analyze_regions(results: Dict[str, dict]):
    """Compare les rapports élémentaires observés vs modélisés pour plusieurs régions.

    – Trois sous‑graphes côte à côte (Mg/Si, Ca/Si, Al/Si), chacun avec ses propres limites.
    – Axes recadrés sur l’intervalle pertinent (plus de détour par l’origine 0, 0).
    – Diagonale pointillée *x = y* traversant toute la zone, identifiée dans la légende.
    """

    # Préparation des données --------------------------------------------------
    ga = lasagne.regions_array()
    region_names = [
        "High-Mg",
        "Al-rich",
        "Caloris",
        "Rach",
        "High-al NVP",
        "Low-al NVP",
    ]
    ratios = {"Mg/Si": 0, "Ca/Si": 1, "Al/Si": 2}
    colors = plt.cm.get_cmap("tab10", len(region_names))
    markers = {"8": "*", "15": "D"}
    marker_labels = {"8": "Mer8", "15": "Mer15"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    rows = []  # pour la table des R²
    region_handles = {}
    mission_handles = {}

    # Parcourir chaque ratio et sous‑graphe ------------------------------------
    for ax, (ratio, idx_r) in zip(axes, ratios.items()):
        obs_all, mod_all = [], []  # pour déterminer les limites d’axe
        for mission in markers:
            obs_m, mod_m = [], []
            for i, reg in enumerate(region_names):
                mask = ~np.isnan(np.asarray(ga[idx_r, i], float))
                obs = np.asarray(ga[idx_r, i], float)[mask]
                idxs = results[mission]["best_idx"][mask].astype(int)
                valid = idxs >= 0
                obs, idxs = obs[valid], idxs[valid]
                if obs.size == 0:
                    obs_m.append(np.nan)
                    mod_m.append(np.nan)
                    continue

                mod = results[mission]["df_int"][ratio].values[idxs]
                obs_mean = float(np.nanmean(obs))
                mod_mean = float(np.nanmean(mod))
                obs_m.append(obs_mean)
                mod_m.append(mod_mean)

                sc = ax.scatter(
                    mod_mean,
                    obs_mean,
                    marker=markers[mission],
                    s=110,
                    color=colors(i),
                    label=f"{reg}" if mission == "8" else "_"
                )
                obs_all.append(obs_mean)
                mod_all.append(mod_mean)

                # Handle légende couleur une seule fois
                if reg not in region_handles:
                    patch = plt.Line2D([0], [0], marker='s', color='none',
                                       markerfacecolor=colors(i), markersize=12,
                                       label=reg)
                    region_handles[reg] = patch

            # Handle légende mission une seule fois
            if mission not in mission_handles:
                handle = plt.Line2D([0], [0], marker=markers[mission], color='black',
                                    linestyle='None', markersize=10, label=marker_labels[mission])
                mission_handles[mission] = handle

            # Coefficient de corrélation pour le ratio/mission -----------------
            mask_valid = ~np.isnan(obs_m) & ~np.isnan(mod_m)
            if np.sum(mask_valid) >= 2:
                r = np.corrcoef(np.array(obs_m)[mask_valid], np.array(mod_m)[mask_valid])[0, 1]
                rows.append({"Ratio": ratio, "Mission": f"Mer{mission}", "R²": r ** 2})
            else:
                rows.append({"Ratio": ratio, "Mission": f"Mer{mission}", "R²": np.nan})

        # Ajustements esthétiques du sous‑graphe ------------------------------
        all_vals = np.array(obs_all + mod_all)
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
        ax_lim_min = vmin - pad
        ax_lim_max = vmax + pad
        ax.plot([ax_lim_min, ax_lim_max], [ax_lim_min, ax_lim_max], "k--", linewidth=0.8,
                label=r"$x=y$" if ratio == "Mg/Si" else "_")
        ax.set_xlim(ax_lim_min, ax_lim_max)
        ax.set_ylim(ax_lim_min, ax_lim_max)
        ax.set_xlabel("Modélisé")
        ax.set_ylabel("Observé")
        ax.set_title(ratio)

    # Légende unique ----------------------------------------------------------
    all_handles = list(region_handles.values()) + list(mission_handles.values())
    all_labels = [h.get_label() for h in all_handles]
    fig.legend(all_handles, all_labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.1))

    fig.suptitle("", y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

    # Affichage du tableau récapitulatif des R² -------------------------------
    df_r2 = pd.DataFrame(rows).pivot(index="Ratio", columns="Mission", values="R²").sort_index()
    print("\nTable des R² par ratio :\n", df_r2.to_string())


###############################################################################
# SCRIPT PRINCIPAL
###############################################################################

if __name__ == "__main__":
    # Chemins des fichiers CSV --------------------------------------------------
    files = {"8": "data_Mer8.csv", "15": "data_Mer15.csv"}
    data = {k: pd.read_csv(os.path.join(DATA_FOLDER, v)) for k, v in files.items()}

    # Pré‑traitement -----------------------------------------------------------
    results = {k: process_dataset(df) for k, df in data.items()}

    # Ratios à étudier ---------------------------------------------------------
    ratios = ["Mg/Si", "Ca/Si", "Al/Si"]

    # VISUALISATIONS -----------------------------------------------------------
    plot_interp(results, data, ratios)
    plot_map(results, "pressure_map", "Pression optimale", unit="(GPa)")
    plot_map(results, "F_map", "F optimale", unit="(%)")
    plot_hist_res(results)
    plot_frac_table(results)
    plot_region_counts(results)
    plot_Fopt(results)
    analyze_regions(results)
