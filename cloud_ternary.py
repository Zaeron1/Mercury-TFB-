"""
Diagrammes interactifs Mercure :
  1. Figure ternaire : pixels régionaux + courbes expé Mer8/Mer15 (detailed)
  2. Figure ternaire "moyennes" : un point par région + un point par (pression, jeu de données)
  3. Figures 2D (Ca/Si vs Mg/Si, Al/Si vs Mg/Si) et 3D sont générées ailleurs

Couleurs & symboles
──────────────────
  Régions   : couleurs vives R‑G‑B‑Y‑M‑C
  Pressions : 1 · 5 GPa → orange  | 3 · 5 GPa → violet | 5 GPa → turquoise
  Jeux      : Mer8 → cercle noir  | Mer15 → losange noir (bord noir, remplissage couleur pression)
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lasagne import regions_array  # (3 axes × 6 régions × N)

# ──────────────────────────────
# Dossiers
# ──────────────────────────────
DATA_FOLDER = "data"
OUTPUT_FOLDER = "interactive_diagrams"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ──────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────

def get_pixels(data):
    """Dict {id: (N,3)} pour les pixels régionaux."""
    pix = {}
    for i in range(6):
        mg, ca, al = data[0, i], data[1, i], data[2, i]
        mask = ~np.isnan(mg) & ~np.isnan(ca) & ~np.isnan(al)
        pix[i] = np.vstack([mg[mask], ca[mask], al[mask]]).T
    return pix

def get_region_means(pix_dict):
    """Renvoie array (6,3) des moyennes Mg, Ca, Al pour chaque région."""
    means = []
    for i in range(6):
        means.append(pix_dict[i].mean(axis=0))
    return np.asarray(means)


def load_exp(filename):
    """Renvoie {groupe: (coords[N,3], pression)}."""
    df = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    df = df.rename(columns={"Mg/Si": "MgSi", "Ca/Si": "CaSi", "Al/Si": "AlSi"})
    return {
        g: (gdf[["MgSi", "CaSi", "AlSi"]].values, float(gdf["Pression"].iloc[0]))
        for g, gdf in df.groupby("Groupe")
    }

def load_exp_means(filename):
    """Renvoie {pression: mean_coords(3,)} sur l'ensemble du fichier."""
    df = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    df = df.rename(columns={"Mg/Si": "MgSi", "Ca/Si": "CaSi", "Al/Si": "AlSi"})
    means = {}
    for p, pdf in df.groupby("Pression"):
        means[float(p)] = pdf[["MgSi", "CaSi", "AlSi"]].mean().values
    return means  # dict press → np.array(3,)


def add_exp_traces_ternary(fig, symbol, filename, press_colors):
    """Lignes + markers noirs (courbes complètes) ; utilisé pour la figure détaillée."""
    for coords, press in load_exp(filename).values():
        colour = press_colors.get(press, 'grey')
        fig.add_trace(go.Scatterternary(
            a=coords[:, 0], b=coords[:, 1], c=coords[:, 2],
            mode='lines+markers',
            line=dict(color=colour, width=4),
            marker=dict(symbol=symbol, size=10, color='black'),
            showlegend=False))

# ──────────────────────────────
# Paramètres communs
# ──────────────────────────────
region_opacity = 0.6
region_colors = [
    "rgb(255,0,0)",   # High‑Mg
    "rgb(0,255,0)",   # Al‑rich
    "rgb(0,0,255)",   # Caloris
    "rgb(255,255,0)", # Rach
    "rgb(255,0,255)", # High‑Al NVP
    "rgb(0,255,255)"  # Low‑Al NVP
]
region_names = ["high‑Mg", "Al‑rich", "Caloris", "Rach", "high‑Al NVP", "low‑Al NVP"]
pressure_colors = {1.5: "rgb(255,165,0)", 3.5: "rgb(128,0,128)", 5.0: "rgb(0,191,255)"}
dshape = {"Mer8": ("circle", "sphère"), "Mer15": ("diamond", "losange")}

# Pré‑calculs
region_pixels = get_pixels(regions_array())
region_means = get_region_means(region_pixels)   # (6,3)
mer8_means = load_exp_means("data_Mer8.csv")
mer15_means = load_exp_means("data_Mer15.csv")

# ╭──────────────────────────────────────────────────────────╮
# │ FIGURE TERNAIRE DÉTAILLÉE  (pixels + courbes complètes) │
# ╰──────────────────────────────────────────────────────────╯
fig_det = go.Figure()
# Pixels régionaux
for i, pts in region_pixels.items():
    fig_det.add_trace(go.Scatterternary(
        a=pts[:, 0], b=pts[:, 1], c=pts[:, 2],
        mode='markers', name=region_names[i],
        marker=dict(size=5, color=region_colors[i], opacity=region_opacity)))
# Courbes expé
a dset_files = [("Mer8", "data_Mer8.csv"), ("Mer15", "data_Mer15.csv")]
for dset, fcsv in dset_files:
    add_exp_traces_ternary(fig_det, dshape[dset][0], fcsv, pressure_colors)
# Contour & légendes
fig_det.add_trace(go.Scatterternary(a=[1,0,0,1], b=[0,1,0,0], c=[0,0,1,0], mode='lines', line=dict(color='black', width=3), hoverinfo='skip', showlegend=False))
for dset in dshape:
    fig_det.add_trace(go.Scatterternary(a=[None], b=[None], c=[None], mode='markers', marker=dict(symbol=dshape[dset][0], size=10, color='black'), name=f"{dset} ({dshape[dset][1]})"))
for p, col in pressure_colors.items():
    fig_det.add_trace(go.Scatterternary(a=[None], b=[None], c=[None], mode='lines', line=dict(color=col, width=4), name=f"{p} GPa"))
fig_det.update_layout(title=dict(text='Ternaire complet : pixels + courbes Mer8/Mer15', font=dict(size=24), y=0.95), ternary=dict(sum=1, aaxis=dict(title='Mg/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'), baxis=dict(title='Ca/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'), caxis=dict(title='Al/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'), bgcolor='rgb(250,250,250)'), legend=dict(font=dict(size=16), bgcolor='rgba(255,255,255,0.8)'), margin=dict(l=40, r=40, t=100, b=80))
fig_det.write_html(os.path.join(OUTPUT_FOLDER, "cloud_ternary_detailed.html"), include_plotlyjs='cdn')

# ╭──────────────────────────────────────────────────────────╮
# │ FIGURE TERNAIRE MOYENNES  (un point par région & press) │
# ╰──────────────────────────────────────────────────────────╯
fig_mean = go.Figure()

# 1) Moyennes régionales
for i, coord in enumerate(region_means):
    fig_mean.add_trace(go.Scatterternary(
        a=[coord[0]], b=[coord[1]], c=[coord[2]],
        mode='markers', name=f"{region_names[i]} (moy.)",
        marker=dict(size=12, color=region_colors[i], symbol='circle', line=dict(color='black', width=1))))

# 2) Moyennes par pression & jeu de données
for dset, means_dict in [("Mer8", mer8_means), ("Mer15", mer15_means)]:
    symb = dshape[dset][0]
    for press, coord in means_dict.items():
        fig_mean.add_trace(go.Scatterternary(
            a=[coord[0]], b=[coord[1]], c=[coord[2]],
            mode='markers',
            name=f"{dset} {press} GPa",
            marker=dict(size=13, color=pressure_colors.get(press, 'grey'), symbol=symb, line=dict(color='black', width=1))))

# Contour du triangle
fig_mean.add_trace(go.Scatterternary(a=[1,0,0,1], b=[0,1,0,0], c=[0,0,1,0], mode='lines', line=dict(color='black', width=3), hoverinfo='skip', showlegend=False))

# Layout
fig_mean.update_layout(
    title=dict(text='Ternaire des moyennes : régions vs pressions Mer8/Mer15', font=dict(size=24), y=0.95),
    ternary=dict(sum=1, aaxis=dict(title='Mg/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'), baxis=dict(title='Ca/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'), caxis=dict(title='Al/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'), bgcolor='rgb(250,250,250)'),
    legend=dict(font=dict(size=16), bgcolor='rgba(255,255,255,0.85)'),
    margin=dict(l=40, r=40, t=100, b=80)
)

fig_mean.write_html(os.path.join(OUTPUT_FOLDER, "cloud_ternary_means.html"), include_plotlyjs='cdn')

print("✔️  Deux diagrammes ternaires exportés dans", OUTPUT_FOLDER)
