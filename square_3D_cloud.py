"""
Affichage interactif des courbes expérimentales Mer8/Mer15 et des nuages de points régionaux de Mercure :
- 1 figure 3D (Mg/Si, Ca/Si, Al/Si)
- 2 diagrammes 2D : Ca/Si vs Mg/Si et Al/Si vs Mg/Si
Les courbes Mer8/Mer15 apparaissent devant les nuages ; leur transparence est réglable par `region_opacity`.

⚠️  Cette version :
  • augmente la police via Plotly (`update_layout`).
  • affiche les marqueurs Mer8 ⬤ et Mer15 ◇ en **noir** et plus grands dans les 2D.
  • conserve les lignes colorées selon la pression (1.5, 3.5, 5 GPa).
  • ajoute **tous les éléments manquants dans la légende** : lignes de pression + marqueurs Mer8/Mer15.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lasagne import regions_array  # <-- ta fonction qui renvoie le tableau 3×6×N

# ─────────────────────────────────────────────────────────────
# Dossiers I/O
# ─────────────────────────────────────────────────────────────
DATA_FOLDER = "data"
OUTPUT_FOLDER = "interactive_diagrams"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────

def get_pixels(data):
    """Convertit le tableau (3 axes × 6 régions × N) en dict {id: (N,3)}."""
    pix = {}
    for i in range(6):
        mg, ca, al = data[0, i], data[1, i], data[2, i]
        mask = ~np.isnan(mg) & ~np.isnan(ca) & ~np.isnan(al)
        pix[i] = np.vstack([mg[mask], ca[mask], al[mask]]).T
    return pix

def load_exp(filename):
    """Charge un CSV expérimental et regroupe par colonne *Groupe*."""
    df = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    df = df.rename(columns={"Mg/Si": "MgSi", "Ca/Si": "CaSi", "Al/Si": "AlSi"})
    return {
        g: (gdf[["MgSi", "CaSi", "AlSi"]].values, float(gdf["Pression"].iloc[0]))
        for g, gdf in df.groupby("Groupe")
    }

def add_exp_2d(fig, symbol, filename, press_colors, dim_x, dim_y):
    """Trace en 2D : ligne colorée (pression) + marqueurs noirs agrandis."""
    for coords, press in load_exp(filename).values():
        colour = press_colors.get(press, "grey")
        # ligne colorée
        fig.add_trace(
            go.Scatter(x=coords[:, dim_x], y=coords[:, dim_y], mode="lines",
                        line=dict(color=colour, width=5), showlegend=False))
        # marqueurs noirs (plus grands)
        fig.add_trace(
            go.Scatter(x=coords[:, dim_x], y=coords[:, dim_y], mode="markers",
                        marker=dict(symbol=symbol, size=11, color="black"), showlegend=False))

def add_exp_3d(fig, symbol, filename, press_colors):
    """Trace en 3D (lignes+points) ; points et lignes gardent la même couleur (pression)."""
    for coords, press in load_exp(filename).values():
        colour = press_colors.get(press, "grey")
        fig.add_trace(
            go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                          mode="lines+markers",
                          line=dict(color=colour, width=5),
                          marker=dict(symbol=symbol, size=6, color=colour),
                          showlegend=False))

def add_dummy_marker(fig, symbol, name, is3d=False):
    """Ajoute un marqueur noir factice pour la légende."""
    if is3d:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                                   marker=dict(symbol=symbol, size=8, color="black"),
                                   name=name))
    else:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(symbol=symbol, size=11, color="black"),
                                 name=name))

def add_dummy_line(fig, colour, name, is3d=False):
    """Ajoute une ligne factice colorée pour la légende."""
    if is3d:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                                   line=dict(color=colour, width=3), name=name))
    else:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=colour, width=2), name=name))

# ─────────────────────────────────────────────────────────────
# Paramètres globaux
# ─────────────────────────────────────────────────────────────
region_opacity = 0.6
pressure_colors = {
    1.5: "rgb(255,165,0)",  # orange
    3.5: "rgb(128,0,128)",  # violet
    5.0: "rgb(0,191,255)"   # turquoise foncé
}
dshapes = {"Mer8": ("circle", "sphère"), "Mer15": ("diamond", "losange")}
region_colors = [
    "rgb(255,0,0)",      # High-Mg (rouge vif)
    "rgb(0,255,0)",      # Al-rich (vert vif)
    "rgb(0,0,255)",      # Caloris (bleu vif)
    "rgb(255,255,0)",    # Rach (jaune vif)
    "rgb(255,0,255)",    # High-al NVP (magenta vif)
    "rgb(0,255,255)"     # Low-al NVP (cyan vif)
]
region_names = ["high‑Mg", "Al‑rich", "Caloris", "Rach", "high‑Al NVP", "low‑Al NVP"]
region_pixels = get_pixels(regions_array())

# ─────────────────────────────────────────────────────────────
# === FIGURE 3D ===
# ─────────────────────────────────────────────────────────────
fig3d = go.Figure()

for dset, csv in [("Mer8", "data_Mer8.csv"), ("Mer15", "data_Mer15.csv")]:
    add_exp_3d(fig3d, dshapes[dset][0], csv, pressure_colors)

for i, pts in region_pixels.items():
    fig3d.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                 mode="markers", name=region_names[i],
                                 marker=dict(size=3, color=region_colors[i], opacity=region_opacity)))

# Légende fictive (marqueurs noirs + lignes pression)
add_dummy_marker(fig3d, "circle", "Mer8 (sphère)", is3d=True)
add_dummy_marker(fig3d, "diamond", "Mer15 (losange)", is3d=True)
for p, c in pressure_colors.items():
    add_dummy_line(fig3d, c, f"{p} GPa", is3d=True)

fig3d.update_layout(
    title="Mer8 / Mer15 + pixels régionaux (3D)",
    title_font=dict(size=22),
    scene=dict(
        xaxis=dict(title="Mg/Si", title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title="Ca/Si", title_font=dict(size=18), tickfont=dict(size=14)),
        zaxis=dict(title="Al/Si", title_font=dict(size=18), tickfont=dict(size=14)),
        bgcolor="rgb(250,250,250)"
    ),
    legend=dict(font=dict(size=20)),
    margin=dict(l=30, r=30, t=80, b=30)
)
fig3d.write_html(os.path.join(OUTPUT_FOLDER, "cloud_3D_diagram.html"), include_plotlyjs="cdn")

# ─────────────────────────────────────────────────────────────
# === FIGURE 2D : Ca/Si vs Mg/Si ===
# ─────────────────────────────────────────────────────────────
fig_ca = go.Figure()
for i, pts in region_pixels.items():
    fig_ca.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers",
                                name=region_names[i],
                                marker=dict(size=5, color=region_colors[i], opacity=region_opacity)))
for dset, csv in [("Mer8", "data_Mer8.csv"), ("Mer15", "data_Mer15.csv")]:
    add_exp_2d(fig_ca, dshapes[dset][0], csv, pressure_colors, 0, 1)

# Légende fictive
add_dummy_marker(fig_ca, "circle", "Mer8 (sphère)")
add_dummy_marker(fig_ca, "diamond", "Mer15 (losange)")
for p, c in pressure_colors.items():
    add_dummy_line(fig_ca, c, f"{p} GPa")

fig_ca.update_layout(
    title="Ca/Si vs Mg/Si", title_font=dict(size=22),
    xaxis=dict(title="Mg/Si", title_font=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title="Ca/Si", title_font=dict(size=18), tickfont=dict(size=14)),
    legend=dict(font=dict(size=20))
)
fig_ca.write_html(os.path.join(OUTPUT_FOLDER, "cloud_Ca_diagram.html"), include_plotlyjs="cdn")

# ─────────────────────────────────────────────────────────────
# === FIGURE 2D : Al/Si vs Mg/Si ===
# ─────────────────────────────────────────────────────────────
fig_al = go.Figure()
for i, pts in region_pixels.items():
    fig_al.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 2], mode="markers",
                                name=region_names[i],
                                marker=dict(size=5, color=region_colors[i], opacity=region_opacity)))
for dset, csv in [("Mer8", "data_Mer8.csv"), ("Mer15", "data_Mer15.csv")]:
    add_exp_2d(fig_al, dshapes[dset][0], csv, pressure_colors, 0, 2)

add_dummy_marker(fig_al, "circle", "Mer8 (sphère)")
add_dummy_marker(fig_al, "diamond", "Mer15 (losange)")
for p, c in pressure_colors.items():
    add_dummy_line(fig_al, c, f"{p} GPa")

fig_al.update_layout(
    title="Al/Si vs Mg/Si", title_font=dict(size=22),
    xaxis=dict(title="Mg/Si", title_font=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title="Al/Si", title_font=dict(size=18), tickfont=dict(size=14)),
    legend=dict(font=dict(size=20))
)
fig_al.write_html(os.path.join(OUTPUT_FOLDER, "cloud_Al_diagram.html"), include_plotlyjs="cdn")

print("✔️  Figures interactives générées dans", OUTPUT_FOLDER)
