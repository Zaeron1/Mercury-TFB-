"""
Diagramme ternaire (Mg/Si – Ca/Si – Al/Si) des pixels régionaux de Mercure
+ courbes expérimentales Mer8 / Mer15.

• Les pixels régionaux sont tracés en premier (opacity réglable)  
• Les courbes expérimentales sont tracées ensuite pour rester visibles  
• Un quadrillage et un contour noir soulignent le triangle
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lasagne import regions_array          # >> votre fonction qui renvoie (3, 6, N)
import os

DATA_FOLDER = "data"

# ──────────────────────────────
# Lectures & helpers
# ──────────────────────────────
def get_pixels(data):
    """Dict {id : (N, 3)} des pixels régionaux Mg/Si, Ca/Si, Al/Si."""
    pixels = {}
    for i in range(6):
        mg, ca, al = data[0, i], data[1, i], data[2, i]
        mask = ~np.isnan(mg) & ~np.isnan(ca) & ~np.isnan(al)
        pixels[i] = np.vstack([mg[mask], ca[mask], al[mask]]).T
    return pixels


def load_exp(filename):
    """Renvoie {groupe : (coords[N, 3], pression_GPa)} pour un CSV expé."""
    path = os.path.join(DATA_FOLDER, filename)
    df = pd.read_csv(path)
    df.rename(columns={'Mg/Si': 'MgSi', 'Ca/Si': 'CaSi', 'Al/Si': 'AlSi'}, inplace=True)
    return {
        g: (gdf[['MgSi', 'CaSi', 'AlSi']].values, float(gdf['Pression'].iloc[0]))
        for g, gdf in df.groupby('Groupe')
    }


def add_exp_traces_ternary(fig, dshape, colors, dataset, filename):
    """Courbes expérimentales dans le ternaire."""
    symbol = dshape[dataset][0]
    for coords, press in load_exp(filename).values():
        col = colors.get(press, 'grey')           # couleurs ≠ couleurs régionales
        fig.add_trace(
            go.Scatterternary(
                a=coords[:, 0],
                b=coords[:, 1],
                c=coords[:, 2],
                mode='lines+markers',
                line=dict(color=col, width=3),
                marker=dict(symbol=symbol, size=6, color=col),
                showlegend=False,
            )
        )


def legend_mark_ternary(fig, symbol, name):
    """Marque factice pour légende (forme)."""
    fig.add_trace(
        go.Scatterternary(
            a=[None], b=[None], c=[None], mode='markers',
            marker=dict(symbol=symbol, size=8, color='black'),
            name=name, showlegend=True
        )
    )


# ──────────────────────────────
# Paramètres
# ──────────────────────────────
region_opacity = 0.6
# ► Couleurs des courbes : noir / gris / magenta (≠ rouges‑bleues‑vertes rég.)
pressure_colors = {1.5: 'black', 3.5: 'grey', 5.0: 'magenta'}
dshape = {'Mer8': ('circle', 'sphère'), 'Mer15': ('diamond', 'losange')}

region_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
region_names = [
    "high-Mg", "Al-rich", "Caloris", "Rach", "high-al NVP", "low-al NVP"
]
region_pixels = get_pixels(regions_array())

# ──────────────────────────────
# Figure ternaire
# ──────────────────────────────
fig = go.Figure()

# Nuages régionaux (d’abord)
for i, pts in region_pixels.items():
    fig.add_trace(
        go.Scatterternary(
            a=pts[:, 0], b=pts[:, 1], c=pts[:, 2],
            mode='markers',
            name=region_names[i],
            marker=dict(size=4, color=region_colors[i], opacity=region_opacity)
        )
    )

# Courbes expérimentales (ensuite)
for dset, fname in [('Mer8', 'data_Mer8.csv'), ('Mer15', 'data_Mer15.csv')]:
    add_exp_traces_ternary(fig, dshape, pressure_colors, dset, fname)

# Contour noir du triangle (en dernier pour être au‑dessus de tout)
fig.add_trace(
    go.Scatterternary(
        a=[1, 0, 0, 1],
        b=[0, 1, 0, 0],
        c=[0, 0, 1, 0],
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False,
        hoverinfo='skip'
    )
)

# Légendes : formes experim. + pressions
for dset in dshape:
    legend_mark_ternary(fig, dshape[dset][0], f'{dset} ({dshape[dset][1]})')
for p, c in pressure_colors.items():
    fig.add_trace(
        go.Scatterternary(
            a=[None], b=[None], c=[None],
            mode='lines', line=dict(color=c, width=3),
            name=f'{p} GPa'
        )
    )

# Mise en forme générale
fig.update_layout(
    title='Courbes expérimentales Mer8/Mer15 + pixels régionaux (ternaire)',
    ternary=dict(
        sum=1,  # normalisation automatique
        aaxis=dict(title='Mg/Si', showgrid=True, gridcolor='lightgrey'),
        baxis=dict(title='Ca/Si', showgrid=True, gridcolor='lightgrey'),
        caxis=dict(title='Al/Si', showgrid=True, gridcolor='lightgrey'),
        bgcolor='rgb(250,250,250)'
    ),
    legend=dict(x=0, y=0, bgcolor='rgba(255,255,255,0.75)'),
    margin=dict(l=30, r=30, t=90, b=30)
)

fig.show()
OUTPUT_FOLDER = "interactive_diagrams"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Crée le dossier s'il n'existe pas
fig.write_html(os.path.join(OUTPUT_FOLDER, "cloud_ternary_diagram.html"), include_plotlyjs='cdn')