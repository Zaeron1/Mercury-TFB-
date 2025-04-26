"""
Affichage interactif des courbes expérimentales Mer8/Mer15 et des nuages de points régionaux de Mercure :
- 1 figure 3D (Mg/Si, Ca/Si, Al/Si)
- 2 diagrammes 2D : Ca/Si vs Mg/Si et Al/Si vs Mg/Si
Les courbes sont affichées devant les nuages. Transparence réglable par `region_opacity`.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from lasagne import regions_array

# ─────────────────────────────────────────────────────────────
# Dossiers
# ─────────────────────────────────────────────────────────────
DATA_FOLDER = "data"
OUTPUT_FOLDER = "interactive_diagrams"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────

def get_pixels(data):
    """Retourne {id : (N, 3)} pour les pixels régionaux Mg/Si, Ca/Si, Al/Si."""
    pixels = {}
    for i in range(6):
        mg, ca, al = data[0, i], data[1, i], data[2, i]
        mask = ~np.isnan(mg) & ~np.isnan(ca) & ~np.isnan(al)
        pixels[i] = np.vstack([mg[mask], ca[mask], al[mask]]).T
    return pixels

def load_exp(filename):
    """Charge un fichier expérimental en formatant les colonnes."""
    path = os.path.join(DATA_FOLDER, filename)
    df = pd.read_csv(path)
    df.rename(columns={'Mg/Si': 'MgSi', 'Ca/Si': 'CaSi', 'Al/Si': 'AlSi'}, inplace=True)
    return {g: (gdf[['MgSi', 'CaSi', 'AlSi']].values, float(gdf['Pression'].iloc[0]))
            for g, gdf in df.groupby('Groupe')}

def add_exp_traces(fig, dshape, colors, dataset, filename, dim_x, dim_y):
    """Ajoute des courbes expérimentales sur une figure 2D."""
    symbol = dshape[dataset][0]
    for coords, press in load_exp(filename).values():
        col = colors.get(press, 'grey')
        fig.add_trace(go.Scatter(
            x=coords[:, dim_x], y=coords[:, dim_y],
            mode='lines+markers',
            line=dict(color=col, width=3),
            marker=dict(symbol=symbol, size=6, color=col),
            showlegend=False
        ))

def add_exp_traces_3d(fig, dshape, colors, dataset, filename):
    """Ajoute des courbes expérimentales sur une figure 3D."""
    symbol = dshape[dataset][0]
    for coords, press in load_exp(filename).values():
        col = colors.get(press, 'grey')
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='lines+markers',
            line=dict(color=col, width=3),
            marker=dict(symbol=symbol, size=6, color=col),
            showlegend=False
        ))

def legend_mark_2d(fig, symbol, name):
    """Ajoute une marque factice pour légende en 2D."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(symbol=symbol, size=8, color='black'),
                             name=name, showlegend=True))

def legend_mark_3d(fig, symbol, name):
    """Ajoute une marque factice pour légende en 3D."""
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                               marker=dict(symbol=symbol, size=8, color='black'),
                               name=name, showlegend=True))

# ─────────────────────────────────────────────────────────────
# Script principal
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Paramètres
    region_opacity = 0.6
    pressure_colors = {1.5: 'blue', 3.5: 'green', 5.0: 'red'}
    dshape = {'Mer8': ('circle', 'sphère'), 'Mer15': ('diamond', 'losange')}
    region_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    region_names = ["high-Mg", "Al-rich", "Caloris", "Rach", "high-al NVP", "low-al NVP"]
    region_pixels = get_pixels(regions_array())

    # ─── Figure 3D ──────────────────────────────────────────────
    fig3d = go.Figure()

    # Courbes expérimentales
    for dset, fname in [('Mer8', 'data_Mer8.csv'), ('Mer15', 'data_Mer15.csv')]:
        add_exp_traces_3d(fig3d, dshape, pressure_colors, dset, fname)

    # Nuages régionaux
    for i, pts in region_pixels.items():
        fig3d.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                     mode='markers', name=region_names[i],
                                     marker=dict(size=3, color=region_colors[i],
                                                 opacity=region_opacity, symbol='circle')))

    # Légendes
    for dset in dshape:
        legend_mark_3d(fig3d, dshape[dset][0], f'{dset} ({dshape[dset][1]})')
    for p, c in pressure_colors.items():
        fig3d.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                                     line=dict(color=c, width=3), name=f'{p} GPa'))

    # Layout
    fig3d.update_layout(
        title="Courbes expérimentales Mer8/Mer15 + pixels régionaux (3D)",
        scene=dict(xaxis_title='Mg/Si', yaxis_title='Ca/Si', zaxis_title='Al/Si',
                   bgcolor='rgb(250,250,250)'),
        legend=dict(x=0, y=0, bgcolor='rgba(255,255,255,0.75)'),
        margin=dict(l=30, r=30, t=90, b=30)
    )

    fig3d.show()
    fig3d.write_html(os.path.join(OUTPUT_FOLDER, "cloud_3D_diagram.html"), include_plotlyjs="cdn")

    # ─── Figure 2D Ca/Si vs Mg/Si ───────────────────────────────
    fig2d_ca = go.Figure()

    for i, pts in region_pixels.items():
        fig2d_ca.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='markers',
                                      name=region_names[i],
                                      marker=dict(size=5, color=region_colors[i],
                                                  opacity=region_opacity)))

    for dset, fname in [('Mer8', 'data_Mer8.csv'), ('Mer15', 'data_Mer15.csv')]:
        add_exp_traces(fig2d_ca, dshape, pressure_colors, dset, fname, 0, 1)

    for dset in dshape:
        legend_mark_2d(fig2d_ca, dshape[dset][0], f'{dset} ({dshape[dset][1]})')
    for p, c in pressure_colors.items():
        fig2d_ca.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                      line=dict(color=c, width=3), name=f'{p} GPa'))

    fig2d_ca.update_layout(
        title='Courbes expérimentales Mer8/Mer15 + pixels régionaux (Ca/Si vs Mg/Si)',
        xaxis_title='Mg/Si', yaxis_title='Ca/Si'
    )

    fig2d_ca.show()
    fig2d_ca.write_html(os.path.join(OUTPUT_FOLDER, "cloud_Ca_diagram.html"), include_plotlyjs="cdn")

    # ─── Figure 2D Al/Si vs Mg/Si ───────────────────────────────
    fig2d_al = go.Figure()

    for i, pts in region_pixels.items():
        fig2d_al.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 2], mode='markers',
                                      name=region_names[i],
                                      marker=dict(size=5, color=region_colors[i],
                                                  opacity=region_opacity)))

    for dset, fname in [('Mer8', 'data_Mer8.csv'), ('Mer15', 'data_Mer15.csv')]:
        add_exp_traces(fig2d_al, dshape, pressure_colors, dset, fname, 0, 2)

    for dset in dshape:
        legend_mark_2d(fig2d_al, dshape[dset][0], f'{dset} ({dshape[dset][1]})')
    for p, c in pressure_colors.items():
        fig2d_al.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                      line=dict(color=c, width=3), name=f'{p} GPa'))

    fig2d_al.update_layout(
        title='Courbes expérimentales Mer8/Mer15 + pixels régionaux (Al/Si vs Mg/Si)',
        xaxis_title='Mg/Si', yaxis_title='Al/Si'
    )

    fig2d_al.show()
    fig2d_al.write_html(os.path.join(OUTPUT_FOLDER, "cloud_Al_diagram.html"), include_plotlyjs="cdn")