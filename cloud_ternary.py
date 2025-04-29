import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lasagne import regions_array

# ─────────────────────────────────────
# Dossiers
# ─────────────────────────────────────
DATA_FOLDER = "data"
OUTPUT_FOLDER = "interactive_diagrams"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────────────────────────────────────
# Fonctions
# ─────────────────────────────────────
def get_pixels(data):
    pix = {}
    for i in range(6):
        mg, ca, al = data[0, i], data[1, i], data[2, i]
        mask = ~np.isnan(mg) & ~np.isnan(ca) & ~np.isnan(al)
        pix[i] = np.vstack([mg[mask], ca[mask], al[mask]]).T
    return pix

def load_exp(filename):
    df = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    df.rename(columns={'Mg/Si': 'MgSi', 'Ca/Si': 'CaSi', 'Al/Si': 'AlSi'}, inplace=True)
    return {
        g: (gdf[['MgSi', 'CaSi', 'AlSi']].values, float(gdf['Pression'].iloc[0]))
        for g, gdf in df.groupby('Groupe')
    }

def add_exp_traces_ternary(fig, dshape, colors, dataset, filename):
    symbol = dshape[dataset]['symbol']
    for coords, press in load_exp(filename).values():
        col = colors.get((dataset, press), 'grey')
        fig.add_trace(
            go.Scatterternary(
                a=coords[:, 0],
                b=coords[:, 1],
                c=coords[:, 2],
                mode='lines+markers',
                line=dict(color=col, width=4),
                marker=dict(symbol=symbol, size=10, color='black'),
                showlegend=False
            )
        )

def compute_region_means(region_pixels):
    means = []
    for i, pts in region_pixels.items():
        mean = np.mean(pts, axis=0)
        means.append((region_names[i], mean, region_colors[i]))
    return means

def compute_pressure_means(datasets, dshape, pressure_colors):
    means = []
    for dset_name, filename in datasets:
        df = pd.read_csv(os.path.join(DATA_FOLDER, filename))
        df.rename(columns={'Mg/Si': 'MgSi', 'Ca/Si': 'CaSi', 'Al/Si': 'AlSi'}, inplace=True)
        for pressure, group in df.groupby('Pression'):
            coords = group[['MgSi', 'CaSi', 'AlSi']].dropna().values
            if len(coords) > 0:
                mean = np.mean(coords, axis=0)
                color = pressure_colors[(dset_name, pressure)]
                means.append((f"{dset_name} {pressure} GPa", mean, color, dshape[dset_name]['mean_symbol']))
    return means

def add_dummy_marker(fig, symbol, name):
    fig.add_trace(
        go.Scatterternary(
            a=[None], b=[None], c=[None], mode='markers',
            marker=dict(symbol=symbol, size=10, color='black'),
            name=name
        )
    )

def add_dummy_line(fig, color, name):
    fig.add_trace(
        go.Scatterternary(
            a=[None], b=[None], c=[None], mode='lines',
            line=dict(color=color, width=4),
            name=name
        )
    )

def create_layout(title):
    return dict(
        title=dict(text=title, font=dict(size=24), y=0.95),
        ternary=dict(
            sum=1,
            aaxis=dict(title='Mg/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'),
            baxis=dict(title='Ca/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'),
            caxis=dict(title='Al/Si', title_font=dict(size=18), tickfont=dict(size=14), showgrid=True, gridcolor='lightgrey'),
            bgcolor='rgb(250,250,250)'
        ),
        legend=dict(font=dict(size=16), bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=40, r=40, t=100, b=80)
    )

# ─────────────────────────────────────
# Paramètres
# ─────────────────────────────────────
region_opacity = 0.6
region_colors = [
    "rgb(255,0,0)", "rgb(0,255,0)", "rgb(0,0,255)",
    "rgb(255,255,0)", "rgb(255,0,255)", "rgb(0,255,255)"
]
region_names = ["High‑Mg", "Al‑rich", "Caloris", "Rach", "High‑Al NVP", "Low‑Al NVP"]

dshape = {
    'Mer8': {'symbol': 'star', 'mean_symbol': 'star'},
    'Mer15': {'symbol': 'diamond', 'mean_symbol': 'diamond'}
}

# Dégradés de couleurs pour les pressions
pressure_colors = {
    ('Mer8', 1.5): "rgb(200,230,255)",   # bleu clair
    ('Mer8', 3.5): "rgb(100,150,255)",   # bleu moyen
    ('Mer8', 5.0): "rgb(0,70,200)",      # bleu foncé
    ('Mer15', 1.5): "rgb(255,220,180)",  # orange clair
    ('Mer15', 3.5): "rgb(255,165,0)",    # orange moyen
    ('Mer15', 5.0): "rgb(200,100,0)"     # orange foncé
}

datasets = [('Mer8', 'data_Mer8.csv'), ('Mer15', 'data_Mer15.csv')]
region_pixels = get_pixels(regions_array())

# ─────────────────────────────────────
# Figure 1 — Courbes expérimentales + Pixels régionaux
# ─────────────────────────────────────
fig1 = go.Figure()

# Pixels régionaux
for i, pts in region_pixels.items():
    fig1.add_trace(go.Scatterternary(
        a=pts[:, 0], b=pts[:, 1], c=pts[:, 2],
        mode='markers',
        name=region_names[i],
        marker=dict(size=5, color=region_colors[i], opacity=region_opacity)
    ))

# Courbes expérimentales
for dset, fname in datasets:
    add_exp_traces_ternary(fig1, dshape, pressure_colors, dset, fname)

# Triangle contour
fig1.add_trace(go.Scatterternary(
    a=[1, 0, 0, 1], b=[0, 1, 0, 0], c=[0, 0, 1, 0],
    mode='lines', line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'
))

# Légende
for dset in dshape:
    add_dummy_marker(fig1, dshape[dset]['symbol'], f'{dset} (courbes)')
for (dset, p), c in pressure_colors.items():
    add_dummy_line(fig1, c, f'{dset} {p} GPa')

# Layout
fig1.update_layout(
    create_layout('Ternaire des compositions expérimentales (Mer8/Mer15) sur fond des régions'),
    annotations=[
        dict(
            text="<b>📌 Astuce :</b><br>Cliquez sur un élément de la légende pour le masquer.<br>Sélectionnez pour zoomer.<br>Double-clic pour dézoomer.",
            x=0.96,
            y=0.5,
            showarrow=False,
            align='left',
            bordercolor='black',
            borderwidth=1,
            bgcolor='white',
            font=dict(size=14),
            xref='paper',
            yref='paper'
        )
    ]
)

fig1.show()
fig1.write_html(os.path.join(OUTPUT_FOLDER, "cloud_ternary_diagram.html"), include_plotlyjs="cdn")

print("✔️ Diagramme ternaire 1 exporté dans", OUTPUT_FOLDER)

# ─────────────────────────────────────
# Figure 2 — Moyennes régionales et pressions
# ─────────────────────────────────────
fig2 = go.Figure()

# Moyennes par région
region_means = compute_region_means(region_pixels)
for name, mean, color in region_means:
    fig2.add_trace(go.Scatterternary(
        a=[mean[0]], b=[mean[1]], c=[mean[2]],
        mode='markers',
        name=f"Région: {name}",
        marker=dict(size=12, symbol='circle', color=color)
    ))

# Moyennes par pression pour Mer8/Mer15
pressure_means = compute_pressure_means(datasets, dshape, pressure_colors)
for name, mean, color, symbol in pressure_means:
    fig2.add_trace(go.Scatterternary(
        a=[mean[0]], b=[mean[1]], c=[mean[2]],
        mode='markers',
        name=name,
        marker=dict(size=14, symbol=symbol, color=color)
    ))

# Triangle contour
fig2.add_trace(go.Scatterternary(
    a=[1, 0, 0, 1], b=[0, 1, 0, 0], c=[0, 0, 1, 0],
    mode='lines', line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'
))

# Layout
fig2.update_layout(
    create_layout('Ternaire des moyennes des régions et des moyennes de chaque pression au sein de Mer8 et Mer15'),
    annotations=[
        dict(
            text="<b>📌 Astuce :</b><br>Cliquez sur un élément de la légende pour le masquer.<br>Sélectionnez pour zoomer.<br>Double-clic pour dézoomer.",
            x=0.96,
            y=0.5,
            showarrow=False,
            align='left',
            bordercolor='black',
            borderwidth=1,
            bgcolor='white',
            font=dict(size=14),
            xref='paper',
            yref='paper'
        )
    ]
)

fig2.show()
fig2.write_html(os.path.join(OUTPUT_FOLDER, "means_ternary_diagram.html"), include_plotlyjs="cdn")

print("✔️ Diagramme ternaire 2 (moyennes) exporté dans", OUTPUT_FOLDER)