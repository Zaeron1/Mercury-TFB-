import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
import os

DATA_FOLDER = "data"

# ============================================================
# Fonctions utilitaires
# ============================================================

def idw_interpolation(query_points, points, values, power=2):
    interpolated = []
    for qp in query_points:
        d = np.linalg.norm(points - qp, axis=1)
        if np.any(d < 1e-8):
            interpolated.append(values[d.argmin()])
        else:
            w = 1 / d**power
            interpolated.append(np.sum(w * values) / np.sum(w))
    return np.array(interpolated)


def compute_curve(df_subset, n_points=200, power=2):
    X = df_subset[["Mg/Si", "Ca/Si", "Al/Si"]].values
    F_exp = df_subset["F"].values
    t = PCA(1).fit_transform(X).flatten()
    order = np.argsort(t)
    t_sorted = t[order]
    splines = [UnivariateSpline(t_sorted, X[order, i], s=len(t_sorted)) for i in range(3)]
    t_fine = np.linspace(t_sorted.min(), t_sorted.max(), n_points)
    curve = np.vstack([s(t_fine) for s in splines]).T
    F_curve = idw_interpolation(curve, X, F_exp, power=power)
    return curve, F_curve

# ============================================================
# Chargement des données
# ============================================================

datasets = {
    "8": pd.read_csv(os.path.join(DATA_FOLDER, "data_Mer8.csv")),
    "15": pd.read_csv(os.path.join(DATA_FOLDER, "data_Mer15.csv")),
}
pressures = [1.5, 3.5, 5.0]

# ============================================================
# Styles : palettes claires + positions des barres de couleur
# ============================================================

cmap = {
    ("8", 1.5): "YlGnBu",
    ("8", 3.5): "BuGn",
    ("8", 5.0): "GnBu",
    ("15", 1.5): "YlOrBr",
    ("15", 3.5): "OrRd",
    ("15", 5.0): "RdPu",
}
marker_shape = {"8": "circle", "15": "diamond"}
bar_positions = {("8", 1.5): 0.75, ("8", 3.5): 0.80, ("8", 5.0): 0.85,
                 ("15", 1.5): 0.90, ("15", 3.5): 0.95, ("15", 5.0): 1.00}
bar_thickness = 12
bar_length = 0.5

# Annotations verticales et titre des colorbars
annotations = []
# Titre général au-dessus des barres de couleur (taille augmentée ici)
annotations.append(dict(
    x=0.96, y=0.82, xref="paper", yref="paper",
    text="Taux de fusion partielle (%)", showarrow=False,
    font=dict(size=20, color="black")  # <-- augmenté ici
))
# Légendes Mer/pression
for (mer, p), xpos in bar_positions.items():
    annotations.append(dict(
        x=xpos, y=0.60, xref="paper", yref="paper",
        text=f"{p} GPa (Mer{mer})", textangle=-90,
        showarrow=False, font=dict(size=18, color="dimgray")  # <-- augmenté ici
    ))

# ============================================================
# Construction de la figure
# ============================================================
fig = go.Figure()
for mer, df in datasets.items():
    for p in pressures:
        sub = df[df["Pression"] == p]
        if sub.empty:
            continue
        pts = sub[["Mg/Si", "Ca/Si", "Al/Si"]].values
        Fvals = sub["F"].values
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(symbol=marker_shape[mer], size=7, opacity=0.8,
                        line=dict(width=0.5, color="grey"),
                        color=Fvals, colorscale=cmap[(mer, p)], cmin=0, cmax=100),
            name=f"{p} GPa (Mer{8})", legendgroup=f"M{mer}P{p}"
        ))
        curve, Fcurve = compute_curve(sub)
        fig.add_trace(go.Scatter3d(
            x=curve[:,0], y=curve[:,1], z=curve[:,2], mode="lines",
            line=dict(width=12, color=Fcurve, colorscale=cmap[(mer, p)], cmin=0, cmax=100),
            showlegend=False, legendgroup=f"M{mer}P{p}"
        ))
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(colorscale=cmap[(mer, p)], cmin=0, cmax=100,
                        showscale=True,
                        colorbar=dict(len=bar_length, thickness=bar_thickness,
                                      y=0.5, x=bar_positions[(mer, p)],
                                      tickmode='array', tickvals=[0,100], ticktext=['0','100'],
                                      tickfont=dict(size=12),  # <-- augmenté ici
                                      outlinewidth=0)),
            showlegend=False
        ))
# Layout
fig.update_layout(
    title="Tendances dans les compositions expérimentales (Mer8 et Mer15)<br>en corrélation avec le taux fusion partielle",
    scene=dict(
        xaxis_title="Mg/Si",
        yaxis_title="Ca/Si",
        zaxis_title="Al/Si",
        bgcolor="rgb(250,250,250)"
    ),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)", borderwidth=0),
    margin=dict(l=30, r=100, t=100, b=30),
    annotations=annotations + [
        dict(
            text="<b>📌 Astuce :</b><br>Cliquez sur un élément de la légende pour le masquer.<br>Sélectionnez pour zoomer.<br>Double-clic pour dézoomer.",
            x=1.08,
            y=0,
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
fig.show()
OUTPUT_FOLDER = "interactive_diagrams"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
fig.write_html(os.path.join(OUTPUT_FOLDER, "FP.html"), include_plotlyjs='cdn')