%matplotlib qt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from nilearn.plotting import plot_connectome


###### fig 1_A ######
fname = Path.cwd().parent / "identifiability_matrix" / "graph_comparison.csv"
df = pd.read_csv(fname)
vals = [0.25, 0.5, 1, 1.5]
df = df.query("alpha in @vals and beta in @vals")

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(df, col="frequency_range",  row="alpha", hue="beta", height=2.5, aspect=1.3, palette="YlGnBu_d", legend_out=True)
kwargs = {"markersize": 2.5, "lw": 1.5}
g.map_dataframe(sns.pointplot, x="times_used", y="matching_score", **kwargs)
g.set(xticklabels=[], title="", ylabel="", xlabel="")
legend_kwargs = {"bbox_to_anchor": (0.5, 0.7, 0.5, 0.5)}
g.add_legend(**legend_kwargs)
g.tight_layout()
g.savefig(Path.cwd().parent / "figures" / "2.pdf")

###### fig 1_B ######
fname = Path.cwd().parent / "identifiability_matrix" / "ranks.csv"
df = pd.read_csv(fname)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
color_1 = sns.cubehelix_palette(10, rot=-.25, light=.7)[3]
color_2 = list(sns.hls_palette()[5])
g = sns.FacetGrid(df, col="group",  row="freq_band", hue="method", height=2.5, aspect=1.7, palette=[color_1, color_2], legend_out=True)
kwargs = {"element": "step", "discrete": True}
g.map(sns.histplot, "rank", **kwargs)
legend_kwargs = {"bbox_to_anchor": (0.5, 0.6, 0.5, 0.5)}
g.add_legend(**legend_kwargs)
g.tight_layout()
g.savefig(Path.cwd().parent / "figures" / "1.pdf")

###### fig 1_C ######
fname = Path.cwd().parent / "identifiability_matrix" / "connections.csv"
df = pd.read_csv(fname)
df_alpha_1 = df[(df["freq_band"]=="alpha") & (df["label_2"]=="superiorfrontal-lh")]
df_alpha_2 = df[(df["freq_band"]=="alpha") & (df["label_2"]=="superiorfrontal-rh")]

df_beta_1 = df[(df["freq_band"]=="beta") & (df["label_2"]=="rostralmiddlefrontal-lh")]
df_beta_2 = df[(df["freq_band"]=="beta") & (df["label_2"]=="lateraloccipital-lh")]

df_gamma_1 = df[(df["freq_band"]=="gamma") & (df["label_2"]=="parsorbitalis-lh")]
df_gamma_2 = df[(df["freq_band"]=="gamma") & (df["label_2"]=="superiorfrontal-lh")]
df_gamma_3 = df[(df["freq_band"]=="gamma") & (df["label_2"]=="superiorfrontal-rh")]

dfs = [df_alpha_1, df_alpha_2, df_beta_1, df_beta_2, df_gamma_1, df_gamma_2, df_gamma_3]
palette_color = ['#1f77b4', '#d62728']
hue_order = ["control", "case"]

lefts = [0.05, 0.15, 0.30, 0.40, 0.55, 0.65, 0.75]
fig = plt.figure(figsize=(13, 3))
axs = []
for left in lefts:
    ax = fig.add_axes([left, 0.1, 0.09, 0.7])
    axs.append(ax)
for ax_id, df_sub, ax in zip(range(len(dfs)), dfs, axs):
    sns.boxplot(data=df_sub, x="group", y="connection_strength", hue="group",
                hue_order=hue_order, palette=palette_color, fill=False, width=0.6,
                gap=0, linewidth=1.8, ax=ax)
    sns.stripplot(data=df_sub, x="group", y="connection_strength", hue="group", 
                hue_order=hue_order, palette=palette_color, linewidth=0, size=2.6,
                edgecolor=None, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("")
    ax.set_ylim([None, 0.04])
    if ax_id in [1, 3, 5, 6]:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
fig.tight_layout()
fig.savefig(Path.cwd().parent / "figures" / "4.pdf")

# optional
dfs = [df_alpha_1, df_alpha_2, df_beta_1, df_beta_2, df_gamma_1, df_gamma_2, df_gamma_3]
mean_strengths = []
for df in dfs:
    a = df[df['group'] == 'case']['connection_strength'].mean()
    b = df[df['group'] == 'control']['connection_strength'].mean()
    mean_strengths.append(float(a) - float(b))

###### fig 1_C ######
labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc', subjects_dir=None, verbose=False)[:-1]
node_coords = []
for label in labels:
    if label.hemi == 'lh':
        hemi = 0
    if label.hemi == 'rh':
        hemi = 1
    center_vertex = label.center_of_mass(subject='fsaverage', 
                                        restrict_vertices=False, 
                                        subjects_dir=None)
    mni_pos = mne.vertex_to_mni(center_vertex, hemis=hemi,
                            subject='fsaverage', subjects_dir=None)
    node_coords.append(mni_pos)
node_coords = np.array(node_coords)
ticks = [lb.name for lb in labels]

matrix = np.zeros(shape=(len(labels), len(labels)))
for df_idx, df in enumerate(dfs[:2]):
    idx1 = ticks.index(df["label_1"].unique()[0])
    idx2 = ticks.index(df["label_2"].unique()[0])
    matrix[idx1][idx2] = 1 

graph = matrix + matrix.T
custom_cmap = sns.light_palette("seagreen", as_cmap=True)
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
plot_connectome(adjacency_matrix=graph, node_coords=node_coords, display_mode="lzry", edge_cmap=custom_cmap,
                node_color='k', node_size=10, axes=ax, colorbar=False,
                edge_threshold="90%")

matrix = np.zeros(shape=(len(labels), len(labels)))
for df_idx, df in enumerate(dfs[2:4]):
    idx1 = ticks.index(df["label_1"].unique()[0])
    idx2 = ticks.index(df["label_2"].unique()[0])
    matrix[idx1][idx2] = 1 

graph = matrix + matrix.T
custom_cmap = sns.light_palette("#79C", as_cmap=True)
plot_connectome(adjacency_matrix=graph, node_coords=node_coords, display_mode="lzry", edge_cmap=custom_cmap,
                node_color='k', node_size=10, axes=ax, colorbar=False,
                edge_threshold="90%")

matrix = np.zeros(shape=(len(labels), len(labels)))
for df_idx, df in enumerate(dfs[4:]):
    idx1 = ticks.index(df["label_1"].unique()[0])
    idx2 = ticks.index(df["label_2"].unique()[0])
    matrix[idx1][idx2] = 1 

graph = matrix + matrix.T
custom_cmap = sns.light_palette("xkcd:copper", 6, as_cmap=True)
plot_connectome(adjacency_matrix=graph, node_coords=node_coords, display_mode="lzry", edge_cmap=custom_cmap,
                node_color='k', node_size=10, axes=ax, colorbar=False,
                edge_threshold="90%")
fig.tight_layout()
fig.savefig(Path.cwd().parent / "figures" / "3.pdf")
