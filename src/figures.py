from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from nilearn.plotting import plot_connectome

###### fig 1_A ######
fname = Path.cwd().parent / "data" / "ranks.csv"
df = pd.read_csv(fname)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
color_1 = sns.cubehelix_palette(10, rot=-.25, light=.7)[2]
color_2 = sns.cubehelix_palette(10, rot=.4, light=.9)[4]
g = sns.FacetGrid(df, col="freq_band", row="group", hue="method", height=2.5, palette=[color_2, color_1], aspect=1.7, legend_out=True)
kwargs = {"element": "step", "discrete": True, "multiple": "dodge", "fill": True}
g.map(sns.histplot, "rank", **kwargs)
legend_kwargs = {"bbox_to_anchor": (0.5, 0.6, 0.5, 0.5)}
g.add_legend(**legend_kwargs)
g.tight_layout()
g.savefig(Path.cwd().parent / "figures" / "ranks.pdf")

###### fig 1_B ######
fname = Path.cwd().parent / "data" / "graph_comparison_2.csv"
df = pd.read_csv(fname)
vals = [0.25, 0.5, 1, 1.5]
df = df.query("alpha in @vals and beta in @vals")
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(df, col="frequency_range", row="alpha", hue="beta", height=2.5, aspect=1.3, palette="YlGnBu_d", legend_out=True)
kwargs = {"markersize": 2.5, "lw": 2}
g.map_dataframe(sns.pointplot, x="times_used", y="PDiv", **kwargs)
g.set(xticklabels=[], title="", ylabel="", xlabel="")
legend_kwargs = {"bbox_to_anchor": (0.5, 0.7, 0.5, 0.5)}
g.add_legend(**legend_kwargs)
g.tight_layout()
g.savefig(Path.cwd().parent / "figures" / "PDiv.pdf")

###### fig 1_C ######
fname = Path.cwd().parent / "data" / "graph_comparison_2.csv"
df = pd.read_csv(fname)
vals = [0.25, 0.5, 1, 1.5]
df = df.query("alpha in @vals and beta in @vals")
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(df, col="frequency_range", row="alpha", hue="beta", height=2.5, aspect=1.3, palette="YlGnBu_d", legend_out=True)
kwargs = {"markersize": 2.5, "lw": 2}
g.map_dataframe(sns.pointplot, x="times_used", y="frobenius", **kwargs)
g.set(xticklabels=[], title="", ylabel="", xlabel="")
legend_kwargs = {"bbox_to_anchor": (0.5, 0.7, 0.5, 0.5)}
g.add_legend(**legend_kwargs)
g.tight_layout()
g.savefig(Path.cwd().parent / "figures" / "Frobenius.pdf")

###### fig 1_D ######
fname = Path.cwd().parent / "data" / "df_diff.csv"
df = pd.read_csv(fname)
vals = [1, 1.5]
df = df.query("alpha in @vals and beta in @vals")
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
cl1, cl2 = sns.color_palette("YlGnBu_d")[3], sns.color_palette("YlGnBu_d")[5] 
g = sns.FacetGrid(df, col="frequency_range", row="alpha", hue="beta", height=2.5, aspect=1.7, palette=[cl1, cl2], legend_out=True)
kwargs = {"markersize": 2.5, "lw": 2}
g.map_dataframe(sns.pointplot, x="times_used", y="PDiv", **kwargs)
g.set(xticklabels=[], title="", ylabel="", xlabel="")
legend_kwargs = {"bbox_to_anchor": (0.5, 0.6, 0.5, 0.5)}
g.add_legend(**legend_kwargs)
def add_hline(data, **kwargs):
    ax = plt.gca()
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
g.map_dataframe(add_hline)
g.tight_layout()
g.savefig(Path.cwd().parent / "figures" / "PDiv_dist_hline.pdf")

###### fig 2 ######
fname = Path.cwd().parent / "data" / "connections.csv"
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


for serie, title in zip([range(2), range(2, 4), range(4, 7)], ["alpha", "beta", "gamma"]):
    if title == "gamma":
        fig, axs = plt.subplots(1, 3, figsize=(5, 3))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(5, 3))
    
    for ax_idx, (idx, ax) in enumerate(zip(serie, axs)):
        sns.boxplot(data=dfs[idx], x="group", y="connection_strength", hue="group",
                        hue_order=hue_order, palette=palette_color, fill=False, width=0.6,
                        gap=0, linewidth=1.8, ax=ax)
        sns.stripplot(data=dfs[idx], x="group", y="connection_strength", hue="group", 
                        hue_order=hue_order, palette=palette_color, linewidth=0, size=3.5,
                        edgecolor=None, ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("")
        ax.set_ylim([None, 0.04])
        if ax_idx in [1, 2]:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])

    # fig.savefig(Path.cwd().parent / "figures" / f"{title}.pdf")

# optional
dfs = [df_alpha_1, df_alpha_2, df_beta_1, df_beta_2, df_gamma_1, df_gamma_2, df_gamma_3]
def plot_glass_brains(dfs, title):
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
    for df in dfs:
        idx1 = ticks.index(df["label_1"].unique()[0])
        idx2 = ticks.index(df["label_2"].unique()[0])
        matrix[idx1][idx2] = 1 

    graph = matrix + matrix.T
    # custom_cmap = sns.dark_palette("#b285bc", reverse=True, as_cmap=True)
    custom_cmap = sns.cubehelix_palette(10, rot=-.45, light=.35, reverse=True, as_cmap=True)
    fig, ax = plt.subplots(1, 1, figsize=(11, 3))
    edge_kwargs = {"lw": 3}
    plot_connectome(adjacency_matrix=graph, node_coords=node_coords, display_mode="lzry", edge_cmap=custom_cmap,
                    node_color='k', node_size=10, axes=ax, colorbar=False,
                    edge_threshold="90%", edge_kwargs=edge_kwargs)
    fig.tight_layout()
    fig.savefig(Path.cwd().parent / "figures" / f"{title}_glass_brain.pdf")

for sub_dfs, title in zip([dfs[:2], dfs[2:4], dfs[4:]], ["alpha", "beta", "gamma"]):
    plot_glass_brains(sub_dfs, title)

###### supplementary figure A ######
fname = Path.cwd().parent / "data" / "graph_comparison.csv"
df = pd.read_csv(fname)
vals = [0.25, 0.5, 1, 1.5]
df = df.query("alpha in @vals and beta in @vals")
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(df, col="alpha", row="frequency_range", hue="beta", height=2.5, aspect=1.3, palette="YlGnBu_d", legend_out=True)
kwargs = {"markersize": 2.5, "lw": 2}
g.map_dataframe(sns.pointplot, x="times_used", y="euc_dist", **kwargs)
g.set(xticklabels=[], title="", ylabel="", xlabel="")
legend_kwargs = {"bbox_to_anchor": (0.5, 0.7, 0.5, 0.5)}
g.add_legend(**legend_kwargs)
g.tight_layout()
g.savefig(Path.cwd().parent / "figures" / "Euclidean.pdf")

###### supplementary figure C ######
fname = Path.cwd().parent / "data" / "df_diff.csv"
df = pd.read_csv(fname)
vals = [1, 1.5]
df = df.query("alpha in @vals and beta in @vals")
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
cl1, cl2 = sns.color_palette("YlGnBu_d")[3], sns.color_palette("YlGnBu_d")[5] 
g = sns.FacetGrid(df, col="alpha", row="frequency_range", hue="beta", height=2.5, aspect=1.3, palette=[cl1, cl2], legend_out=True)
kwargs = {"markersize": 2.5, "lw": 2}
g.map_dataframe(sns.pointplot, x="times_used", y="euc_dist", **kwargs)
g.set(xticklabels=[], title="", ylabel="", xlabel="")
legend_kwargs = {"bbox_to_anchor": (0.5, 0.6, 0.5, 0.5)}
g.add_legend(**legend_kwargs)
def add_hline(data, **kwargs):
    ax = plt.gca()
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
g.map_dataframe(add_hline)
g.tight_layout()
g.savefig(Path.cwd().parent / "figures" / "Euclidean_diff.pdf")

###### supplementary figure B ######
label_names = ["parahippocampal-lh", "superiorfrontal-lh", "superiorfrontal-rh",
                "bankssts-lh", "rostralmiddlefrontal-lh", "frontalpole-lh",
                "lateraloccipital-lh", "parsorbitalis-lh", "frontalpole-rh"] 

brain_labels = np.array(mne.read_labels_from_annot(subject="fsaverage", parc="aparc", verbose=False))
lb_names = [lb.name for lb in brain_labels]
idxs = np.array([lb_names.index(lb) for lb in label_names])
lbs = list(brain_labels[idxs])
cl1 = sns.cubehelix_palette(10, rot=2.5, light=.7, reverse=True)[9]
cl2 = sns.cubehelix_palette(10, rot=-2*np.pi/10, light=.7, reverse=True)[1]
cl3 = sns.cubehelix_palette(10, rot=4.5, light=.7, reverse=True)[3]
cl4 = list(sns.color_palette("Reds")[-1])# sns.cubehelix_palette(10, rot=.5, light=.7, reverse=True)[2]
cl5 = sns.cubehelix_palette(10, rot=4.5, light=.7, reverse=True)[2]
cl6 = sns.cubehelix_palette(10, rot=8.5, light=.7, reverse=True)[8]
cl7 = sns.cubehelix_palette(10, rot=8.5, light=.7, reverse=True)[0]
cl8 = sns.cubehelix_palette(10, rot=1.5, light=.7, reverse=True)[2]
cl9 = sns.cubehelix_palette(10, rot=4.5, light=.7, reverse=True)[5]
colors = [cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9]

brain_kwargs = dict(background="white", surf="pial_semi_inflated", cortex=["#b8b4ac", "#b8b4ac"])
brain = mne.viz.Brain("fsaverage", subjects_dir=None, hemi="lh", views="lateral", **brain_kwargs)
for lb, lb_name, color in zip(lbs, label_names, colors):
    if lb.hemi == "lh":
        brain.add_label(lb, hemi="lh", color=color, borders=False, alpha=0.7)
        # sub_lbs = mne.read_labels_from_annot(subject="fsaverage", parc="aparc_sub", regexp=lb_name[:-3], hemi="lh", verbose=False)
        # for sub_lb in sub_lbs:
        #     brain.add_label(sub_lb, hemi="lh", color=color, borders=True, alpha=0.7)

brain_scr_1 = brain.screenshot()
brain = mne.viz.Brain("fsaverage", subjects_dir=None, hemi="rh", views="lateral", **brain_kwargs)
for lb, lb_name, color in zip(lbs, label_names, colors):
    if lb.hemi == "rh":
        brain_scr_2 = brain.add_label(lb, hemi="rh", color=color, borders=False, alpha=0.7)
        # sub_lbs = mne.read_labels_from_annot(subject="fsaverage", parc="aparc_sub", regexp=lb_name[:-3], hemi="rh", verbose=False)
        # for sub_lb in sub_lbs:
        #     brain.add_label(sub_lb, hemi="rh", color=color, borders=True, alpha=0.7)

brain_scr_2 = brain.screenshot()
brain = mne.viz.Brain("fsaverage", subjects_dir=None, hemi="lh", views="medial", **brain_kwargs)
for lb, lb_name, color in zip(lbs, label_names, colors):
    if lb.hemi == "lh":
        brain_scr_3 = brain.add_label(lb, hemi="lh", color=color, borders=False, alpha=0.7)
        # sub_lbs = mne.read_labels_from_annot(subject="fsaverage", parc="aparc_sub", regexp=lb_name[:-3], hemi="lh", verbose=False)
        # for sub_lb in sub_lbs:
        #     brain.add_label(sub_lb, hemi="lh", color=color, borders=True, alpha=0.7)

brain_scr_3 = brain.screenshot()
brain = mne.viz.Brain("fsaverage", subjects_dir=None, hemi="rh", views="medial", **brain_kwargs)
for lb, lb_name, color in zip(lbs, label_names, colors):
    if lb.hemi == "rh":
        brain_scr_4 = brain.add_label(lb, hemi="rh", color=color, borders=False, alpha=0.7)
        # sub_lbs = mne.read_labels_from_annot(subject="fsaverage", parc="aparc_sub", regexp=lb_name[:-3], hemi="rh", verbose=False)
        # for sub_lb in sub_lbs:
        #     brain.add_label(sub_lb, hemi="rh", color=color, borders=True, alpha=0.7)
brain_scr_4 = brain.screenshot()

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
fig.subplots_adjust(hspace=0.1)
for ax, brain in zip([axes[0][0], axes[0][1], axes[1][0], axes[1][1]], [brain_scr_1, brain_scr_2, brain_scr_3, brain_scr_4]):
    nonwhite_pix = (brain != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = brain[nonwhite_row][:, nonwhite_col]
    ax.imshow(cropped_screenshot)
    ax.axis("off")
fig.savefig(Path.cwd().parent / "figures" / "brains_aparc.pdf")

