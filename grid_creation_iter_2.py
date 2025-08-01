import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import numpy as np
from libpysal.weights import lat2W
from esda.moran import Moran
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import NMF
import shutil
import h5py
from PyMca5.PyMcaPhysics.xrf.FastXRFLinearFit import FastXRFLinearFit
from PyMca5.PyMcaIO.OutputBuffer import OutputBuffer
import seaborn as sns
import time
from plot_help import *

start_time = time.time()

hdf_file = '../XRF_Karydas/XRF_data/ScanE.hdf'
dataset_path = '/map'
cfg_file = '../XRF_Karydas/XRF_data/Vergina_Demo.cfg'


with h5py.File(hdf_file, 'r') as f:
    # Explore the structure to find your dataset
    def visit(name, obj):
        print(name, obj)


    f.visititems(visit)

    # Example: load a 3D dataset
    data_ed = f['/map'][:]  # Replace with actual path

height, width, channels = data_ed.shape  # data = your 3D array

# Generate coordinate strings
yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
coord_strings = [f"{x};{y}" for x, y in zip(xx.ravel(), yy.ravel())]

# Reshape data to 2D: one row per pixel, one column per channel
flattened_spectra = data_ed.reshape(-1, channels)  # shape: (height*width, channels)

channel_names = [f"Channel_{i}" for i in range(channels)]

df_2d = pd.DataFrame(flattened_spectra, columns=channel_names)
df_2d.insert(0, "X;Y", coord_strings)  # insert coordinate column at front

# df_2d.to_csv("semi_proc_data.csv")


with h5py.File(hdf_file, 'r') as f:
    data = np.array(f[dataset_path])



output = OutputBuffer(outputDir="output", csv=True, overwrite=True)
output.saveDataDiagnostics = False
output.saveFOM = False
output.saveResiduals = False
output.saveFit = False
output.saveImages = False
output.saveData = True

fast_fit = FastXRFLinearFit()
fast_fit.setFitConfigurationFile(cfg_file)
fast_fit.fitMultipleSpectra(y=data, weight=0)


##############################################################################################
pymca_time = time.time()
##############################################################################################

synthetic_data = df_2d.values
# Διαχωρισμός θέσης και φασμάτων
pixel_indices = synthetic_data[:, 0]  # δεν το επεξεργαζόμαστε εδώ, απλώς το κρατάμε
spectral_data = synthetic_data[:, 1:]


df_elemental = pd.read_csv("output/IMAGES/images.csv",sep=";")  # replace with your filename
df_elemental = df_elemental.loc[:, ~df_elemental.columns.str.contains("1")]

# Rename the first two columns
df_elemental.rename(columns={df_elemental.columns[0]: "Y", df_elemental.columns[1]: "X"}, inplace=True)

# Reorder to have "X" first, then "Y", then the rest
cols = df_elemental.columns.tolist()
new_order = ["X", "Y"] + cols[2:]
df_elemental = df_elemental[new_order]

##############################################################################################
##############################################################################################
# Get your element names (excluding "X", "Y", etc.)
element_columns = df_elemental.drop(columns=["X", "Y"]).columns

# Get image dimensions
width = df_elemental["X"].max() + 1
height = df_elemental["Y"].max() + 1

# Initialize results
moran_scores = {}

# Create spatial weights matrix (based on 2D grid)
w = lat2W(height, width)

for element in element_columns:
    # Initialize empty image
    img = np.zeros((height, width))
    # Fill image with element values
    for _, row in df_elemental.iterrows():
        x, y = int(row["X"]), int(row["Y"])
        img[y, x] = row[element]  # note: (row = Y, col = X)

    # Flatten and compute Moran's I
    moran = Moran(img.flatten(), w)
    moran_scores[element] = moran.I

# Convert to DataFrame for inspection
moran_df = pd.DataFrame.from_dict(moran_scores, orient='index', columns=["Moran_I"])
# print(moran_df)
##############################################################################################
##############################################################################################
X = df_elemental.drop(columns=["X", "Y"])

# Υπολόγισε μέση τιμή, διασπορά, % μη μηδενικών
element_stats = pd.DataFrame({
    "mean": X.mean(),
    # "std": 1-np.exp(-X.std()),
    "std": X.std(),
    # "non_zero_ratio": (X > 0).sum() / len(X),
    "Moran's I": moran_df["Moran_I"],
})
# print(element_stats)
element_stats=element_stats[element_stats["Moran's I"]>0.4]
# print(element_stats)
##############################################################################################
##############################################################################################

clustering = KMeans(n_clusters=2, random_state=0)
labels = clustering.fit_predict(element_stats)
# print(labels)

element_stats["cluster_foreground"] = labels
# print("[1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5")

# print(element_stats[element_stats["cluster"]==0].index.tolist())

##############################################################################################
##############################################################################################
important_elements =element_stats[element_stats["cluster_foreground"]==0].index.tolist()

X_data = X[important_elements].copy()
X_scaled = MinMaxScaler().fit_transform(X_data)

X_data_xy = np.concatenate((df_elemental[["X","Y"]].values, X_data), axis=1)
X_scaled_xy = MinMaxScaler().fit_transform(X_data_xy)
##############################################################################################
data_processing_time = time.time()
##############################################################################################
#Variables
BASE_NAME = "X_imp_elements"
BASE = X_data
METHOD = "Kmeans"
N_CLUSTERS=3
TSNE_PERPLEXITY = 10

x_str = df_elemental["X"].iloc[-1]
y_str = df_elemental["Y"].iloc[-1]
width = int(x_str) + 1
height = int(y_str) + 1
# Create empty image
cluster_map = np.zeros((height, width), dtype=int)


nmf = NMF(n_components=2, init='random',max_iter=500, random_state=0)
W = nmf.fit_transform(BASE)  # pixels × components
H = nmf.components_              # components × features

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(BASE)  # Apply PCA to foreground elements

tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, learning_rate='auto', init='pca', random_state=42)
X_tsne = tsne.fit_transform(BASE)

##############################################################################################
##############################################################################################

column_names = ["IMSHOW", "NMF", "PCA", "T-SNE"]
row_titles = [
    "No clustering",
    f"Clustering based \non {BASE_NAME}",
    "Clustering \nbased on NMF",
    "Clustering \nbased on PCA",
    "Clustering \nbased on TSNE"
]

# N_CLUSTERS = 3

kmeans_base   = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(BASE)
kmeans_W      = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(W)
kmeans_PCA    = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(X_pca)
kmeans_TSNE   = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(X_tsne)


cluster_sets = [
    None,
    kmeans_base.labels_,
    kmeans_W.labels_,
    kmeans_PCA.labels_,
    kmeans_TSNE.labels_,
]


# === Create cluster maps ===
width = int(df_elemental["X"].max()) + 1
height = int(df_elemental["Y"].max()) + 1
cluster_maps = [None] + [
    make_cluster_map(labels, df_elemental, width, height)
    for labels in cluster_sets[1:]
]

# === Optional: base image (original XRF image or RGB)
original_image = data_ed.sum(axis=2)  # Simple example: sum across channels

rows, cols = len(row_titles), len(column_names)
fig, axes = plt.subplots(rows, cols, figsize=(12, 14))
plt.subplots_adjust(wspace=0.1, hspace=0.1)


for i in range(rows):
    for j in range(cols):
        ax = axes[i, j]
        if j == 0:
            # First column: image or cluster map
            if i == 0:
                ax.imshow(original_image, cmap="gray")
                ax.axis("off")
            else:
                plot_imshow(fig, ax, cluster_maps[i], title=None)
        elif j == 1:
            plot_projection(
                ax, W, clusters=cluster_sets[i] if cluster_sets[i] is not None else None,
                title="NMF projection", xlabel="W0", ylabel="W1"
            )
        elif j == 2:
            plot_projection(
                ax, X_pca, clusters=cluster_sets[i] if cluster_sets[i] is not None else None,
                title="PCA projection", xlabel="PC1", ylabel="PC2"
            )
        elif j == 3:
            plot_projection(
                ax, X_tsne, clusters=cluster_sets[i] if cluster_sets[i] is not None else None,
                title="t-SNE projection", xlabel="t-SNE 1", ylabel="t-SNE 2"
            )

        # Column headers
        if i == 0:
            ax.set_title(column_names[j], fontsize=14, fontweight='bold')

# Row labels (left side)
for i, title in enumerate(row_titles):
    fig.text(0.005, 1-(0.85*i+1)/rows, title, va='center', ha='left',
             fontsize=13,rotation="vertical", fontweight='bold')

# Final layout and save
fig.suptitle(f"Data: {BASE_NAME}, Method: {METHOD}, Clusters:{N_CLUSTERS}", fontsize=16, fontweight='bold', y=0.92)
plt.tight_layout(rect=[0.07, 0, 1, 0.93])
plt.savefig(f"grid_summary_{BASE_NAME}_{METHOD}_{N_CLUSTERS}clust.png", dpi=1000)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

df_test = df_elemental.copy()
df_test[f"cluster_{BASE_NAME}"] = kmeans_base.labels_
df_test["cluster_NMF"]    = kmeans_W.labels_
df_test["cluster_PCA"]    = kmeans_PCA.labels_
df_test["cluster_TSNE"]   = kmeans_TSNE.labels_
cluster_col_names = [
    f"cluster_{BASE_NAME}",
    "cluster_NMF",
    "cluster_PCA",
    "cluster_TSNE"
]
for i, ax in enumerate(axes):
    col = cluster_col_names[i]
    title = f"Mean Composition per Cluster ({col.replace('cluster_', '')})"
    plot_cluster_heatmap(ax, df_test, col, element_columns, title)

plt.tight_layout()
plt.savefig(f"cluster_means_grid_{BASE_NAME}_{METHOD}_{N_CLUSTERS}clust.png", dpi=300)

# Columns containing the XRF channels (exclude "X;Y" and "Cluster")
channel_cols = [col for col in df_2d.columns if col.startswith("Channel_")]

base_cmap = plt.get_cmap("viridis", N_CLUSTERS)
cluster_colors = [base_cmap(i) for i in range(N_CLUSTERS)]

# Save summed spectra per cluster
fig, axs = plt.subplots(N_CLUSTERS, 1, figsize=(10, 3 * N_CLUSTERS), sharex=True)
df_2d['Cluster'] = kmeans_base.labels_
for i in range(N_CLUSTERS):
    cluster_df = df_2d[df_2d["Cluster"] == i][channel_cols]
    sum_spectrum = cluster_df.sum()

    axs[i].semilogy(sum_spectrum.values, label=f"Cluster {i} (Sum)",
                    color = cluster_colors[i])
    axs[i].set_ylabel("Total Counts")
    axs[i].legend(loc="upper right")
    axs[i].grid(True)

axs[-1].set_xlabel("Channel Index")
fig.suptitle("Summed 4096-Channel Spectra per Cluster", y=0.92)
plt.tight_layout()
plt.savefig(f"Cluster_sum_4096channels_{BASE_NAME}_{METHOD}.png", dpi=300)

end_time = time.time()
print(f"Time for PyMCA: {(-start_time+pymca_time)/60}")
print(f"Time for processing: {(-pymca_time+data_processing_time)/60}")
print(f"Total time: {(end_time-start_time)/60}")



