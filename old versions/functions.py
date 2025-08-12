import numpy as np
import pandas as pd
import h5py
import tkinter as tk
from tkinter import filedialog, simpledialog
from PyMca5.PyMcaIO.OutputBuffer import OutputBuffer
from PyMca5.PyMcaPhysics.xrf.FastXRFLinearFit import FastXRFLinearFit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from libpysal.weights import lat2W
from esda.moran import Moran

def elemental_conversion():
    # === File picker ===
    root = tk.Tk()
    root.withdraw()

    hdf_file = filedialog.askopenfilename(title="Select HDF file", filetypes=[("HDF5 files", "*.hdf *.h5")])
    cfg_file = filedialog.askopenfilename(title="Select .cfg file", filetypes=[("CFG files", "*.cfg")])
    dataset_path = simpledialog.askstring("Dataset path", "Enter the dataset path inside the HDF (e.g., /map)")

    if not hdf_file or not cfg_file or not dataset_path:
        print("Cancelled or invalid input.")
        return None

    # === Load HDF data ===
    with h5py.File(hdf_file, 'r') as f:
        data = np.array(f[dataset_path])

    # === Prepare OutputBuffer ===
    output = OutputBuffer(outputDir="output", csv=True, overwrite=True)
    output.saveDataDiagnostics = False
    output.saveFOM = False
    output.saveResiduals = False
    output.saveFit = False
    output.saveImages = False
    output.saveData = True

    # === Run Fit ===
    fast_fit = FastXRFLinearFit()
    fast_fit.setFitConfigurationFile(cfg_file)
    fast_fit.fitMultipleSpectra(y=data, weight=0, outbuffer=output)

    # === Load and clean results ===
    df_elemental = pd.read_csv("output/IMAGES/images.csv",sep=";")  # replace with your filename
    df_elemental = df_elemental.loc[:, ~df_elemental.columns.str.contains("1")]

    # Rename the first two columns
    df_elemental.rename(columns={df_elemental.columns[0]: "Y", df_elemental.columns[1]: "X"}, inplace=True)

    # Reorder to have "X" first, then "Y", then the rest
    cols = df_elemental.columns.tolist()
    new_order = ["X", "Y"] + cols[2:]
    df_elemental = df_elemental[new_order]
    return df_elemental

# df_elemental = elemental_conversion()

# print(df_elemental.head())

def popup_input(title, message, display_lines=None, validate_fn=None):
    """
    Shows a popup with an optional scrollable text display and an entry box.
    
    Parameters:
        title (str): Window title.
        message (str): Label above entry.
        display_lines (list[str]): Optional list of strings to show in a text area.
        validate_fn (function): Optional function to validate input. Must raise ValueError on failure.
        
    Returns:
        The validated input (as str, or as cast by validate_fn).
    """
    result = None

    def on_submit():
        nonlocal result
        val = entry.get()
        try:
            if validate_fn:
                val = validate_fn(val)
            result = val
            root.quit()
            root.destroy()
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, "Invalid input")

    root = tk.Tk()
    root.title(title)

    if display_lines:
        tk.Label(root, text="Info:").pack(pady=(5, 0))
        text = tk.Text(root, height=min(len(display_lines), 15), width=150)
        text.configure(font=("Courier New", 10))
        text.pack()
        for line in display_lines:
            text.insert(tk.END, line + "\n")
        text.config(state=tk.DISABLED)

    tk.Label(root, text=message).pack(pady=(10, 0))
    entry = tk.Entry(root)
    entry.pack(pady=5)
    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)

    root.mainloop()

    if result is None:
        raise ValueError("No input provided")
    return result

def moran_filter(df_elemental, threshold=None):
    element_columns = df_elemental.drop(columns=["X", "Y"]).columns
    width = df_elemental["X"].max() + 1
    height = df_elemental["Y"].max() + 1
    w = lat2W(height, width)
    moran_scores = {}
    for element in element_columns:
        img = np.zeros((height, width))
        for _, row in df_elemental.iterrows():
            x, y = int(row["X"]), int(row["Y"])
            img[y, x] = row[element]
        moran = Moran(img.flatten(), w)
        moran_scores[element] = moran.I

    moran_df = pd.DataFrame.from_dict(moran_scores, orient='index', columns=["Morans_I"])
    stats_sorted = moran_df.sort_values("Morans_I", ascending=False)
    if threshold is None:
        display_lines = [f"{idx}: {row['Morans_I']:.4f}" for idx, row in stats_sorted.iterrows()]
        threshold = popup_input(
            title="Select Moran's I Threshold",
            message="Enter threshold value:",
            display_lines=display_lines,
            validate_fn=lambda x: float(x)
        )

    return stats_sorted[stats_sorted["Morans_I"] > threshold]

# important_elements= moran_filter(df_elemental)
# print(important_elements)
# print(df_elemental.head())

def select_element_stats(defaults=["mean", "std"]):
    root = tk.Tk()
    root.title("Select Statistical Features")
    options = ["mean", "std", "min", "max", "median"]
    selected = []

    check_vars = {}

    def on_submit():
        nonlocal selected
        selected = [opt for opt in options if check_vars[opt].get()]
        root.quit()  # Stop mainloop
        root.destroy()  # Close window

    tk.Label(root, text="Select stats to compute:").pack(pady=5)

    for opt in options:
        var = tk.BooleanVar(value=opt in defaults)
        check_vars[opt] = var
        tk.Checkbutton(root, text=opt, variable=var).pack(anchor='w')

    submit_btn = tk.Button(root, text="Submit", command=on_submit)
    submit_btn.pack(pady=10)

    root.mainloop()
    return selected


def compute_and_cluster_stats(X, moran_df, stats_to_compute=None, pick_cluster=None):
    if stats_to_compute is None:
        stats_to_compute = select_element_stats()

    stats = {}
    for stat in stats_to_compute:
        if stat == "mean":
            stats["mean"] = X.mean()
        elif stat == "std":
            stats["std"] = X.std()
        elif stat == "min":
            stats["min"] = X.min()
        elif stat == "max":
            stats["max"] = X.max()
        elif stat == "median":
            stats["median"] = X.median()

    stats["Moran's I"] = moran_df["Morans_I"]
    df_stats = pd.DataFrame(stats)

    # Cluster
    clustering = KMeans(n_clusters=2, random_state=0)
    labels = clustering.fit_predict(df_stats)
    df_stats["cluster_foreground"] = labels

    if pick_cluster is None:
        # Columns for display (excluding cluster label)
        stat_columns = [col for col in df_stats.columns if col != "cluster_foreground"]
        label_width = 10

        # --- Dynamically calculate column widths based on header/number content ---
        col_widths = {}
        for col in stat_columns:
            col_label_len = len(col)
            max_val_len = df_stats[col].map(lambda x: f"{x:.3f}").map(len).max()
            col_widths[col] = max(col_label_len + 2, max_val_len + 2)  # add padding

        def format_header():
            parts = [f"{'Element':<{label_width}}"]
            for col in stat_columns:
                parts.append(f"{col:^{col_widths[col]}}")  # centered label
            return "".join(parts)

        def format_row(label, row):
            parts = [f"{label:<{label_width}}"]
            for col in stat_columns:
                val = row[col]
                parts.append(f"{val:>{col_widths[col]}.3f}")  # right-aligned value
            return "".join(parts)

        # Prepare clusters
        cluster_0 = df_stats[df_stats["cluster_foreground"] == 0].drop(columns="cluster_foreground").round(3).sort_index()
        cluster_1 = df_stats[df_stats["cluster_foreground"] == 1].drop(columns="cluster_foreground").round(3).sort_index()

        # Build display
        header_left = format_header()
        header_right = format_header()
        row_width = len(header_left)

        display_lines = []
        display_lines.append(f"{'Cluster 0':<{row_width}} | {'Cluster 1'}")
        display_lines.append(f"{header_left} | {header_right}")

        # Build body
        c0_items = list(cluster_0.iterrows())
        c1_items = list(cluster_1.iterrows())
        max_len = max(len(c0_items), len(c1_items))

        for i in range(max_len):
            left = format_row(*c0_items[i]) if i < len(c0_items) else " " * row_width
            right = format_row(*c1_items[i]) if i < len(c1_items) else ""
            display_lines.append(f"{left} | {right}")


        chosen = popup_input(
            title="Select Cluster",
            message="Enter cluster number to keep (0 or 1):",
            display_lines=display_lines,
            validate_fn=lambda x: int(x) if int(x) in [0, 1] else (_ for _ in ()).throw(ValueError())
        )
        pick_cluster = int(chosen)

    selected_elements = df_stats[df_stats["cluster_foreground"] == pick_cluster]
    return selected_elements
# element_stats=compute_and_cluster_stats(df_elemental[important_elements.index], important_elements)

# print(element_stats)
