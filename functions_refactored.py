# functions_pyside.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from PyMca5.PyMcaIO.OutputBuffer import OutputBuffer
from PyMca5.PyMcaPhysics.xrf.FastXRFLinearFit import FastXRFLinearFit
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, 
    QDialogButtonBox, QFileDialog, QInputDialog, QMessageBox,
    QCheckBox, QScrollArea, QWidget, QGridLayout)
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from libpysal.weights import lat2W
from esda.moran import Moran
from sklearn.cluster import KMeans

def elemental_conversion_qt(parent=None):
    try:
        # File selection
        hdf_file, _ = QFileDialog.getOpenFileName(parent, "Select HDF file", "", "HDF5 Files (*.h5 *.hdf)")
        if not hdf_file:
            return None

        cfg_file, _ = QFileDialog.getOpenFileName(parent, "Select .cfg file", "", "CFG Files (*.cfg)")
        if not cfg_file:
            return None

        dataset_path, ok = QInputDialog.getText(parent, "Dataset Path", "Enter the dataset path inside the HDF (e.g., /map):")
        if not ok or not dataset_path.strip():
            return None

        # Create temp directory
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="pymca_output_")
        
        # Configure output
        output = OutputBuffer(outputDir=temp_dir, csv=True, overwrite=True)
        output.saveDataDiagnostics = False
        output.saveFOM = False
        output.saveResiduals = False
        output.saveFit = False
        output.saveImages = False
        output.saveData = True

        # Read input data
        with h5py.File(hdf_file, 'r') as f:
            data = np.array(f[dataset_path])

        # Run fit
        fast_fit = FastXRFLinearFit()
        fast_fit.setFitConfigurationFile(cfg_file)
        fast_fit.fitMultipleSpectra(y=data, weight=0, outbuffer=output)

        # Process results
        result_path = os.path.join(temp_dir, "IMAGES", "images.csv")
        df_elemental = pd.read_csv(result_path, sep=";")
        df_elemental = df_elemental.loc[:, ~df_elemental.columns.str.contains("1")]
        df_elemental.rename(columns={df_elemental.columns[0]: "Y", df_elemental.columns[1]: "X"}, inplace=True)
        cols = df_elemental.columns.tolist()
        new_order = ["X", "Y"] + cols[2:]
        df_elemental = df_elemental[new_order]

        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return df_elemental

    except Exception as e:
        QMessageBox.critical(parent, "Conversion Error", f"Error during conversion: {str(e)}")
        return None
    

class PopupInputDialog(QDialog):
    def __init__(self, title, message, display_lines=None, validate_fn=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.result = None
        self.validate_fn = validate_fn

        layout = QVBoxLayout()

        if display_lines:
            info_label = QLabel("Info:")
            layout.addWidget(info_label)

            text_display = QTextEdit()
            text_display.setReadOnly(True)
            text_display.setPlainText("\n".join(display_lines))
            layout.addWidget(text_display)

        message_label = QLabel(message)
        layout.addWidget(message_label)

        self.input_field = QLineEdit()
        layout.addWidget(self.input_field)

        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.on_submit)
        layout.addWidget(submit_button)

        self.setLayout(layout)
        self.setMinimumWidth(500)

    def on_submit(self):
        val = self.input_field.text()
        try:
            if self.validate_fn:
                val = self.validate_fn(val)
            self.result = val
            self.accept()
        except ValueError:
            self.input_field.setText("")
            self.input_field.setPlaceholderText("Invalid input. Try again.")

class DataPreviewDialog(QDialog):
    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout()

        label = QLabel("Preview:")
        layout.addWidget(label)

        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setPlainText(content)
        layout.addWidget(text_area)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

def image_from_array(img, cmap='inferno'):
    fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=100)
    ax.imshow(img, cmap=cmap)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)

    buf.seek(0)
    image = QImage.fromData(buf.getvalue())
    return QPixmap.fromImage(image)

def generate_image(df, element, height, width):
    image = np.zeros((height, width))
    x_vals = df["X"].astype(int).values
    y_vals = df["Y"].astype(int).values
    intensities = df[element].values
    image[y_vals, x_vals] = intensities
    return image

class ElementSelectionDialog(QDialog):
    def __init__(self, all_elements, selected_elements, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Elements to Keep")
        self.setMinimumSize(400, 500)
        self.selected = []

        layout = QVBoxLayout(self)

        label = QLabel("Check the elements you want to include:")
        layout.addWidget(label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)

        self.checkboxes = {}
        for elem in all_elements:
            cb = QCheckBox(elem)
            cb.setChecked(elem in selected_elements)
            self.checkboxes[elem] = cb
            inner_layout.addWidget(cb)

        scroll.setWidget(inner)
        layout.addWidget(scroll)

        btn = QPushButton("Apply")
        btn.clicked.connect(self.on_submit)
        layout.addWidget(btn)

    def on_submit(self):
        self.selected = [k for k, cb in self.checkboxes.items() if cb.isChecked()]
        self.accept()


class ElementImageSelectionDialog(QDialog):
    def __init__(self, element_list, checked_elements, df, width, height, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Elements")
        self.setMinimumSize(800, 600)

        self.selected_elements = []
        layout = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll_widget = QWidget()
        grid_layout = QGridLayout(scroll_widget)

        # Access cache via parent
        cache = getattr(parent, "pipeline_config", {}).setdefault("_element_images", {})

        for i, el in enumerate(element_list):
            row, col = divmod(i, 5)

            if el in cache:
                image = cache[el]
            else:
                image = np.zeros((height, width))
                for _, row_data in df.iterrows():
                    x, y = int(row_data["X"]), int(row_data["Y"])
                    image[y, x] = row_data[el]
                cache[el] = image  # Save for later

            pixmap = image_from_array(image)

            label = QLabel(el)
            label.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio))
            label.setAlignment(Qt.AlignCenter)

            checkbox = QCheckBox(el)
            checkbox.setChecked(el in checked_elements)

            grid_layout.addWidget(label, row * 2, col)
            grid_layout.addWidget(checkbox, row * 2 + 1, col)

            self.selected_elements.append((el, checkbox))

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # OK / Cancel button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_selected_elements(self):
        return [el for el, cb in self.selected_elements if cb.isChecked()]


def scale_features(df, scaler_type="standard"):
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X = df.drop(columns=["X", "Y"])
    scaled = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(scaled, columns=X.columns)
    scaled_df["X"] = df["X"].values
    scaled_df["Y"] = df["Y"].values
    return scaled_df


def moran_filter(df_features, coordinates_df, threshold=None, parent=None):
    """
    Apply Moran's I to filter elements based on spatial autocorrelation.

    Parameters:
        df_features (DataFrame): Element-only dataframe (no X/Y).
        coordinates_df (DataFrame): DataFrame with columns "X" and "Y".
        threshold (float or None): Moran's I threshold; if None, prompt user via Qt.
        parent: Optional Qt parent for modal dialogs.

    Returns:
        DataFrame: Filtered Moran's I scores (index = kept element names).
    """
    if "X" not in coordinates_df.columns or "Y" not in coordinates_df.columns:
        raise ValueError("Coordinates must include 'X' and 'Y' columns")

    width = coordinates_df["X"].max() + 1
    height = coordinates_df["Y"].max() + 1
    w = lat2W(height, width)

    moran_scores = {}
    for element in df_features.columns:
        img = np.zeros((height, width))
        for i, row in df_features.iterrows():
            x = int(coordinates_df.loc[i, "X"])
            y = int(coordinates_df.loc[i, "Y"])
            img[y, x] = row[element]
        moran = Moran(img.flatten(), w)
        moran_scores[element] = moran.I

    moran_df = pd.DataFrame.from_dict(
        moran_scores, orient='index', columns=["Morans_I"]
    ).sort_values("Morans_I", ascending=False)

    # Prompt user via Qt popup if threshold not given
    if threshold is None and parent is not None:
        from PySide6.QtWidgets import QInputDialog
        threshold_str, ok = QInputDialog.getText(
            parent,
            "Select Moran's I Threshold",
            "Enter threshold value:\n\n" +
            "\n".join([f"{idx}: {row['Morans_I']:.4f}" for idx, row in moran_df.iterrows()])
        )
        if not ok:
            return pd.DataFrame()  # Cancelled
        try:
            threshold = float(threshold_str)
        except ValueError:
            raise ValueError("Invalid threshold input.")

    return moran_df[moran_df["Morans_I"] > float(threshold)]

def foreground_clustering(df, k=3, use_mean=True, use_std=False, return_clustered_data=False):
        """
        Cluster element columns using statistics (mean, std).
        Returns list of elements in each cluster.

        If return_clustered_data=True, also returns the full cluster dataframe.
        """
        stats = []
        for col in df.columns:
            values = []
            if use_mean:
                values.append(df[col].mean())
            if use_std:
                values.append(df[col].std())
            stats.append(values)

        if not stats:
            return [], None if return_clustered_data else []

        X = np.array(stats)
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)

        cluster_map = {col: label for col, label in zip(df.columns, labels)}
        cluster_df = pd.DataFrame([
            {"Element": col, "Cluster": cluster_map[col], "Mean": df[col].mean(), "Std": df[col].std()}
            for col in df.columns
        ])

        if return_clustered_data:
            return labels, cluster_df
        else:
            return cluster_df

def scale_dataset(df, scaler_type="standard"):
    """
    Scales all columns in df (including X/Y if present).

    Parameters:
        df (pd.DataFrame): Dataset to scale
        scaler_type (str): 'standard' or 'minmax'

    Returns:
        pd.DataFrame: Scaled dataset with original column order preserved
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled_values = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_values, columns=df.columns)

    return scaled_df

def create_augmented_dataset(df, original_coords, element_scaler="standard", xy_scaler="standard", weight_elements=1.0, weight_xy=1.0):
    # Ensure XY is added
    if "X" not in df.columns or "Y" not in df.columns:
        df = df.copy()
        df.insert(0, "X", original_coords["X"].values)
        df.insert(1, "Y", original_coords["Y"].values)

    # Separate components
    XY = df[["X", "Y"]]
    elements = df.drop(columns=["X", "Y"])

    # Scale
    scaler_e = StandardScaler() if element_scaler == "standard" else MinMaxScaler()
    scaler_xy = StandardScaler() if xy_scaler == "standard" else MinMaxScaler()

    scaled_elements = scaler_e.fit_transform(elements) * weight_elements
    scaled_xy = scaler_xy.fit_transform(XY) * weight_xy

    # Combine
    df_aug = pd.DataFrame(
        np.hstack([scaled_xy, scaled_elements]),
        columns=["X", "Y"] + list(elements.columns)
    )
    return df_aug


