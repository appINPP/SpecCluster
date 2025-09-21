import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QCheckBox, QFileDialog,
    QSplitter, QTextEdit, QGroupBox, QFormLayout,
    QScrollArea, QSpinBox, QDoubleSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QColorDialog,
    QGridLayout, QTreeWidget, QTreeWidgetItem, 
    QHeaderView, QDialogButtonBox, QTabWidget,
    QToolButton, QMenu, QStyle, 
    )
from PySide6.QtCore import Qt, QTimer, QCoreApplication
from PySide6.QtGui import (
    QPalette, QColor, QGuiApplication, 
    QImage, QPainter, QPixmap,QAction
    )
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, to_hex, LogNorm
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.ticker import LogLocator, NullFormatter
from functions_refactored import *
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
import umap
import numpy as np
from functools import partial
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches

import json, math
import h5py
from dataclasses import dataclass, asdict
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt

class ClusterSelectionDialog(QDialog):
    def __init__(self, clustered_df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Foreground Cluster")
        layout = QVBoxLayout(self)

        self.selected_cluster = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        grouped = clustered_df.groupby("Cluster")
        for cluster_id, group in grouped:
            label = QLabel(f"Cluster {cluster_id}")
            label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            content_layout.addWidget(label)
            for _, row in group.iterrows():
                line = QLabel(f"{row['Element']:10}  Mean: {row['Mean']:.2f}  Std: {row['Std']:.2f}")
                content_layout.addWidget(line)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        form = QHBoxLayout()
        form.addWidget(QLabel("Select Cluster ID:"))
        self.spin = QSpinBox()
        self.spin.setRange(0, clustered_df["Cluster"].max())
        form.addWidget(self.spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_cluster(self):
        return self.spin.value()

class ImageCanvas(FigureCanvas):
    def __init__(self, title="Canvas"):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        super().__init__(self.fig)

    def show_dummy(self, label="Nothing yet"):
        self.ax.clear()
        self.ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=12)
        self.draw()



class XRFGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRF Imaging GUI")
        self.setMinimumSize(1400, 800)
        self.dataset = None  # Placeholder for loaded/processed dataset
        self.pipeline_config = {}
        self.init_ui()
        self.element_thumbnails = {}  # in __init__ of your class
        
        # in __init__
        self._pick_cid = self.right_canvas.mpl_connect("pick_event", self.on_projection_pick)
        self.visible_clusters = set()  # filled after clustering

        self._spectra_dlg = None
        self._spectra_canvas = None
        self._spectra_axs = None
        self._spectra_use_mean = False
        self._spectra_current = None      # list of arrays currently plotted
        self._spectra_agg_combo = None    # the Sum/Mean dropdown in the dialog
        self._sil_cache = {}   # key: (id(df), n, d, tuple(kvals), pca_dim, sample_size) -> (kvals, scores)

    def load_dataset_qt(self):
        df = elemental_conversion_qt(parent=self)
        if df is not None:
            self.dataset = df
                        
            element_cols = [col for col in df.columns if col not in ["X", "Y"]]

            self.pipeline_config["original_dataset"] = df.copy()    
            self.pipeline_config["original_coordinates"] = df[["X", "Y"]].copy()
            self.original_coordinates = df[["X", "Y"]].copy()
            self.pipeline_config["important_elements"] = element_cols

            self.dataset = df[element_cols].copy()  # drop X/Y from feature set

            self.log_output.append("Dataset loaded and processed successfully.")
            self.left_canvas.show_dummy("Loaded elemental map")
        else:
            self.log_output.append("Dataset loading canceled or failed.")

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- Top Button Row ---
        top_buttons = QHBoxLayout()

        # Split-button: Load (with menu)
        self.load_btn = QToolButton()
        self.load_btn.setText("Load")
        self.load_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.load_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.load_btn.setPopupMode(QToolButton.MenuButtonPopup)

        menu = QMenu(self.load_btn)

        act_load = QAction("Load from HDF…", self)
        act_load.triggered.connect(self.load_dataset_qt)

        act_load_save = QAction("Load + Save raw CSV…", self)
        act_load_save.triggered.connect(self.load_dataset_and_save_raw_qt)

        menu.addAction(act_load)
        menu.addAction(act_load_save)
        menu.addSeparator()

        act_import = QAction("Import images.csv…", self)
        act_import.triggered.connect(self.import_raw_images_csv_qt)

        menu.addAction(act_import)

        self.load_btn.setMenu(menu)

        # default (left-click) = Load from HDF unless user picked something else before
        default_txt = self.pipeline_config.get("load_button_default", "Load from HDF…")
        for a in menu.actions():
            if a.text() == default_txt:
                self.load_btn.setDefaultAction(a)
                break
        else:
            self.load_btn.setDefaultAction(act_load)

        # Remember the last chosen action as default (per session)
        def _remember_default(action):
            self.load_btn.setDefaultAction(action)
            self.pipeline_config["load_button_default"] = action.text()

        for a in (act_load, act_load_save, act_import):
            a.triggered.connect(lambda checked=False, aa=a: _remember_default(aa))

        # Other top buttons (unchanged)
        self.run_pipeline_button = QPushButton("Run Full Pipeline")
        self.preview_button = QPushButton("Preview Dataset")
        self.preview_button.clicked.connect(lambda: self._open_dialog_safely(self.show_dataset_preview))

        top_buttons.addWidget(self.load_btn)
        top_buttons.addWidget(self.run_pipeline_button)
        top_buttons.addWidget(self.preview_button)
        main_layout.addLayout(top_buttons)

        # --- Scrollable Config Area ---
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        scroll_layout.addWidget(self.create_filtering_section())
        scroll_layout.addWidget(self.create_feature_fusion_section())
        scroll_layout.addWidget(self.create_projection_section())
        scroll_layout.addWidget(self.create_clustering_section())

        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 3)

        # --- Splitter Canvas View ---
        splitter = QSplitter(Qt.Horizontal)
        self.left_canvas = ImageCanvas("Image View")
        self.right_canvas = ImageCanvas("Projection View")
        self.left_canvas.show_dummy("Raw image here")
        self.right_canvas.show_dummy("PCA/NMF projection here")
        splitter.addWidget(self.left_canvas)
        splitter.addWidget(self.right_canvas)
        main_layout.addWidget(splitter, 3)

        # --- Export canvases row ---
        export_row = QHBoxLayout()
        self.save_canvases_btn = QPushButton("Save canvases → PNG")
        self.save_canvases_btn.clicked.connect(self.save_canvases_png)
        export_row.addWidget(self.save_canvases_btn)
        main_layout.addLayout(export_row)

        # --- Log Output ---
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Pipeline log output will appear here.")
        main_layout.addWidget(self.log_output, 1)

    def unmark_button_done(self, button):
        if button:
            button.setStyleSheet("")  # Reset styling
   
    def show_dataset_preview(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded to preview.")
            return

        # Preview only a slice for responsiveness
        if hasattr(self.dataset, "head"):
            df_preview = self.dataset.head(200).reset_index(drop=True)
        else:
            import pandas as pd
            df_preview = pd.DataFrame(self.dataset)

        # NEW: pass full df as well (so export uses everything)
        dialog = DataPreviewDialog("Dataset Preview", df_preview, full_df=self.dataset, parent=self)
        dialog.exec()

    def mark_button_done(self, button):
        palette = button.palette()
        palette.setColor(QPalette.Button, QColor(180, 255, 180))
        button.setAutoFillBackground(True)
        button.setPalette(palette)
        button.update()

    def create_filtering_section(self):
        group = QGroupBox("1. Filtering")
        layout = QVBoxLayout()

        # --- Moran's Filter ---
        moran_row = QHBoxLayout()
        self.moran_button = QPushButton("Run Moran's Filter")
        self.moran_button.clicked.connect(self.run_moran_filter)
        self.moran_threshold = QDoubleSpinBox()
        self.moran_threshold.setRange(-1.0, 1.0)
        self.moran_threshold.setValue(0.4)
        moran_row.addWidget(self.moran_button)
        moran_row.addWidget(QLabel("Threshold:"))
        moran_row.addWidget(self.moran_threshold)
        layout.addLayout(moran_row)

        # --- Foreground Clustering ---
        layout.addWidget(QLabel("Foreground Clustering:"))

        fg_row = QHBoxLayout()
        fg_row.addWidget(QLabel("Clusters:"))
        self.foreground_clusters = QSpinBox()
        self.foreground_clusters.setRange(2, 10)
        self.foreground_clusters.setValue(2)
        fg_row.addWidget(self.foreground_clusters)

        fg_row.addWidget(QLabel("Pick cluster:"))
        self.cluster_selector = QComboBox()
        fg_row.addWidget(self.cluster_selector)
        layout.addLayout(fg_row)

        # Populate cluster dropdown when cluster count changes
        def update_cluster_dropdown():
            self.cluster_selector.clear()
            self.cluster_selector.addItem("Manual selection")
            for i in range(self.foreground_clusters.value()):
                self.cluster_selector.addItem(f"Cluster {i}")

        self.foreground_clusters.valueChanged.connect(update_cluster_dropdown)
        update_cluster_dropdown()

        # Add Foreground Clustering button in its own row
        fg_button_row = QHBoxLayout()
        self.foreground_button = QPushButton("Run Foreground Clustering")
        self.foreground_button.clicked.connect(self.run_foreground_clustering)
        # fg_button_row.addStretch()  # To center-align if desired
        fg_button_row.addWidget(self.foreground_button)
        # fg_button_row.addStretch()
        layout.addLayout(fg_button_row)

        # Mean / Std checkboxes row
        fg_opts_row = QHBoxLayout()
        self.foreground_mean = QCheckBox("Use Mean")
        self.foreground_mean.setChecked(True)
        self.foreground_std = QCheckBox("Use Std")
        fg_opts_row.addWidget(self.foreground_mean)
        fg_opts_row.addWidget(self.foreground_std)
        layout.addLayout(fg_opts_row)

        # --- Element Exclusion ---
        self.element_button = QPushButton("Run Element Exclusion")
        self.element_button.clicked.connect(self.run_element_exclusion)
        layout.addWidget(self.element_button)

        group.setLayout(layout)
        return group

    def create_feature_fusion_section(self):
        group = QGroupBox("2. Feature Fusion")
        layout = QVBoxLayout()

        # Add XY Coordinates
        xy_row = QHBoxLayout()
        self.xy_button = QPushButton("Add XY Coordinates")
        self.xy_button.clicked.connect(self.run_add_xy)
        xy_row.addWidget(self.xy_button)
        layout.addLayout(xy_row)

        # Enable Scaling
        scale_row = QHBoxLayout()
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Standard", "MinMax"])
        self.scale_button = QPushButton("Apply Scaling")
        self.scale_button.clicked.connect(self.run_scaler)
        scale_row.addWidget(self.scale_button)
        scale_row.addWidget(QLabel("Method:"))
        scale_row.addWidget(self.scale_combo)
        layout.addLayout(scale_row)

        # Create Augmented Dataset
        augment_row = QHBoxLayout()
        self.augment_button = QPushButton("Create Augmented Dataset")
        self.augment_button.clicked.connect(self.run_augmented_dataset)
        augment_row.addWidget(self.augment_button)
        layout.addLayout(augment_row)

        self.augment_element_scale = QComboBox()
        self.augment_element_scale.addItems(["Standard", "MinMax"])
        self.augment_xy_scale = QComboBox()
        self.augment_xy_scale.addItems(["Standard", "MinMax"])
        self.weight_elements = QDoubleSpinBox()
        self.weight_elements.setValue(1.0)
        self.weight_xy = QDoubleSpinBox()
        self.weight_xy.setValue(1.0)

        layout.addWidget(QLabel("Element Scaler:"))
        layout.addWidget(self.augment_element_scale)
        layout.addWidget(QLabel("XY Scaler:"))
        layout.addWidget(self.augment_xy_scale)
        layout.addWidget(QLabel("Weight (Elements):"))
        layout.addWidget(self.weight_elements)
        layout.addWidget(QLabel("Weight (XY):"))
        layout.addWidget(self.weight_xy)

        group.setLayout(layout)
        return group

    def create_projection_section(self):
        group = QGroupBox("3. Projections")
        layout = QVBoxLayout()

        # --- Shared components ---
        shared_layout = QHBoxLayout()
        self.proj_components = QSpinBox()
        self.proj_components.setMinimum(1)
        self.proj_components.setValue(2)
        shared_layout.addWidget(QLabel("Components:"))
        shared_layout.addWidget(self.proj_components)
        layout.addLayout(shared_layout)

        # --- PCA / NMF / t-SNE ---
        self.pca_checkbox = QCheckBox("Enable PCA")
        layout.addWidget(self.pca_checkbox)

        self.nmf_checkbox = QCheckBox("Enable NMF")
        layout.addWidget(self.nmf_checkbox)
        self.nmf_max_iter = QSpinBox()
        self.nmf_max_iter.setRange(10, 1000)
        self.nmf_max_iter.setValue(200)
        self.nmf_iter_enabled = QCheckBox("Set max_iter")
        nmf_row = QHBoxLayout()
        nmf_row.addWidget(self.nmf_iter_enabled)
        nmf_row.addWidget(QLabel("max_iter:"))
        nmf_row.addWidget(self.nmf_max_iter)
        layout.addLayout(nmf_row)

        self.tsne_checkbox = QCheckBox("Enable t-SNE")
        layout.addWidget(self.tsne_checkbox)
        self.tsne_perplexity = QDoubleSpinBox()
        self.tsne_perplexity.setRange(5.0, 100.0)
        self.tsne_perplexity.setValue(30.0)
        self.tsne_perplexity.setSingleStep(1.0)
        tsne_row = QHBoxLayout()
        tsne_row.addWidget(QLabel("Perplexity:"))
        tsne_row.addWidget(self.tsne_perplexity)
        layout.addLayout(tsne_row)

        # --- UMAP ---
        self.umap_checkbox = QCheckBox("Enable UMAP")
        layout.addWidget(self.umap_checkbox)

        self.umap_neighbors = QSpinBox()
        self.umap_neighbors.setRange(2, 200)
        self.umap_neighbors.setValue(15)
        umap_row1 = QHBoxLayout()
        umap_row1.addWidget(QLabel("Neighbors:"))
        umap_row1.addWidget(self.umap_neighbors)
        layout.addLayout(umap_row1)

        # Metric dropdown + options button + 1-line description
        self.umap_metric = QComboBox()
        self.umap_metric.setEditable(False)
        self._populate_umap_metrics()
        self.umap_metric.currentIndexChanged.connect(self._umap_metric_changed)

        self.umap_metric_opts = QPushButton("Metric options…")
        self.umap_metric_opts.setVisible(False)
        self.umap_metric_opts.setEnabled(False)
        self.umap_metric_opts.clicked.connect(self.open_umap_metric_params)

        umap_row2 = QHBoxLayout()
        umap_row2.addWidget(QLabel("Metric:"))
        umap_row2.addWidget(self.umap_metric, 1)
        umap_row2.addWidget(self.umap_metric_opts)
        layout.addLayout(umap_row2)

        self.umap_metric_desc = QLabel("")
        self.umap_metric_desc.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.umap_metric_desc)

        # Prime description & button state once
        self._umap_metric_changed()

        # --- Run button ---
        self.projection_button = QPushButton("Run Projection(s)")
        self.projection_button.clicked.connect(self.run_projection_methods)
        layout.addWidget(self.projection_button)

        group.setLayout(layout)
        return group

    def create_clustering_section(self):
        group = QGroupBox("4. Clustering")
        layout = QVBoxLayout()

        # Select base for clustering
        self.clustering_base_selector = QComboBox()
        self.clustering_base_selector.addItem("Dataset")  # Default choice
        layout.addWidget(QLabel("Select clustering base:"))
        layout.addWidget(self.clustering_base_selector)

        # Number of clusters
        cluster_layout = QHBoxLayout()
        self.cluster_count_spin = QSpinBox()
        self.cluster_count_spin.setMinimum(2)
        self.cluster_count_spin.setMaximum(20)
        self.cluster_count_spin.setValue(2)
        cluster_layout.addWidget(QLabel("Number of clusters:"))
        cluster_layout.addWidget(self.cluster_count_spin)
        layout.addLayout(cluster_layout)

        # Run Clustering
        self.run_clustering_button = QPushButton("Run Clustering")
        self.run_clustering_button.clicked.connect(self.run_kmeans_clustering)
        layout.addWidget(self.run_clustering_button)

        score_row = QHBoxLayout()
        self.silhouette_button = QPushButton("Silhouette Score")
        self.silhouette_button.clicked.connect(lambda: self._open_dialog_safely(self.show_silhouette_popup))
        self.dbscore_button = QPushButton("DB Score")
        self.dbscore_button.clicked.connect(lambda: self._open_dialog_safely(self.show_db_score_popup))
        score_row.addWidget(self.silhouette_button)
        score_row.addWidget(self.dbscore_button)
        layout.addLayout(score_row)

        self.edit_colors_button = QPushButton("Edit colors…")
        self.edit_colors_button.setEnabled(False)
        self.edit_colors_button.clicked.connect(self.open_color_picker)
        layout.addWidget(self.edit_colors_button)

        # --- Cluster analysis row (spectra + concentrations)
        row = QHBoxLayout()
        self.cluster_spectra_button = QPushButton("Cluster Spectra (log)")
        self.cluster_spectra_button.setEnabled(True)
        self.cluster_spectra_button.clicked.connect(
            lambda: self._open_dialog_safely(self.show_cluster_spectra_popup)
        )

        self.spectra_roi_button = QPushButton("Spectra & ROI…")
        self.spectra_roi_button.setEnabled(True)
        self.spectra_roi_button.clicked.connect(
            lambda: self._open_dialog_safely(self.open_spectra_roi_dialog)
        )
        row.addWidget(self.spectra_roi_button)

        self.cluster_conc_button = QPushButton("Cluster Concentrations")
        self.cluster_conc_button.setEnabled(True)
        self.cluster_conc_button.clicked.connect(
            lambda: self._open_dialog_safely(self.show_cluster_concentrations_popup)
        )

        row.addWidget(self.cluster_spectra_button)
        row.addWidget(self.cluster_conc_button)
        layout.addLayout(row)

        group.setLayout(layout)
        return group

    def run_moran_filter(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded. Please load a dataset first.")
            return

        coords = self.pipeline_config.get("original_coordinates")
        if coords is None:
            self.log_output.append("Original coordinates not found.")
            return

        # Current feature set (drop XY if present)
        feature_df = self.dataset.drop(columns=["X", "Y"], errors="ignore")
        if feature_df.empty:
            self.log_output.append("No element features available for Moran's I.")
            return

        # Cache: full scores + grid shape
        cache = self.pipeline_config.get("_moran_cache")
        width = int(coords["X"].max()) + 1
        height = int(coords["Y"].max()) + 1

        # Decide cache hit/miss
        cache_hit = False
        if cache and cache.get("width") == width and cache.get("height") == height:
            cached_scores = cache.get("scores")
            if cached_scores is not None:
                # If the cache has all current columns, subset; otherwise recompute
                cur_cols = list(feature_df.columns)
                if set(cur_cols).issubset(set(cached_scores.index)):
                    moran_df_all = cached_scores.loc[cur_cols].copy()
                    cache_hit = True

        if not cache_hit:
            # Compute and store
            try:
                moran_df_all = compute_moran_scores(feature_df, coordinates_df=coords)
                self.pipeline_config["_moran_cache"] = {
                    "scores": moran_df_all.copy(),
                    "width": width,
                    "height": height,
                }
                self.log_output.append("[Moran] cache miss → computed scores.")
            except Exception as e:
                self.log_output.append(f"Moran's I computation failed: {str(e)}")
                return
        else:
            self.log_output.append("[Moran] cache hit → reused scores.")

        # Threshold handling (reuse your spinbox; if <=0, prompt)
        threshold = self.moran_threshold.value()
        if threshold <= 0.0:
            from PySide6.QtWidgets import QInputDialog
            # Build a readable list (top 50 to keep dialog short)
            listing = "\n".join(
                f"{idx}: {row['Morans_I']:.4f}"
                for idx, row in moran_df_all.head(50).iterrows()
            )
            threshold_str, ok = QInputDialog.getText(
                self,
                "Select Moran's I Threshold",
                "Enter threshold value (showing top 50 by Moran's I):\n\n" + listing
            )
            if not ok:
                self.log_output.append("Moran's I threshold selection cancelled.")
                return
            try:
                threshold = float(threshold_str)
            except ValueError:
                self.log_output.append("Invalid threshold input.")
                return

        # Apply threshold to the full (cached) scores
        moran_kept = moran_df_all[moran_df_all["Morans_I"] > float(threshold)]
        important_elements = moran_kept.index.tolist()

        if not important_elements:
            self.log_output.append("No elements passed the Moran's I threshold.")
            return

        self.pipeline_config['important_elements'] = important_elements
        # Reduce dataset to those elements (XY excluded — use Add XY to toggle)
        self.dataset = feature_df[important_elements].copy()

        self.log_output.append(f"{len(important_elements)} elements passed Moran's I threshold.")
        self.left_canvas.show_dummy("Moran filtering result")
        self.mark_button_done(self.moran_button)

    def run_foreground_clustering(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded.")
            return

        k = self.foreground_clusters.value()
        use_mean = self.foreground_mean.isChecked()
        use_std = self.foreground_std.isChecked()

        # Exclude XY unless explicitly added
        feature_cols = [col for col in self.dataset.columns if col not in ("X", "Y")]
        feature_df = self.dataset[feature_cols]

        labels, clustered_df = foreground_clustering(
            feature_df,
            k=k,
            use_mean=use_mean,
            use_std=use_std,
            return_clustered_data=True
        )

        # Log all clusters
        log_lines = []
        for cluster_id in sorted(clustered_df["Cluster"].unique()):
            elements = clustered_df[clustered_df["Cluster"] == cluster_id]["Element"].tolist()
            log_lines.append(f"Cluster {cluster_id}: " + ", ".join(elements))
        self.log_output.append("\n".join(log_lines))

        self.pipeline_config["clustered_elements"] = clustered_df
        self.pipeline_config["cluster_labels"] = sorted(clustered_df["Cluster"].unique().tolist())

        # Determine selected cluster
        selected_text = self.cluster_selector.currentText()
        if selected_text == "Manual selection":
            dialog = ClusterSelectionDialog(clustered_df, parent=self)
            if dialog.exec():
                cluster_idx = dialog.get_selected_cluster()
            else:
                self.log_output.append("Cluster selection cancelled.")
                return
        else:
            try:
                cluster_idx = int(selected_text.split()[-1])
            except Exception:
                self.log_output.append("Invalid cluster selection.")
                return

        # Final selection and update
        selected_elements = clustered_df[clustered_df["Cluster"] == cluster_idx]["Element"].tolist()
        self.dataset = self.pipeline_config["original_dataset"][selected_elements].copy()
        self.pipeline_config["important_elements"] = selected_elements

        self.log_output.append(f"Selected Cluster {cluster_idx} with {len(selected_elements)} elements.")
        self.left_canvas.show_dummy(f"Cluster {cluster_idx} foreground")
        self.mark_button_done(self.foreground_button)

    def run_element_exclusion(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded.")
            return

        current_elements = self.pipeline_config.get("important_elements", list(self.dataset.columns))
        all_elements = list(self.pipeline_config["original_dataset"].columns)
        all_elements = [el for el in all_elements if el not in ("X", "Y")]

        coords = self.pipeline_config.get("original_coordinates")
        if coords is None:
            self.log_output.append("Missing coordinates. Cannot display image grid.")
            return

        dialog = ElementImageSelectionDialog(
            all_elements,
            current_elements,
            self.pipeline_config["original_dataset"],
            coords["X"].max() + 1,
            coords["Y"].max() + 1,
            self
        )

        # live apply without closing
        def _apply_live(selected):
            selected = [el for el in selected if el not in ("X", "Y")]
            self.pipeline_config["important_elements"] = selected
            self.dataset = self.pipeline_config["original_dataset"][selected].copy()
            self.log_output.append(f"[Apply] {len(selected)} elements staged into dataset (dialog open).")
            self.left_canvas.show_dummy("Filtered element dataset (applied)")

        dialog.applied.connect(_apply_live)

        res = dialog.exec()                  # <<< single exec
        if res:                               # QDialog.Accepted
            selected = dialog.get_selected_elements()
            selected = [el for el in selected if el not in ("X", "Y")]
            self.pipeline_config["important_elements"] = selected
            self.dataset = self.pipeline_config["original_dataset"][selected].copy()
            self.log_output.append(f"Selected {len(selected)} elements after exclusion.")
            self.left_canvas.show_dummy("Filtered element dataset")
            self.mark_button_done(self.element_button)

    def run_add_xy(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded.")
            return

        xy = self.pipeline_config.get("original_coordinates")
        elements = self.pipeline_config.get("important_elements")

        if xy is None or elements is None:
            self.log_output.append("Original coordinates or elements not available.")
            return

        df = self.dataset.copy()

        # Toggle behavior
        if "X" in df.columns and "Y" in df.columns:
            # Remove X and Y
            df = df.drop(columns=["X", "Y"])
            self.pipeline_config["xy_added"] = False
            self.log_output.append("Removed X and Y from dataset.")
            self.unmark_button_done(self.xy_button)  # Optional helper
        else:
            # Add X and Y to beginning
            if len(xy) != len(df):
                self.log_output.append("Mismatch in coordinate length.")
                return

            if "X" not in df.columns:
                df.insert(0, "X", xy["X"].values)
            if "Y" not in df.columns:
                df.insert(1, "Y", xy["Y"].values)

            self.pipeline_config["xy_added"] = True
            self.log_output.append("Added X and Y coordinates to dataset.")
            self.mark_button_done(self.xy_button)

        self.dataset = df
        self.left_canvas.show_dummy("Toggled XY")

    def run_scaler(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded.")
            return

        method = self.scale_combo.currentText().lower()  # 'standard' or 'minmax'

        try:
            self.dataset = scale_dataset(self.dataset, scaler_type=method)
            self.pipeline_config["scaled"] = True
            self.pipeline_config["scaler_used"] = method
            self.log_output.append(f"Scaled dataset using {method.capitalize()} scaler.")
            self.left_canvas.show_dummy(f"{method.capitalize()} Scaled Dataset")
            self.mark_button_done(self.scale_button)
        except Exception as e:
            self.log_output.append(f"Scaling failed: {str(e)}")

    def run_augmented_dataset(self):

        if self.dataset is None:
            self.log_output.append("No dataset loaded.")
            return

        original_coords = self.pipeline_config.get("original_coordinates")
        if original_coords is None:
            self.log_output.append("Missing original coordinates.")
            return

        # Get user-selected options
        element_scaler = self.augment_element_scale.currentText().lower()
        xy_scaler = self.augment_xy_scale.currentText().lower()
        weight_elements = self.weight_elements.value()
        weight_xy = self.weight_xy.value()

        try:
            df_aug = create_augmented_dataset(
                self.dataset.copy(),
                original_coords=original_coords,
                element_scaler=element_scaler,
                xy_scaler=xy_scaler,
                weight_elements=weight_elements,
                weight_xy=weight_xy
            )
            self.dataset = df_aug
            self.pipeline_config["augmented"] = True
            self.pipeline_config["scaler_xy"] = xy_scaler
            self.pipeline_config["scaler_elements"] = element_scaler
            self.log_output.append(
                f"Augmented dataset created using {element_scaler} scaler for elements and {xy_scaler} for XY "
                f"with weights {weight_elements:.2f} and {weight_xy:.2f}."
            )
            self.left_canvas.show_dummy("Augmented dataset")
            self.mark_button_done(self.augment_button)

        except Exception as e:
            self.log_output.append(f"Augmentation failed: {str(e)}")

    def update_clustering_base_options(self):
        # Start with "Dataset"
        options = ["Dataset"]
        
        # Check if each projection exists in pipeline_config
        projections = self.pipeline_config.get("projections", {})
        for name in ["PCA", "NMF", "TSNE", "UMAP"]:
            if name in projections:
                options.append(name)
        
        # Clear and update the dropdown
        self.clustering_base_selector.clear()
        self.clustering_base_selector.addItems(options)

    def run_projection_methods(self):
        """
        Run enabled projection methods (PCA, NMF, UMAP, TSNE) on the current dataset.
        Stores projections in self.pipeline_config["projections"]
        """
        if self.dataset is None:
            self.log_output.append("No dataset loaded. Please load and filter first.")
            return

        X = self.dataset.values
        projections = {}

        try:
            n_components = self.proj_components.value()

            if self.pca_checkbox.isChecked():
                pca = PCA(n_components=n_components, random_state=0)
                projections["PCA"] = pca.fit_transform(X)
                self.log_output.append(f"PCA projection done ({n_components} components).")

            if self.nmf_checkbox.isChecked():
                nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000)
                projections["NMF"] = nmf.fit_transform(np.maximum(X, 0))  # NMF requires non-negative
                self.log_output.append(f"NMF projection done ({n_components} components).")

            if self.umap_checkbox.isChecked():
                neighbors = self.umap_neighbors.value()
                metric = self.umap_metric.currentText()
                if metric.startswith("—") or not metric:
                    metric = "euclidean"

                X_umap = self.dataset.values  # current base
                raw_kwds = self.pipeline_config.get("umap_metric_kwds", {}).get(metric, None)
                metric_kwds = self._umap_prepare_metric_kwds(metric, X_umap, raw_kwds)

                umap_model = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=neighbors,
                    metric=metric,
                    metric_kwds=metric_kwds if metric_kwds else None,
                    n_jobs=-1,
                )
                projections["UMAP"] = umap_model.fit_transform(X_umap)
                
            if self.tsne_checkbox.isChecked():
                perplexity = self.tsne_perplexity.value()
                tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=0)
                projections["TSNE"] = tsne.fit_transform(X)
                self.log_output.append(f"t-SNE projection done ({n_components} components, perplexity={perplexity}).")

            if not projections:
                self.log_output.append("No projection method selected.")

            self.pipeline_config["projections"] = projections
            self.left_canvas.show_dummy("Projection(s) complete.")
            self.mark_button_done(self.projection_button)

        except Exception as e:
            self.log_output.append(f"Projection error: {str(e)}")
        

        self.update_clustering_base_options()

    def prompt_cluster_choice(self, num_clusters):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Foreground Cluster")

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Which cluster do you want to keep as foreground?"))

        combo = QComboBox()
        for i in range(num_clusters):
            combo.addItem(f"Cluster {i}")
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            return combo.currentIndex()
        else:
            return None

    def run_kmeans_clustering(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded.")
            return

        k = self.cluster_count_spin.value()
        self.visible_clusters = set(range(k))
        source = self.clustering_base_selector.currentText()

        # Get clustering base data
        if source == "Dataset":
            X = self.dataset.values
            used_base = "Dataset"
        else:
            projection_data = self.pipeline_config.get("projections", {}).get(source)
            if projection_data is None:
                self.log_output.append(f"Projection '{source}' not found.")
                return
            X = projection_data
            used_base = source

        # Run KMeans
        try:
            model = KMeans(n_clusters=k, random_state=0)
            labels = model.fit_predict(X)

            # Store results
            self.pipeline_config["cluster_labels_kmeans"] = labels
            self.pipeline_config["kmeans_model"] = model
            self.pipeline_config["clustering"] = {
                "labels": labels,
                "num_clusters": k,
                "used_base": used_base
            }
            
            # Store stable RGBA list
            base = mpl.colormaps["tab20"].resampled(k)
            self.cluster_rgba = list(base.colors)          # k RGBA rows
            self.cluster_cmap = ListedColormap(self.cluster_rgba)
            
            self.pipeline_config["cluster_cmap"] = self.cluster_cmap

            # Log summary
            unique, counts = np.unique(labels, return_counts=True)
            log_lines = [f"KMeans clustering complete using base: {used_base}"]
            for u, c in zip(unique, counts):
                log_lines.append(f"Cluster {u}: {c} samples")

            total = counts.sum()
            pct_by_cluster = {int(u): (100.0 * int(c) / float(total)) for u, c in zip(unique, counts)}
            self.pipeline_config["cluster_pct"] = pct_by_cluster

            self.log_output.append("\n".join(log_lines))
            self.mark_button_done(self.run_clustering_button)

            # After logging:
            self.mark_button_done(self.run_clustering_button)

            # enable color editor
            if hasattr(self, "edit_colors_button"):
                self.edit_colors_button.setEnabled(True)
                
            self.update_visibility_render()         # renders image + projections with outlines
                        

        except Exception as e:
            self.log_output.append(f"KMeans clustering failed: {str(e)}")

    def visualize_clustering_results(self):
        clustering_info = self.pipeline_config.get("clustering", {})
        labels = clustering_info.get("labels")
        if labels is None:
            self.log_output.append("No clustering labels found.")
            return

        # === LEFT: Image view based on original coordinates ===
        xy = self.pipeline_config.get("original_coordinates")
        if xy is None:
            self.log_output.append("No original coordinates found.")
            return

        try:
            labels = np.array(labels).flatten()  # Ensures 1D array

            if len(labels) != len(xy):
                self.log_output.append(f"Label count ({len(labels)}) does not match coordinate count ({len(xy)})")
                return

            width = int(np.max(xy["X"])) + 1
            height = int(np.max(xy["Y"])) + 1
            image = np.full((height, width), -1, dtype=int)

            for idx, (x, y) in enumerate(xy[["X", "Y"]].values):
                xi, yi = int(x), int(y)
                if 0 <= xi < width and 0 <= yi < height:
                    image[yi, xi] = int(labels[idx])
                else:
                    self.log_output.append(f"Out-of-bounds coordinate: ({xi}, {yi}) skipped.")

            self.display_image_on_canvas(image, canvas="left")
        except Exception as e:
            self.log_output.append(f"Error displaying image: {str(e)}")

        # === RIGHT: Projection scatter plots ===
        projections = self.pipeline_config.get("projections", {})
        if not projections:
            self.log_output.append("No projections to display.")
            return

        try:
            self.display_projections_on_canvas(projections, labels)
        except Exception as e:
            self.log_output.append(f"Error displaying projections: {str(e)}")

    def display_image_on_canvas(self, image, canvas="left"):
        target_canvas = self.left_canvas if canvas == "left" else self.right_canvas
        ax = target_canvas.ax
        ax.clear()
        im = ax.imshow(image, 
                       cmap=self.pipeline_config.get("cluster_cmap", "tab20"), 
                       interpolation="nearest",
                       )
        target_canvas.draw()
        
    def display_projections_on_canvas(self, projections, labels):
        fig = self.right_canvas.figure
        fig.clear()

        keys = list(projections.keys())
        n = len(keys)
        cols = 2
        rows = (n + cols - 1) // cols

        axes = fig.subplots(rows, cols).flatten()

        for idx, name in enumerate(keys):
            if idx >= len(axes):
                break
            ax = axes[idx]
            proj = projections[name]
            scatter = ax.scatter(
                proj[:, 0], proj[:, 1], c=labels, 
                cmap=self.pipeline_config.get("cluster_cmap", "tab20"), 
                s=2, picker=True)
            ax.set_title(name)
            ax.axis("off")

        # Hide extra subplots
        for j in range(len(keys), len(axes)):
            axes[j].axis("off")

        self.right_canvas.draw()

    def on_projection_pick(self, event):
        if not hasattr(event, "ind") or not event.ind.size:
            return

        ind = event.ind[0]
        labels = self.pipeline_config.get("clustering", {}).get("labels")
        if labels is None:
            return

        cluster_id = int(labels[ind])

        # Toggle ON/OFF
        if cluster_id in self.visible_clusters:
            self.visible_clusters.remove(cluster_id)
        else:
            self.visible_clusters.add(cluster_id)

        # Redraw both panes with the new visibility
        self.update_visibility_render()

    def update_visibility_render(self):
        clustering = self.pipeline_config.get("clustering", {})
        labels = clustering.get("labels")
        if labels is None:
            return
        labels = np.asarray(labels).flatten()
        k = int(clustering.get("num_clusters", labels.max() + 1))
        xy = self.pipeline_config.get("original_coordinates")
        if xy is None:
            return

        width = int(np.max(xy["X"])) + 1
        height = int(np.max(xy["Y"])) + 1
        img = np.full((height, width, 4), (1, 1, 1, 0), dtype=float)  # RGBA

        for idx, (x, y) in enumerate(xy[["X", "Y"]].values.astype(int)):
            if 0 <= x < width and 0 <= y < height:
                lbl = int(labels[idx])
                if lbl in self.visible_clusters:
                    img[y, x] = self.cluster_rgba[lbl]
                else:
                    img[y, x] = (1, 1, 1, 0)  # fully transparent

        # Left image — light gray background so transparent shows
        axL = self.left_canvas.ax
        axL.clear()
        axL.set_facecolor("#f2f2f2")
        axL.imshow(img, interpolation="nearest")
        axL.axis("off")
        self.left_canvas.draw()

        # --- Right projections (replot with visibility)
        projections = self.pipeline_config.get("projections", {})
        figR = self.right_canvas.figure
        figR.clear()
        keys = list(projections.keys())
        if keys:
            cols = 2
            rows = (len(keys) + cols - 1) // cols
            axes = figR.subplots(rows, cols).flatten()

            vis = np.isin(labels, list(self.visible_clusters))

            facecolors = np.array([
                self.cluster_rgba[lbl] if v else self.lighten_color(self.cluster_rgba[lbl], 0.3)
                for lbl, v in zip(labels, vis)
            ])

            # edge: visible = 0.5 alpha, hidden = 0.15 alpha (subtle outline stays)
            edgecolors = np.array([
                (0, 0, 0, 0.5) if v else (0.6, 0.6, 0.6, 0.5)
                for v in vis
            ])

            for i, name in enumerate(keys):
                if i >= len(axes):
                    break
                ax = axes[i]
                proj = projections[name]
                ax.set_facecolor("#f2f2f2")  # light bg so faint points still show
                ax.scatter(
                    proj[:, 0], proj[:, 1],
                    s=6,
                    facecolors=facecolors,
                    edgecolors=edgecolors,
                    linewidths=0.2,
                    picker=True
                )
                ax.set_title(name)
                ax.axis("off")

            for j in range(len(keys), len(axes)):
                axes[j].axis("off")
        self.right_canvas.draw()
        self._draw_left_cluster_legend()  # show palette/ids next to image

    def _draw_left_cluster_legend(self):
        # remove previous legend, if any
        if hasattr(self, "_left_legend_ax") and self._left_legend_ax in self.left_canvas.figure.axes:
            self._left_legend_ax.remove()
            self._left_legend_ax = None

        if not hasattr(self, "cluster_rgba"):
            self.left_canvas.draw()
            return

        # --- pull labels + stats ---
        clustering = self.pipeline_config.get("clustering", {})
        labels = np.asarray(clustering.get("labels", []), dtype=int)
        if labels.size == 0:
            self.left_canvas.draw()
            return

        k = len(self.cluster_rgba)
        counts = np.bincount(labels, minlength=k)
        total = int(counts.sum())
        # avoid div-by-zero
        pcts = (counts / total * 100.0) if total > 0 else np.zeros_like(counts, dtype=float)

        ax = self.left_canvas.ax

        # place a slim inset just OUTSIDE the left edge of the image axes
        # (slightly wider so the new text fits nicely)
        LEG_SHIFT = 0.75
        leg_ax = inset_axes(
            ax,
            width="12%", height="82%",
            loc="center left",
            bbox_to_anchor=(-LEG_SHIFT, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        self._left_legend_ax = leg_ax
        leg_ax.set_axis_off()

        # header: total pixels
        leg_ax.text(
            0.02, 1.02,
            f"Total: {total:,} px",
            transform=leg_ax.transAxes,
            ha="left", va="bottom",
            fontsize=9, fontweight="bold"
        )

        visible = getattr(self, "visible_clusters", set(range(k)))

        # rows
        for i in range(k):
            y = 1 - (i + 0.5) / k
            r, g, b, a = self.cluster_rgba[i]
            color = (r, g, b, 0.25) if i not in visible else (r, g, b, a)

            # swatch
            rect = mpatches.Rectangle(
                (0.05, y - 0.035), 0.38, 0.07,
                transform=leg_ax.transAxes,
                facecolor=color, edgecolor="k", lw=0.5, clip_on=False
            )
            leg_ax.add_patch(rect)

            # label text: "id — XX.X% (n=###)"
            leg_ax.text(
                0.47, y,
                f"{i} — {pcts[i]:.1f}% (n={counts[i]:,})",
                transform=leg_ax.transAxes,
                ha="left", va="center",
                fontsize=9
            )

        self.left_canvas.draw()

    def lighten_color(self, color, factor=0.7):
        """
        Lighten an RGBA color by blending it with white.
        factor: 0 = white, 1 = original color
        """
        r, g, b, a = color
        r = r + (1 - r) * (1 - factor)
        g = g + (1 - g) * (1 - factor)
        b = b + (1 - b) * (1 - factor)
        return (r, g, b, 1.0)  # full alpha
    
    def open_color_picker(self):
        clustering = self.pipeline_config.get("clustering")
        if not clustering:
            self.log_output.append("Run clustering first to edit colors.")
            return
        k = int(clustering["num_clusters"])

        dlg = QDialog(self); dlg.setWindowTitle("Cluster colors")
        grid = QGridLayout(dlg)

        for c in range(k):
            label = QLabel(f"Cluster {c}")
            btn = QPushButton("Pick…")
            btn.setStyleSheet(f"background-color:{to_hex(self.cluster_rgba[c])};")
            btn.clicked.connect(partial(self._pick_cluster_color, c, btn))
            grid.addWidget(label, c, 0)
            grid.addWidget(btn,   c, 1)

        dlg.exec()

    def _pick_cluster_color(self, c, btn):
        dlg = QColorDialog(self)
        dlg.setOption(QColorDialog.DontUseNativeDialog, True)
        if not dlg.exec():
            return
        qcol = dlg.currentColor()
        if not qcol.isValid():
            return

        rgba = (qcol.red()/255.0, qcol.green()/255.0, qcol.blue()/255.0, 1.0)
        self.cluster_rgba[c] = rgba
        if hasattr(self, "cluster_cmap"):
            self.cluster_cmap.colors[c] = rgba
        btn.setStyleSheet(f"background-color:{qcol.name()};")

        # live-refresh projections & image
        self.update_visibility_render()

        # also refresh spectra colors if the dialog is visible
        if self._spectra_dlg and self._spectra_dlg.isVisible():
            # just redraw with the current mean/sum choice
            self.show_cluster_spectra_popup(use_mean=self._spectra_use_mean)
        
        # Keep pipeline_config as the single source of truth for colors
        self.pipeline_config["cluster_rgba"] = list(self.cluster_rgba)

        # If Spectra/ROI dialog is open, push colors live
        dlg = getattr(self, "_spectra_roi_dialog", None)
        if dlg and dlg.isVisible():
            dlg.set_cluster_colors(self.pipeline_config["cluster_rgba"])

    def _populate_umap_metrics(self):
        # sections: (title, [(metric, description, needs_kwds)])
        self._umap_sections = [
            ("Minkowski style metrics", [
                ("euclidean",  "L2 distance (default).", False),
                ("manhattan",  "L1 (cityblock) distance.", False),
                ("chebyshev",  "L∞ (max) distance.", False),
                ("minkowski",  "General Lp distance (p=2).", False),
            ]),
            ("Miscellaneous spatial metrics", [
                ("canberra",   "Weighted L1-type distance.", False),
                ("braycurtis", "Dissimilarity for compositions.", False),
                ("haversine",  "Great-circle distance on lat/long (radians).", False),
            ]),
            ("Normalized spatial metrics", [
                ("mahalanobis","Needs VI (inverse covariance).", True),
                ("wminkowski", "Needs p and weights.", True),
                ("seuclidean", "Needs V (per-feature variances).", True),
            ]),
            ("Angular and correlation metrics", [
                ("cosine",     "Angle-based similarity.", False),
                ("correlation","1 − Pearson correlation.", False),
            ]),
            ("Metrics for binary data", [
                ("hamming",    "Fraction of mismatches.", False),
                ("jaccard",    "1 − (∩/∪).", False),
                ("dice",       "Sørensen–Dice dissimilarity.", False),
            ]),
        ]

        self._umap_descriptions = {}
        self._umap_needs_kwds = set()

        combo = self.umap_metric
        model = combo.model()
        combo.clear()

        # Insert section headers (disabled) and items (with tooltips)
        for title, items in self._umap_sections:
            combo.addItem(f"— {title} —")
            header_idx = combo.count() - 1
            header_item = model.item(header_idx)
            header_item.setFlags(Qt.NoItemFlags)
            f = header_item.font(); f.setBold(True); header_item.setFont(f)

            for name, tip, need in items:
                combo.addItem(name)
                row = combo.count() - 1
                combo.setItemData(row, tip, Qt.ToolTipRole)
                self._umap_descriptions[name] = tip
                if need:
                    self._umap_needs_kwds.add(name)

        # default selection
        idx = combo.findText("euclidean")
        if idx >= 0:
            combo.setCurrentIndex(idx)

        # storage for kwds
        self.pipeline_config.setdefault("umap_metric_kwds", {})

    def _umap_metric_changed(self):
        m = self.umap_metric.currentText()
        # ignore headers
        if m.startswith("—") or not m:
            self.umap_metric_desc.setText("")
            self.umap_metric_opts.setVisible(False)
            self.umap_metric_opts.setEnabled(False)
            return

        # description line
        self.umap_metric_desc.setText(self._umap_descriptions.get(m, ""))

        # toggle the options button for metrics that need params
        need = m in self._umap_needs_kwds
        self.umap_metric_opts.setVisible(need)
        self.umap_metric_opts.setEnabled(need)

    def open_umap_metric_params(self):
        metric = self.umap_metric.currentText()
        if metric not in self._umap_needs_kwds:
            return

        # ensure kwds dict exists
        self.pipeline_config.setdefault("umap_metric_kwds", {})

        dlg = QDialog(self)
        dlg.setWindowTitle(f"UMAP metric options — {metric}")
        lay = QVBoxLayout(dlg)

        def _current_X():
            X = self.dataset.values if hasattr(self.dataset, "values") else np.asarray(self.dataset)
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X

        if metric == "mahalanobis":
            lam_label = QLabel("Regularization λ (adds λI to covariance before inversion):")
            lam = QDoubleSpinBox(); lam.setDecimals(6); lam.setRange(0.0, 1.0); lam.setSingleStep(1e-4); lam.setValue(1e-6)
            lay.addWidget(lam_label); lay.addWidget(lam)

            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            lay.addWidget(btns)
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)

            if dlg.exec() == QDialog.Accepted:
                X = _current_X()
                cov = np.cov(X, rowvar=False)
                cov_reg = cov + lam.value() * np.eye(cov.shape[0], dtype=np.float64)
                try:
                    VI = np.linalg.inv(cov_reg)
                except np.linalg.LinAlgError:
                    QMessageBox.critical(self, "Mahalanobis", "Covariance not invertible even after regularization.")
                    return
                VI = np.ascontiguousarray(VI, dtype=np.float64)
                self.pipeline_config["umap_metric_kwds"][metric] = {"VI": VI}
                self.log_output.append("UMAP: set Mahalanobis VI from current dataset.")

        elif metric == "seuclidean":
            lay.addWidget(QLabel("Standardized Euclidean: uses per-feature variances V from current dataset."))
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            lay.addWidget(btns)
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)

            if dlg.exec() == QDialog.Accepted:
                X = _current_X()
                V = np.var(X, axis=0, ddof=1).astype(np.float64)
                V[V == 0] = 1e-12
                V = np.ascontiguousarray(V, dtype=np.float64)
                self.pipeline_config["umap_metric_kwds"][metric] = {"V": V}
                self.log_output.append("UMAP: set seuclidean V from current dataset.")

        elif metric == "wminkowski":
            form = QFormLayout()
            p = QDoubleSpinBox(); p.setRange(0.01, 10.0); p.setSingleStep(0.1); p.setValue(2.0)
            w_mode = QComboBox(); w_mode.addItems(["Equal weights (1)", "Inverse std (1/σ)"])
            form.addRow("p:", p); form.addRow("Weights:", w_mode)
            lay.addLayout(form)

            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            lay.addWidget(btns)
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)

            if dlg.exec() == QDialog.Accepted:
                mode = "equal" if w_mode.currentIndex() == 0 else "invstd"
                pval = float(p.value())
                # Only store mode+p (no stale arrays!)
                self.pipeline_config.setdefault("umap_metric_kwds", {})
                self.pipeline_config["umap_metric_kwds"]["wminkowski"] = {"mode": mode, "p": pval}
                self.log_output.append(f"UMAP: wminkowski set to mode={mode}, p={pval:.2f}.")

    def _umap_prepare_metric_kwds(self, metric, X, kwds):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_feat = X.shape[1]
        kwds = {} if kwds is None else dict(kwds)

        if metric == "wminkowski":
            p = float(kwds.get("p", 2.0))
            mode = kwds.get("mode", None)

            if mode == "invstd":
                std = np.std(X, axis=0, ddof=1).astype(np.float64)
                std[std == 0] = 1.0
                w = 1.0 / std
            else:
                # default and "equal"
                w = np.ones(n_feat, dtype=np.float64)

            # IMPORTANT: 'w' must come before 'p' for pynndescent
            return {"w": np.ascontiguousarray(w, dtype=np.float64), "p": p}

        if metric == "seuclidean":
            V = kwds.get("V", None)
            if V is None or np.ndim(V) != 1 or len(V) != n_feat:
                V = np.var(X, axis=0, ddof=1).astype(np.float64)
                V[V == 0] = 1e-12
            return {"V": np.ascontiguousarray(V, dtype=np.float64)}

        if metric == "mahalanobis":
            VI = kwds.get("VI", None)
            if VI is None or np.asarray(VI).shape != (n_feat, n_feat):
                cov = np.cov(X, rowvar=False)
                lam = 1e-6
                VI = np.linalg.inv(cov + lam * np.eye(n_feat))
            return {"VI": np.ascontiguousarray(VI, dtype=np.float64)}

        return {}

    def _get_clustering_base_matrix(self):
        source = self.clustering_base_selector.currentText()
        if source == "Dataset":
            if self.dataset is None:
                raise ValueError("No dataset loaded.")
            return self.dataset.values, "Dataset"

        proj = self.pipeline_config.get("projections", {}).get(source)
        if proj is None:
            raise ValueError(f"Projection '{source}' not found. Re-run projections.")
        if proj.shape[0] != self.dataset.shape[0]:
            raise ValueError(
                f"Projection '{source}' has {proj.shape[0]} rows but dataset has {self.dataset.shape[0]}. "
                "Re-run projections."
            )
        return proj, source

    def show_silhouette_popup(self):
        """
        Silhouette sweep with aligned logic:
        • Method:
            - "Full (slow, no PCA)": silhouette on ALL points (no PCA, no sampling)
            - "Fast — sample only": silhouette on a stratified sample; no PCA
            - "Fast — PCA + sample": randomized PCA then silhouette on a stratified sample
        • Clusterer (for ALL methods): KMeans (exact) or MiniBatchKMeans (fast)
        • Controls: k-range, PCA dims, sample size or %, seed
        • Caching uses all knobs to avoid cross-contamination
        """
        import numpy as np
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
            QDoubleSpinBox, QCheckBox, QDialogButtonBox
        )
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        # --- matrix used for clustering in your app (keep this helper in your class)
        try:
            X, base_name = self._get_clustering_base_matrix()
        except Exception as e:
            try: self.log_output.append(f"Silhouette: {e}")
            except Exception: pass
            return

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 3:
            try: self.log_output.append("Silhouette: need a 2D matrix with at least 3 rows.")
            except Exception: pass
            return

        n, d = X.shape

        # lazy imports (faster app startup)
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA

        # -------- dialog UI --------
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Silhouette sweep — base: {base_name}")
        vbox = QVBoxLayout(dlg)

        # Clusterer row (new)
        cl_row = QHBoxLayout()
        cl_row.addWidget(QLabel("Clusterer:"))
        clusterer = QComboBox()
        clusterer.addItems(["KMeans (exact)", "MiniBatchKMeans (fast)"])
        clusterer.setCurrentIndex(0)  # default to exact for alignment
        cl_row.addWidget(clusterer, 1)
        vbox.addLayout(cl_row)

        # Method row
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        method = QComboBox()
        method.addItems([
            "Full (slow, no PCA)",
            "Fast — sample only",
            "Fast — PCA + sample",
        ])
        method_row.addWidget(method, 1)
        vbox.addLayout(method_row)

        # k-range
        k_row = QHBoxLayout()
        k_row.addWidget(QLabel("k min:")); kmin = QSpinBox(); kmin.setRange(2, max(2, n-1)); kmin.setValue(2); k_row.addWidget(kmin)
        k_row.addSpacing(12)
        k_row.addWidget(QLabel("k max:")); kmax = QSpinBox(); kmax.setRange(3, max(3, n)); kmax.setValue(min(12, max(3, n))); k_row.addWidget(kmax)
        k_row.addStretch(1)
        vbox.addLayout(k_row)

        # PCA dims (only for PCA+sample)
        pca_row = QHBoxLayout()
        pca_row.addWidget(QLabel("PCA dims:"))
        pca_dims = QSpinBox(); pca_dims.setRange(1, max(1, d)); pca_dims.setValue(min(20, d))
        pca_row.addWidget(pca_dims); pca_row.addStretch(1)
        vbox.addLayout(pca_row)

        # Sampling controls
        samp_row = QHBoxLayout()
        total_label = QLabel(f"N total: {n:,}"); samp_row.addWidget(total_label); samp_row.addSpacing(18)
        use_percent = QCheckBox("Use % of data"); samp_row.addWidget(use_percent)
        samp_row.addSpacing(12); samp_row.addWidget(QLabel("Sample size:"))
        sample_n = QSpinBox(); sample_n.setRange(2, n); sample_n.setValue(min(5000, n)); samp_row.addWidget(sample_n)
        samp_row.addSpacing(12); samp_row.addWidget(QLabel("%:"))
        sample_pct = QDoubleSpinBox(); sample_pct.setRange(0.1, 100.0); sample_pct.setSingleStep(1.0); sample_pct.setValue(10.0); samp_row.addWidget(sample_pct)
        samp_row.addStretch(1)
        vbox.addLayout(samp_row)

        # Seed
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Random seed:"))
        seed = QSpinBox(); seed.setRange(0, 10_000_000); seed.setValue(0); seed_row.addWidget(seed)
        seed_row.addStretch(1)
        vbox.addLayout(seed_row)

        info = QLabel("Ready."); info.setStyleSheet("color: gray;")
        vbox.addWidget(info)

        fig = Figure(figsize=(6.2, 3.4), constrained_layout=True)
        canvas = FigureCanvas(fig); ax = fig.add_subplot(111)
        vbox.addWidget(canvas)

        btns = QDialogButtonBox(dlg)
        run_btn   = btns.addButton("Run", QDialogButtonBox.ActionRole)
        usek_btn  = btns.addButton("Use best k", QDialogButtonBox.ActionRole); usek_btn.setEnabled(False)
        close_btn = btns.addButton(QDialogButtonBox.Close)
        vbox.addWidget(btns)

        # -------- helpers --------
        def _toggle_enables():
            m = method.currentIndex()
            pca_dims.setEnabled(m == 2)
            sampling_on = (m in (1, 2))
            use_percent.setEnabled(sampling_on)
            if sampling_on:
                sample_n.setEnabled(not use_percent.isChecked())
                sample_pct.setEnabled(use_percent.isChecked())
            else:
                sample_n.setEnabled(False); sample_pct.setEnabled(False)

        def _eval_n():
            if method.currentIndex() in (1, 2):
                return max(2, int(np.ceil(n * (sample_pct.value() / 100.0)))) if use_percent.isChecked() else max(2, int(sample_n.value()))
            return n

        def _cache_key(kvals, evaln):
            return (
                base_name, n, d,
                method.currentIndex(),
                clusterer.currentIndex(),
                (int(pca_dims.value()) if method.currentIndex() == 2 else 0),
                int(evaln),
                int(seed.value()),
                tuple(kvals),
            )

        def _fit_clusterer(k, Xr, seed_val):
            if clusterer.currentIndex() == 0:
                return KMeans(n_clusters=k, random_state=seed_val, n_init=10, max_iter=300)
            else:
                return MiniBatchKMeans(
                    n_clusters=k, random_state=seed_val,
                    batch_size=min(4096, max(512, Xr.shape[0] // 8)),
                    n_init=10, max_iter=300
                )

        def _stratified_indices(labels, take, rng):
            """Return indices of a stratified sample of size 'take' over labels."""
            nL = labels.size
            if take >= nL: return None  # means: use full
            uniq, cnt = np.unique(labels, return_counts=True)
            # allocate proportionally, keep at least 2 if possible
            out = []
            remaining = take
            for u, c in zip(uniq, cnt):
                want = int(round(take * (c / nL)))
                want = min(c, max(1, want))
                idx_u = np.flatnonzero(labels == u)
                choose = rng.choice(idx_u, size=want, replace=False)
                out.append(choose); remaining -= choose.size
            if remaining > 0:
                pool = np.setdiff1d(np.arange(nL), np.concatenate(out), assume_unique=False)
                extra = rng.choice(pool, size=remaining, replace=False)
                out.append(extra)
            return np.concatenate(out)

        _toggle_enables()
        method.currentIndexChanged.connect(_toggle_enables)
        use_percent.toggled.connect(_toggle_enables)

        # -------- run logic --------
        _best_k = None

        def _run():
            nonlocal _best_k
            lo, hi = int(kmin.value()), int(kmax.value())
            if lo >= hi:
                info.setText("k-min must be < k-max.")
                return
            kvals = list(range(lo, hi + 1))
            evaln = _eval_n()
            seed_val = int(seed.value())

            Xbase = X.astype(np.float32, copy=False)
            m = method.currentIndex()

            # PCA only in mode 2
            if m == 2:
                dcap = int(pca_dims.value())
                dcap = max(1, min(dcap, Xbase.shape[1]))
                p = PCA(n_components=dcap, svd_solver="randomized", random_state=seed_val)
                Xr = p.fit_transform(Xbase).astype(np.float32, copy=False)
            else:
                Xr = Xbase

            # cache
            key = _cache_key(kvals, evaln)
            if not hasattr(self, "_sil_cache"): self._sil_cache = {}
            if key in self._sil_cache:
                kvalsC, scores = self._sil_cache[key]
            else:
                from sklearn.metrics import silhouette_score
                rng = np.random.default_rng(seed_val)
                scores = []
                for k in kvals:
                    # fit on FULL Xr with chosen clusterer (aligns logic across modes)
                    km = _fit_clusterer(k, Xr, seed_val)
                    labels_full = km.fit_predict(Xr)
                    # choose eval slice: full for method 0, stratified subset otherwise
                    if m == 0 or evaln >= Xr.shape[0]:
                        idx = None
                        X_eval = Xr
                        labels_eval = labels_full
                    else:
                        idx = _stratified_indices(labels_full, evaln, rng)
                        X_eval = Xr[idx]
                        labels_eval = labels_full[idx]
                    if np.unique(labels_eval).size < 2:
                        scores.append(np.nan); continue
                    s = silhouette_score(X_eval, labels_eval, metric="euclidean")
                    scores.append(float(s))
                self._sil_cache[key] = (tuple(kvals), list(scores))

            # plot
            ax.clear()
            ax.plot(kvals, scores, marker="o", lw=1.2)
            if np.isfinite(scores).any():
                bi = int(np.nanargmax(scores)); _best_k = int(kvals[bi]); ax.axvline(_best_k, ls="--")
                usek_btn.setEnabled(True)
            else:
                _best_k = None; usek_btn.setEnabled(False)
            ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.grid(True, alpha=0.25)
            canvas.draw()

            used_dims = Xr.shape[1]
            mode_txt = ["Full", "Fast(sample)", "Fast(PCA+sample)"][m]
            cl_txt = ["KMeans", "MiniBatchKMeans"][clusterer.currentIndex()]
            info.setText(
                f"{mode_txt} | {cl_txt} | n={n:,}, d={d} → used dims={used_dims}, eval n={(Xr.shape[0] if m==0 else evaln):,}"
                + (f" | best k={_best_k}" if _best_k is not None else "")
            )

        def _apply_best_k():
            if _best_k is not None and hasattr(self, "cluster_count_spin"):
                self.cluster_count_spin.setValue(int(_best_k))
                dlg.accept()

        run_btn.clicked.connect(_run)
        usek_btn.clicked.connect(_apply_best_k)
        close_btn.clicked.connect(dlg.reject)

        dlg.resize(800, 540)
        dlg.setModal(False)
        dlg.show()

    def show_db_score_popup(self):
        # 1) Use the same base as clustering
        try:
            X, base_name = self._get_clustering_base_matrix()
        except Exception as e:
            self.log_output.append(f"DB Score: {e}")
            return

        n_samples = X.shape[0]
        if n_samples < 3:
            self.log_output.append("DB Score: need at least 3 samples.")
            return

        # Sweep k (keep it quick)
        k_min = 2
        k_max = min(10, max(2, n_samples - 1))
        k_values = list(range(k_min, k_max + 1))

        # 2) Compute DB scores (lower is better)
        scores = []
        for k in k_values:
            try:
                model = KMeans(n_clusters=k, random_state=0)
                labels = model.fit_predict(X)
                # DB needs at least 2 non-empty clusters
                if len(np.unique(labels)) < 2:
                    scores.append(np.nan)
                    continue
                s = davies_bouldin_score(X, labels)
                scores.append(s)
            except Exception as ex:
                self.log_output.append(f"DB(k={k}) failed: {ex}")
                scores.append(np.nan)

        if not np.isfinite(scores).any():
            self.log_output.append("DB Score: no valid scores in range.")
            return

        # 3) Pick best k (min score)
        best_idx = int(np.nanargmin(scores))
        best_k = k_values[best_idx]
        best_score = float(scores[best_idx])

        # 4) Show popup with plot + "Use k" button
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Davies–Bouldin sweep — base: {base_name}")
        dlg.resize(700, 400)
        vbox = QVBoxLayout(dlg)

        fig = Figure(figsize=(6, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(k_values, scores, marker='o')
        ax.axvline(best_k, color='r', linestyle='--', label=f"Best k = {best_k}")
        ax.set_title("Davies–Bouldin Score vs k (lower is better)")
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylabel("DB score")
        ax.legend()
        fig.tight_layout()
        canvas.draw()
        vbox.addWidget(canvas)

        info = QLabel(f"Best k = {best_k} (DB score = {best_score:.3f})")
        vbox.addWidget(info)

        btns = QDialogButtonBox(dlg)
        use_btn = btns.addButton(f"Use k = {best_k}", QDialogButtonBox.ActionRole)
        close_btn = btns.addButton(QDialogButtonBox.Close)
        vbox.addWidget(btns)

        def _apply_k():
            self.cluster_count_spin.setValue(best_k)
            dlg.accept()

        use_btn.clicked.connect(_apply_k)
        close_btn.clicked.connect(dlg.reject)

        dlg.exec()

    def show_cluster_spectra_popup(self, use_mean=False):
        # create/reuse dialog
        dlg = self._ensure_spectra_dialog("Cluster Spectra (log)")
        # set dropdown to desired state (so first render matches)
        if self._spectra_agg_combo is not None:
            self._spectra_agg_combo.setCurrentIndex(1 if use_mean else 0)
        # render according to current dropdown
        self._render_spectra(self._spectra_agg_combo.currentIndex() == 1 if self._spectra_agg_combo else bool(use_mean))

        # show (reuse) dialog safely
        dlg.setModal(True)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
    
    def _ensure_spectra_dialog(self, title):
        if self._spectra_dlg is not None:
            self._spectra_dlg.setWindowTitle(title)
            return self._spectra_dlg

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(1000, 600)
        vbox = QVBoxLayout(dlg)

        # --- Aggregation controls (Sum / Mean)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Aggregation:"))
        agg = QComboBox()
        agg.addItems(["Sum of pixels", "Mean per pixel"])
        agg.setCurrentIndex(1 if self._spectra_use_mean else 0)
        ctrl.addWidget(agg)
        ctrl.addStretch(1)
        vbox.addLayout(ctrl)
        self._spectra_agg_combo = agg

        # Matplotlib area
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        vbox.addWidget(canvas)

        # Buttons row
        btns = QDialogButtonBox(dlg)
        save_png = btns.addButton("Save PNG…", QDialogButtonBox.ActionRole)
        export_btn = btns.addButton("Export CSV…", QDialogButtonBox.ActionRole)
        close_btn = btns.addButton(QDialogButtonBox.Close)
        vbox.addWidget(btns)

        def _save_png():
            base = f"cluster_spectra_{'mean' if self._spectra_use_mean else 'sum'}.png"
            path = self._get_save_filename("Save spectra PNG", base, "PNG Image (*.png)")
            if not path: return
            fig.savefig(path, dpi=300)

        def _save_csv():
            if not self._spectra_current:
                QMessageBox.information(self, "Export", "No spectra to export.")
                return
            import pandas as pd, numpy as np
            k = len(self._spectra_current)
            df = pd.DataFrame({f"cluster_{i}": np.asarray(self._spectra_current[i]) for i in range(k)})
            base = f"cluster_spectra_{'mean' if self._spectra_use_mean else 'sum'}.csv"
            path = self._get_save_filename("Save spectra CSV", base, "CSV Files (*.csv)")
            if not path: return
            df.to_csv(path, index_label="channel")
            self.log_output.append(f"Saved spectra to {path}")

        save_png.clicked.connect(_save_png)
        export_btn.clicked.connect(self._export_spectra_with_options)
        close_btn.clicked.connect(dlg.close)

        # keep refs
        self._spectra_dlg = dlg
        self._spectra_canvas = canvas
        self._spectra_axs = None

        # replot when the dropdown changes
        agg.currentIndexChanged.connect(
            lambda _: self._render_spectra(agg.currentIndex() == 1)
        )

        return dlg

    def _stream_cluster_spectra(self, use_mean=False):
        """
        Return a list of K arrays (each length C) by streaming from the HDF dataset
        recorded at load time. No N×C cache is created.
        """
        src = self.pipeline_config.get("raw_hdf_source")
        if not src:
            self.log_output.append("Cluster Spectra: raw HDF source unknown. Load a dataset first.")
            return None

        hdf_file = src.get("file"); dpath = src.get("dataset_path")
        if not hdf_file or not dpath:
            self.log_output.append("Cluster Spectra: missing HDF file or dataset path.")
            return None

        clustering = self.pipeline_config.get("clustering")
        if not clustering or "labels" not in clustering:
            self.log_output.append("Cluster Spectra: run clustering first.")
            return None
        labels = np.asarray(clustering["labels"]).astype(int)
        k = int(clustering["num_clusters"])

        coords = self.pipeline_config.get("original_coordinates")
        if coords is None:
            QMessageBox.critical(self, "Cluster Spectra", "Missing original (X,Y) coordinates.")
            return None

        import h5py
        with h5py.File(hdf_file, "r") as f:
            dset = f[dpath]
            if dset.ndim == 3:
                H, W, C = dset.shape
            elif dset.ndim == 2:
                a, b = dset.shape
                if a == labels.shape[0]:
                    C = b
                elif b == labels.shape[0]:
                    C = a
                else:
                    QMessageBox.critical(self, "Cluster Spectra",
                                        f"2D dataset shape {dset.shape} doesn't match N={labels.shape[0]} pixels.")
                    return None
            else:
                QMessageBox.critical(self, "Cluster Spectra", f"Unsupported dataset rank: {dset.ndim}")
                return None

            sums = [np.zeros(C, dtype=np.float64) for _ in range(k)]
            counts = np.zeros(k, dtype=np.int64)

            if dset.ndim == 3:
                width  = int(coords["X"].max()) + 1
                height = int(coords["Y"].max()) + 1
                if (height, width) != (H, W):
                    self.log_output.append(
                        f"Warning: coord grid {(height,width)} != data grid {(H,W)}; unmatched pixels ignored."
                    )
                label_grid = -np.ones((H, W), dtype=np.int32)
                XY = coords[["X", "Y"]].values.astype(int)
                for i, (x, y) in enumerate(XY):
                    if 0 <= y < H and 0 <= x < W:
                        label_grid[y, x] = labels[i]

                for y in range(H):
                    row = dset[y, :, :]            # (W, C)
                    lab_row = label_grid[y, :]     # (W,)
                    for c_id in range(k):
                        m = (lab_row == c_id)
                        if m.any():
                            sums[c_id] += row[m].sum(axis=0)
                            counts[c_id] += int(m.sum())

            else:  # 2D
                N = labels.shape[0]
                if dset.shape[0] == N:     # (N, C)
                    step = max(1, min(8192, N // 16))
                    for i0 in range(0, N, step):
                        i1 = min(N, i0 + step)
                        block = dset[i0:i1, :]     # (B, C)
                        lbls  = labels[i0:i1]
                        for c_id in range(k):
                            m = (lbls == c_id)
                            if m.any():
                                sums[c_id] += block[m].sum(axis=0)
                                counts[c_id] += int(m.sum())
                else:                       # (C, N)
                    N = dset.shape[1]
                    step = max(1, min(8192, N // 16))
                    for i0 in range(0, N, step):
                        i1 = min(N, i0 + step)
                        block = dset[:, i0:i1].T   # (B, C)
                        lbls  = labels[i0:i1]
                        for c_id in range(k):
                            m = (lbls == c_id)
                            if m.any():
                                sums[c_id] += block[m].sum(axis=0)
                                counts[c_id] += int(m.sum())

        if use_mean:
            out = []
            for c_id in range(k):
                n = max(1, counts[c_id])
                out.append(sums[c_id] / float(n))
        else:
            out = sums
        return out

    def _render_spectra(self, use_mean: bool):
        """Compute (streamed) cluster spectra and draw into the singleton dialog."""
        clustering = self.pipeline_config.get("clustering")
        if not clustering or "labels" not in clustering:
            self.log_output.append("Cluster Spectra: run clustering first.")
            return
        self._spectra_use_mean = bool(use_mean)
        k = int(clustering["num_clusters"])

        # colors consistent with your cluster cmap
        cmap_obj = (
            getattr(self, "cluster_cmap", None)
            or self.pipeline_config.get("cluster_cmap", None)
        )
        if cmap_obj is not None and hasattr(cmap_obj, "colors"):
            colors = [to_hex(cmap_obj.colors[i % len(cmap_obj.colors)]) for i in range(k)]
        elif hasattr(self, "cluster_rgba") and self.cluster_rgba:
            colors = [to_hex(tuple(self.cluster_rgba[i % len(self.cluster_rgba)][:3])) for i in range(k)]
        else:
            base = mpl.colormaps["tab20"].resampled(k)
            colors = [to_hex(c) for c in base.colors]

        spectra = self._stream_cluster_spectra(use_mean=self._spectra_use_mean)
        if spectra is None:
            return
        self._spectra_current = spectra  # cache what’s on screen

        # draw
        fig = self._spectra_canvas.figure
        fig.clear()
        axs = fig.subplots(k, 1, sharex=True)
        if k == 1:
            axs = [axs]

        title_agg = "Mean" if self._spectra_use_mean else "Sum"
        for i, ax in enumerate(axs):
            y = np.asarray(spectra[i], dtype=np.float64)
            y[y <= 0] = np.nan
            ax.set_yscale('log', base=10)
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1))
            ax.yaxis.set_minor_formatter(NullFormatter())
            pos = y[np.isfinite(y)]
            if pos.size:
                ymin = max(np.nanmin(pos) * 0.7, 1e-12)
                ymax = np.nanmax(pos) * 1.3
                if ymin < ymax:
                    ax.set_ylim(ymin, ymax)

            ax.plot(y, lw=1.4, color=colors[i], solid_capstyle='round')
            ax.grid(True, which='both', alpha=0.25)
            ax.set_ylabel("Mean Counts" if self._spectra_use_mean else "Total Counts")
            ax.legend([f"Cluster {i} ({title_agg})"], loc="upper right", frameon=True)

        axs[-1].set_xlabel("Channel Index")
        fig.suptitle(f"{title_agg} Channel Spectra per Cluster", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        self._spectra_canvas.draw()
        
    def _open_dialog_safely(self, builder_fn, retries=3, delay_ms=300):
        """Call builder_fn() to show a dialog, but only when a screen exists.
        Retries a few times if the display temporarily reports 0 screens."""
        if QGuiApplication.screens():
            builder_fn()
            return
        self.log_output.append("No screens detected; retrying popup shortly...")
        attempts = {"n": 0}

        def _retry():
            if QGuiApplication.screens():
                builder_fn()
            elif attempts["n"] < retries - 1:
                attempts["n"] += 1
                QTimer.singleShot(delay_ms, _retry)
            else:
                self.log_output.append("Giving up opening popup: no screens available.")
        QTimer.singleShot(delay_ms, _retry)

    def _get_save_filename(self, title, default_name, name_filter):
        """Non-native Save dialog (more stable across drivers/RDP)."""
        fd = QFileDialog(self)
        fd.setOption(QFileDialog.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptSave)
        fd.selectFile(default_name)
        fd.setNameFilter(name_filter)
        if fd.exec() == QDialog.Accepted:
            files = fd.selectedFiles()
            return files[0] if files else None
        return None

    def _export_spectra_with_options(self):
        # need current spectra in memory
        if not self._spectra_current:
            QMessageBox.information(self, "Export", "No spectra to export.")
            return

        # ask for options
        dlg = SpectraExportDialog(self, default_mode="mean" if self._spectra_use_mean else "sum")
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.get_opts()

        import pandas as pd
        data = {f"cluster_{i}": np.asarray(self._spectra_current[i], dtype=float)
                for i in range(len(self._spectra_current))}
        df = pd.DataFrame(data)
        df.index.name = "channel"

        if opts["log_safe"]:
            # replace NaNs and <=0 with epsilon
            arr = df.values
            arr[~np.isfinite(arr)] = opts["epsilon"]
            arr[arr <= 0] = opts["epsilon"]
            df.iloc[:, :] = arr

        mode = "mean" if self._spectra_use_mean else "sum"

        if opts["format"] == "csv":
            base = f"cluster_spectra_{mode}.csv"
            path = self._get_save_filename("Save spectra CSV", base, "CSV Files (*.csv)")
            if not path: return
            df.to_csv(path)
            self.log_output.append(f"Saved spectra CSV → {path}")
            return

        # Excel with optional metadata + embedded chart
        base = f"cluster_spectra_{mode}.xlsx"
        path = self._get_save_filename("Save spectra Excel", base, "Excel Workbook (*.xlsx)")
        if not path: return

        # write sheets
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="spectra")
                book = writer.book
                ws = writer.sheets["spectra"]

                # optional metadata
                if opts.get("include_meta", True):
                    meta_rows = []
                    src = self.pipeline_config.get("raw_hdf_source", {})
                    clustering = self.pipeline_config.get("clustering", {})
                    labels = np.asarray(clustering.get("labels", []), dtype=int)
                    k = int(clustering.get("num_clusters", len(self._spectra_current)))
                    counts = [int(np.sum(labels == i)) for i in range(k)] if labels.size else [None]*k
                    meta_rows.append(["Aggregation", mode])
                    meta_rows.append(["File", src.get("file", "")])
                    meta_rows.append(["Dataset path", src.get("dataset_path", "")])
                    meta_rows.append(["Clusters (K)", k])
                    meta_rows.append(["Pixels per cluster", ", ".join(map(str, counts))])
                    meta_rows.append(["Log-safe epsilon", opts["epsilon"] if opts["log_safe"] else "disabled"])
                    meta_df = pd.DataFrame(meta_rows, columns=["Key", "Value"])
                    meta_df.to_excel(writer, sheet_name="metadata", index=False)

                # optional chart
                if opts.get("embed_chart", True):
                    k = len(self._spectra_current)
                    colors = self._get_cluster_hex_colors(k)
                    chart = book.add_chart({"type": "line"})

                    # Excel rows/cols are 1-based in the formula references
                    # 'spectra'!B1 is first column header; data start at row 2
                    n_rows = len(df)
                    for i in range(k):
                        # X range: channel index in col A (row 2 .. n_rows+1)
                        x_range = f"=spectra!$A$2:$A${n_rows+1}"
                        # Y range: cluster_i in column (i+2)
                        col_letter = chr(ord('A') + 1 + i)  # B, C, D, ...
                        y_range = f"=spectra!${col_letter}$2:${col_letter}${n_rows+1}"
                        chart.add_series({
                            "name":       f"=spectra!${col_letter}$1",
                            "categories": x_range,
                            "values":     y_range,
                            "line": {"color": colors[i]},
                        })

                    chart.set_title({"name": f"Cluster spectra ({mode})"})
                    chart.set_x_axis({"name": "channel"})
                    # log y-axis; Excel needs positive values, ensured by epsilon if chosen
                    chart.set_y_axis({"name": "counts", "log_base": 10})

                    # place chart to the right of the table
                    ws.insert_chart("H2", chart)

                writer.close()
            self.log_output.append(f"Saved spectra Excel → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def _current_element_matrix(self, use_all: bool = False):
        """
        Return (df_elements, element_names) using raw intensities from original_dataset,
        aligned with clustering labels. Excludes X/Y.

        use_all=False → use pipeline_config['important_elements'] (filtered)
        use_all=True  → use all element columns in original_dataset (except X,Y)
        """
        if "original_dataset" not in self.pipeline_config:
            raise ValueError("Original dataset not available.")
        df_full = self.pipeline_config["original_dataset"]

        if use_all:
            elems = [c for c in df_full.columns if c not in ("X", "Y")]
        else:
            elems = self.pipeline_config.get(
                "important_elements", [c for c in df_full.columns if c not in ("X", "Y")]
            )

        elems = [e for e in elems if e in df_full.columns and e not in ("X", "Y")]
        if not elems:
            raise ValueError("No element columns to summarize.")
        return df_full[elems].copy(), elems

    def _compute_cluster_concentrations(self, agg="mean", only_visible=False, use_all=False):
        """
        Compute per-cluster aggregation over elements.
        Returns a DataFrame: rows=clusters, cols=elements.
        """
        clustering = self.pipeline_config.get("clustering", {})
        labels = clustering.get("labels", None)
        if labels is None:
            raise ValueError("Run clustering first.")
        labels = np.asarray(labels).ravel()
        k = int(clustering.get("num_clusters", labels.max() + 1))

        df, elems = self._current_element_matrix(use_all=use_all)

        clusters = range(k) if not only_visible else sorted(self.visible_clusters)
        rows, index = [], []
        for c in clusters:
            mask = (labels == c)
            if not mask.any():
                vals = np.full(len(elems), np.nan, dtype=float)
            else:
                sub = df.loc[mask, elems]
                if agg == "sum":
                    vals = np.asarray(sub.sum(axis=0), dtype=float)
                elif agg == "median":
                    vals = np.asarray(sub.median(axis=0), dtype=float)
                else:
                    vals = np.asarray(sub.mean(axis=0), dtype=float)
            rows.append(vals); index.append(f"Cluster {c}")
        return pd.DataFrame(rows, columns=elems, index=index)

    def show_cluster_concentrations_popup(self):
        try:
            _ = self.pipeline_config["clustering"]["labels"]
        except Exception:
            QMessageBox.information(self, "Cluster concentrations", "Run clustering first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Cluster concentrations")
        dlg.setMinimumSize(1100, 750)
        vbox = QVBoxLayout(dlg)

        # --- options row
        opt = QHBoxLayout()
        opt.addWidget(QLabel("Aggregation:"))
        agg_combo = QComboBox(); agg_combo.addItems(["mean", "median", "sum"]); opt.addWidget(agg_combo)

        only_vis = QCheckBox("Only visible clusters"); only_vis.setChecked(False); opt.addWidget(only_vis)
        use_all  = QCheckBox("Use all elements");       use_all.setChecked(False);  opt.addWidget(use_all)

        annotate = QCheckBox("Annotate values"); annotate.setChecked(False); opt.addWidget(annotate)
        ann_style = QComboBox(); ann_style.addItems(["horizontal", "vertical"]); ann_style.setEnabled(False)
        opt.addWidget(ann_style)
        annotate.toggled.connect(ann_style.setEnabled)

        opt.addStretch()
        vbox.addLayout(opt)

        # --- tabs
        tabs = QTabWidget(dlg)

        # Heatmap tab
        hm_widget = QWidget(); hm_layout = QVBoxLayout(hm_widget)
        hm_fig = Figure(figsize=(7.5, 4.8), constrained_layout=True)
        hm_canvas = FigureCanvas(hm_fig)
        hm_ax = hm_fig.add_subplot(111)
        hm_layout.addWidget(hm_canvas)
        tabs.addTab(hm_widget, "Heatmap")

        # Table tab
        table_widget = QWidget(); tbl_layout = QVBoxLayout(table_widget)
        table = QTableWidget(table_widget); table.setAlternatingRowColors(True); table.setSortingEnabled(False)
        tbl_layout.addWidget(table, 1)
        tabs.addTab(table_widget, "Table")

        vbox.addWidget(tabs, 1)

        # ---------- helpers (define BEFORE use) ----------
        import re, numpy as np, pandas as pd
        def _parse_cluster_id(x):
            if isinstance(x, (int, np.integer)):
                return int(x)
            m = re.search(r"\d+", str(x))
            return int(m.group()) if m else None

        def _cluster_row_labels(df_index):
            """Return (labels, ids) where labels look like 'Cluster i (xx.x%)'."""
            pct = self.pipeline_config.get("cluster_pct", {})
            idx_vals = list(df_index)
            cluster_ids = [_parse_cluster_id(x) for x in idx_vals]
            labels = [
                (f"Cluster {cid} ({pct.get(cid, 0.0):.1f}%)" if cid is not None else str(x))
                for x, cid in zip(idx_vals, cluster_ids)
            ]
            return labels, cluster_ids

        def _render_table(df):
            n_rows, n_cols = df.shape
            table.clear()
            table.setRowCount(n_rows)
            table.setColumnCount(n_cols + 1)  # +1 for the "Cluster" label column
            headers = ["Cluster"] + list(df.columns)
            table.setHorizontalHeaderLabels(headers)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

            cmap = getattr(self, "cluster_rgba", None)
            k = len(cmap) if cmap is not None else 0

            for r, idx in enumerate(df.index):
                cid = _parse_cluster_id(idx)
                name_item = QTableWidgetItem(str(idx))
                if cmap is not None and cid is not None and 0 <= cid < k:
                    qc = QColor.fromRgbF(*cmap[cid][:3])
                    name_item.setBackground(qc)
                    name_item.setForeground(QColor("white"))
                table.setItem(r, 0, name_item)

                for j, val in enumerate(df.iloc[r].values, start=1):
                    if val is None or (isinstance(val, float) and not np.isfinite(val)):
                        s = ""
                    else:
                        s = f"{val:.6g}"
                    table.setItem(r, j, QTableWidgetItem(s))

        def _render_heatmap(df):
            nonlocal hm_ax
            hm_fig.clear()
            hm_ax = hm_fig.add_subplot(111)

            if df.empty:
                hm_ax.text(0.5, 0.5, "No data", ha="center", va="center")
                hm_canvas.draw(); return

            im = hm_ax.imshow(df.values, aspect="auto", cmap="viridis")
            hm_ax.set_xticks(range(df.shape[1]))
            hm_ax.set_xticklabels(df.columns, rotation=60, ha="right", fontsize=8)

            labels, cluster_ids = _cluster_row_labels(df.index)
            hm_ax.set_yticks(np.arange(df.shape[0]))
            hm_ax.set_yticklabels(labels, fontsize=9)

            # color ticks by cluster id (robust)
            if hasattr(self, "cluster_rgba"):
                k = len(self.cluster_rgba)
                for t, cid in zip(hm_ax.get_yticklabels(), cluster_ids):
                    if cid is not None and 0 <= cid < k:
                        r, g, b, a = self.cluster_rgba[cid]
                        t.set_color((r, g, b))

            hm_ax.set_title(f"Cluster concentrations ({agg_combo.currentText()})")
            cbar = hm_fig.colorbar(im, ax=hm_ax, fraction=0.045, pad=0.02)
            cbar.set_label("value", rotation=90)

            if annotate.isChecked():
                rotate = 90 if ann_style.currentText() == "vertical" else 0
                norm = im.norm
                for rr in range(df.shape[0]):
                    for cc in range(df.shape[1]):
                        val = df.iat[rr, cc]
                        if np.isfinite(val):
                            color = "white" if norm(val) > 0.6 else "black"
                            hm_ax.text(cc, rr, f"{val:.2g}", ha="center", va="center",
                                    fontsize=7, rotation=rotate, color=color)

            hm_canvas.draw()

        def render():
            try:
                df = self._compute_cluster_concentrations(
                    agg=agg_combo.currentText(),
                    only_visible=only_vis.isChecked(),
                    use_all=use_all.isChecked()
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e)); return

            dlg._conc_df = df  # raw

            # Labeled copy for table & CSV (prefix a numeric % column too)
            labels, ids = _cluster_row_labels(df.index)
            df_lbl = df.copy()
            df_lbl.index = labels
            pct_map = self.pipeline_config.get("cluster_pct", {})
            df_lbl.insert(0, "Percent", [pct_map.get(cid, 0.0) if cid is not None else np.nan for cid in ids])
            dlg._conc_df_labeled = df_lbl

            _render_heatmap(df)
            _render_table(df_lbl)

        # initial + wires
        render()
        agg_combo.currentIndexChanged.connect(render)
        only_vis.stateChanged.connect(lambda *_: render())
        use_all.stateChanged.connect(lambda *_: render())
        annotate.stateChanged.connect(lambda *_: _render_heatmap(getattr(dlg, "_conc_df", pd.DataFrame())))
        ann_style.currentIndexChanged.connect(lambda *_: _render_heatmap(getattr(dlg, "_conc_df", pd.DataFrame())))

        # --- exports
        btns = QDialogButtonBox(dlg)
        export_csv = btns.addButton("Export CSV…", QDialogButtonBox.ActionRole)
        export_png = btns.addButton("Save Heatmap PNG…", QDialogButtonBox.ActionRole)
        close_btn  = btns.addButton(QDialogButtonBox.Close)
        vbox.addWidget(btns)

        def save_csv():
            df = getattr(dlg, "_conc_df_labeled", None)  # labeled df (with Percent col)
            if df is None: return
            base = f"cluster_concentrations_{agg_combo.currentText()}"
            if use_all.isChecked(): base += "_ALL"
            if only_vis.isChecked(): base += "_VISIBLE"
            path = self._get_save_filename("Save concentrations CSV", base + ".csv", "CSV Files (*.csv)")
            if not path: return
            try:
                df.to_csv(path)
                self.log_output.append(f"Saved concentrations CSV → {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export error", str(e))

        def save_heatmap():
            df = getattr(dlg, "_conc_df", None)
            if df is None or df.empty: return
            base = f"cluster_concentrations_{agg_combo.currentText()}"
            if use_all.isChecked(): base += "_ALL"
            if only_vis.isChecked(): base += "_VISIBLE"
            path = self._get_save_filename("Save heatmap PNG", base + ".png", "PNG Image (*.png)")
            if not path: return

            fig = Figure(figsize=(max(6, df.shape[1]*0.28), max(3.5, df.shape[0]*0.55)), constrained_layout=True)
            ax = fig.add_subplot(111)
            im = ax.imshow(df.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(df.shape[1]))
            ax.set_xticklabels(df.columns, rotation=60, ha="right", fontsize=8)

            labels, cluster_ids = _cluster_row_labels(df.index)
            ax.set_yticks(range(df.shape[0]))
            ax.set_yticklabels(labels, fontsize=9)

            if hasattr(self, "cluster_rgba"):
                k = len(self.cluster_rgba)
                for t, cid in zip(ax.get_yticklabels(), cluster_ids):
                    if cid is not None and 0 <= cid < k:
                        r, g, b, a = self.cluster_rgba[cid]
                        t.set_color((r, g, b))

            ax.set_title(f"Cluster concentrations ({agg_combo.currentText()})")
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cbar.set_label("value", rotation=90)

            if annotate.isChecked():
                rotate = 90 if ann_style.currentText() == "vertical" else 0
                norm = im.norm
                for rr in range(df.shape[0]):
                    for cc in range(df.shape[1]):
                        val = df.iat[rr, cc]
                        if np.isfinite(val):
                            color = "white" if norm(val) > 0.6 else "black"
                            ax.text(cc, rr, f"{val:.2g}", ha="center", va="center",
                                    fontsize=7, rotation=rotate, color=color)

            try:
                fig.savefig(path, dpi=200)
                self.log_output.append(f"Saved concentrations heatmap → {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export error", str(e))

        export_csv.clicked.connect(save_csv)
        export_png.clicked.connect(save_heatmap)
        close_btn.clicked.connect(dlg.close)

        dlg.exec()

    def _canvas_to_qimage(self, canvas, scale: float = 1.0, tight: bool = False) -> QImage:
        """Render a Matplotlib FigureCanvas to a QImage via savefig bytes (Qt-safe, no QPixmap)."""
        fig = canvas.figure

        # Make sure the renderer exists
        try:
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
        except Exception:
            w = h = 0

        # If renderer is 0×0 (startup/minimized), size the figure from the widget (or a sane default)
        if w <= 0 or h <= 0:
            sz = canvas.size() if hasattr(canvas, "size") else None
            widget_w = sz.width() if sz and sz.isValid() else 800
            widget_h = sz.height() if sz and sz.isValid() else 600
            dpi = max(72, int(fig.dpi))
            fig.set_size_inches(widget_w / dpi, widget_h / dpi, forward=True)
            try:
                fig.canvas.draw()
            except Exception:
                pass

        # Export via PNG bytes (scale through DPI to avoid post-resize blur)
        import io
        buf = io.BytesIO()
        base_dpi = max(96, int(fig.dpi))          # was 72
        eff = max(1.0, float(scale))              # never downscale via DPI
        fig.savefig(
            buf, format="png",
            dpi=int(base_dpi * eff),              # DPI × scale = more pixels
            facecolor="white",
            bbox_inches="tight" if tight else None,
        )
        qimg = QImage.fromData(buf.getvalue())
        return qimg if not qimg.isNull() else QImage()

    def _compose_side_by_side(self, left: QImage, right: QImage, gutter_px: int = 16) -> QImage:
        """Return a white background composite of two QImages laid out horizontally."""
        if left.isNull() or right.isNull():
            return QImage()

        gutter = max(0, int(gutter_px))
        W = left.width() + gutter + right.width()
        H = max(left.height(), right.height())

        out = QImage(W, H, QImage.Format_ARGB32)
        out.fill(Qt.white)

        p = QPainter(out)
        p.drawImage(0, 0, left)
        p.drawImage(left.width() + gutter, 0, right)
        p.end()
        return out

    def save_canvases_png(self, scale: float = 3.0, tight: bool = False, gutter_px: int = 16):
        """Export left & right canvases side-by-side as PNG (pure QImage path; robust everywhere)."""
        try:
            QGuiApplication.processEvents()

            img_left  = self._canvas_to_qimage(self.left_canvas,  scale=scale, tight=tight)
            img_right = self._canvas_to_qimage(self.right_canvas, scale=scale, tight=tight)

            # Helpful diagnostics if something is null
            if img_left.isNull() or img_right.isNull():
                # Log renderer + widget sizes to pinpoint the culprit
                def _sizes(cv):
                    try:
                        cv.figure.canvas.draw()
                        rw, rh = cv.figure.canvas.get_width_height()
                    except Exception:
                        rw = rh = -1
                    sz = cv.size() if hasattr(cv, "size") else None
                    ww = sz.width() if sz and sz.isValid() else -1
                    wh = sz.height() if sz and sz.isValid() else -1
                    return f"renderer={rw}x{rh}, widget={ww}x{wh}"
                self.log_output.append(
                    f"Export aborted: a canvas returned a null image even after redraw. "
                    f"L[{_sizes(self.left_canvas)}] R[{_sizes(self.right_canvas)}]"
                )
                return

            composite = self._compose_side_by_side(img_left, img_right, gutter_px=gutter_px)
            if composite.isNull():
                self.log_output.append("Export aborted: failed to compose images.")
                return

            # Save dialog
            fd = QFileDialog(self)
            fd.setOption(QFileDialog.DontUseNativeDialog, True)
            fd.setAcceptMode(QFileDialog.AcceptSave)
            fd.setNameFilter("PNG Images (*.png)")
            fd.selectFile("canvases.png")
            if fd.exec() != QFileDialog.Accepted:
                return
            path = fd.selectedFiles()[0]
            if not path.lower().endswith(".png"):
                path += ".png"

            if composite.save(path, "PNG"):
                self.log_output.append(f"Saved canvases PNG to: {path}")
            else:
                self.log_output.append("Failed to save PNG (write error).")
        except Exception as e:
            self.log_output.append(f"Save canvases PNG failed: {e}")

    def load_dataset_and_save_raw_qt(self):
        df = elemental_conversion_qt(parent=self, save_raw=True)
        if df is None:
            self.log_output.append("Dataset loading canceled or failed.")
            return
        element_cols = [c for c in df.columns if c not in ("X","Y")]
        self.pipeline_config["original_dataset"] = df.copy()
        self.pipeline_config["original_coordinates"] = df[["X","Y"]].copy()
        self.pipeline_config["important_elements"] = element_cols
        self.dataset = df[element_cols].copy()
        self.log_output.append("Dataset loaded, processed, and raw images.csv saved (if you chose a path).")
        self.left_canvas.show_dummy("Loaded elemental map")

    def import_raw_images_csv_qt(self):
        from functions_refactored import import_images_csv_processed_qt
        df = import_images_csv_processed_qt(parent=self)
        if df is None:
            self.log_output.append("Import canceled or failed.")
            return
        element_cols = [c for c in df.columns if c not in ("X","Y")]
        self.pipeline_config["original_dataset"] = df.copy()
        self.pipeline_config["original_coordinates"] = df[["X","Y"]].copy()
        self.pipeline_config["important_elements"] = element_cols
        self.dataset = df[element_cols].copy()
        self.log_output.append("Imported raw images.csv and processed it exactly like HDF load.")
        self.left_canvas.show_dummy("Imported elemental map")

    def open_spectra_roi_dialog(self):
        # Reuse an open dialog if present
        dlg = getattr(self, "_spectra_roi_dialog", None)
        if dlg and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return

        # Create new (modeless) dialog and keep a handle
        self._spectra_roi_dialog = SpectraRoiDialog(self)
        self._spectra_roi_dialog.setModal(False)

        # Push current colors so the dialog always matches the editor
        try:
            self._spectra_roi_dialog.set_cluster_colors(
                self.pipeline_config.get("cluster_rgba", self.cluster_rgba)
            )
        except Exception:
            pass

        # When the dialog is destroyed, drop the handle so we can open it again
        self._spectra_roi_dialog.destroyed.connect(
            lambda *_: setattr(self, "_spectra_roi_dialog", None)
        )

        self._spectra_roi_dialog.show()
            
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)

    QCoreApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)

    app = QApplication(sys.argv)
    window = XRFGui()
    window.show()
    sys.exit(app.exec())
