import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QCheckBox, QFileDialog,
    QSplitter, QTextEdit, QGroupBox, QFormLayout,
    QScrollArea, QSpinBox, QDoubleSpinBox, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functions_refactored import *
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
import umap
import numpy as np


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
        self.figure1 = Figure(figsize=(5, 5))
        self.canvas1 = FigureCanvas(self.figure1)
        self.ax1 = self.figure1.add_subplot(111)

        self.figure2 = Figure(figsize=(5, 5))
        self.canvas2 = FigureCanvas(self.figure2)
        self.ax2 = self.figure2.add_subplot(111)
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- Top Button Row ---
        top_buttons = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset_qt)
        self.run_pipeline_button = QPushButton("Run Full Pipeline")
        self.preview_button = QPushButton("Preview Dataset")
        self.preview_button.clicked.connect(self.show_dataset_preview)
        top_buttons.addWidget(self.load_button)
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

        preview = self.dataset.head(50).to_string() if hasattr(self.dataset, "head") else str(self.dataset)
        
        dialog = DataPreviewDialog("Dataset Preview", preview, parent=self)
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

        # Shared components input
        shared_layout = QHBoxLayout()
        self.proj_components = QSpinBox()
        self.proj_components.setMinimum(1)
        self.proj_components.setValue(2)
        shared_layout.addWidget(QLabel("Components:"))
        shared_layout.addWidget(self.proj_components)
        layout.addLayout(shared_layout)

        self.pca_checkbox = QCheckBox("Enable PCA")
        self.nmf_checkbox = QCheckBox("Enable NMF")
        self.nmf_max_iter = QSpinBox()
        self.nmf_max_iter.setRange(10, 1000)
        self.nmf_max_iter.setValue(200)
        self.nmf_iter_enabled = QCheckBox("Set max_iter")

        self.tsne_checkbox = QCheckBox("Enable t-SNE")
        self.tsne_perplexity = QDoubleSpinBox()
        self.tsne_perplexity.setRange(5.0, 100.0)
        self.tsne_perplexity.setValue(30.0)
        self.tsne_perplexity.setSingleStep(1.0)

        self.umap_checkbox = QCheckBox("Enable UMAP")
        self.umap_neighbors = QSpinBox()
        self.umap_neighbors.setRange(2, 200)
        self.umap_neighbors.setValue(15)
        self.umap_metric = QLineEdit("euclidean")

        layout.addWidget(self.pca_checkbox)

        layout.addWidget(self.nmf_checkbox)
        nmf_row = QHBoxLayout()
        nmf_row.addWidget(self.nmf_iter_enabled)
        nmf_row.addWidget(QLabel("max_iter:"))
        nmf_row.addWidget(self.nmf_max_iter)
        layout.addLayout(nmf_row)

        layout.addWidget(self.tsne_checkbox)
        tsne_row = QHBoxLayout()
        tsne_row.addWidget(QLabel("Perplexity:"))
        tsne_row.addWidget(self.tsne_perplexity)
        layout.addLayout(tsne_row)

        layout.addWidget(self.umap_checkbox)
        umap_row1 = QHBoxLayout()
        umap_row1.addWidget(QLabel("Neighbors:"))
        umap_row1.addWidget(self.umap_neighbors)
        layout.addLayout(umap_row1)

        umap_row2 = QHBoxLayout()
        umap_row2.addWidget(QLabel("Metric:"))
        umap_row2.addWidget(self.umap_metric)
        layout.addLayout(umap_row2)

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
        self.dbscore_button = QPushButton("DB Score")
        score_row.addWidget(self.silhouette_button)
        score_row.addWidget(self.dbscore_button)
        layout.addLayout(score_row)

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

        # Use threshold from spinbox or prompt via popup
        threshold = self.moran_threshold.value()
        if threshold <= 0.0:
            threshold = None  # triggers popup prompt

        # Drop X/Y if they exist in dataset
        feature_df = self.dataset.drop(columns=["X", "Y"], errors="ignore")

        # Run the refactored filter
        try:
            moran_df = moran_filter(feature_df, coordinates_df=coords, threshold=threshold, parent=self)
        except Exception as e:
            self.log_output.append(f"Moran's I computation failed: {str(e)}")
            return

        important_elements = moran_df.index.tolist()

        if not important_elements:
            self.log_output.append("No elements passed the Moran's I threshold.")
            return

        self.pipeline_config['important_elements'] = important_elements

        # Reduce dataset to only those elements (XY excluded — toggle added by Add XY button)
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

        if dialog.exec():
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
        for name in ["PCA", "NMF", "t-SNE", "UMAP"]:
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
                umap_model = umap.UMAP(n_components=n_components, n_neighbors=neighbors, n_jobs=-1)
                projections["UMAP"] = umap_model.fit_transform(X)
                self.log_output.append(f"UMAP projection done ({n_components} components, {neighbors} neighbors).")

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

            # Log summary
            unique, counts = np.unique(labels, return_counts=True)
            log_lines = [f"KMeans clustering complete using base: {used_base}"]
            for u, c in zip(unique, counts):
                log_lines.append(f"Cluster {u}: {c} samples")

            self.log_output.append("\n".join(log_lines))
            self.right_canvas.show_dummy("KMeans clustering complete")
            self.mark_button_done(self.run_clustering_button)

            self.visualize_clustering_results()

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
        im = ax.imshow(image, cmap="tab20", interpolation="nearest")
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
            scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab20", s=2)
            ax.set_title(name)
            ax.axis("off")

        # Hide extra subplots
        for j in range(len(keys), len(axes)):
            axes[j].axis("off")

        self.right_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XRFGui()
    window.show()
    sys.exit(app.exec())
