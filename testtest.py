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

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- Top Button Row ---
        top_buttons = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.run_pipeline_button = QPushButton("Run Full Pipeline")
        top_buttons.addWidget(self.load_button)
        top_buttons.addWidget(self.run_pipeline_button)
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

    def mark_button_done(self, button):
        palette = button.palette()
        palette.setColor(QPalette.Button, QColor(180, 255, 180))
        button.setAutoFillBackground(True)
        button.setPalette(palette)
        button.update()

    def create_filtering_section(self):
        group = QGroupBox("1. Filtering")
        layout = QVBoxLayout()

        # Moran's Filter
        moran_row = QHBoxLayout()
        self.moran_threshold = QDoubleSpinBox()
        self.moran_threshold.setRange(0.0, 1.0)
        self.moran_threshold.setValue(0.3)
        self.moran_button = QPushButton("Run Moran's Filter")
        self.moran_button.clicked.connect(self.run_moran_filter)
        moran_row.addWidget(self.moran_button)
        moran_row.addWidget(QLabel("Threshold:"))
        moran_row.addWidget(self.moran_threshold)
        layout.addLayout(moran_row)

        # Foreground Clustering
        foreground_row = QHBoxLayout()
        self.foreground_clusters = QSpinBox()
        self.foreground_clusters.setRange(2, 50)
        self.foreground_clusters.setValue(4)
        self.foreground_button = QPushButton("Run Foreground Clustering")
        self.foreground_button.clicked.connect(self.run_foreground_clustering)
        foreground_row.addWidget(self.foreground_button)
        foreground_row.addWidget(QLabel("Clusters:"))
        foreground_row.addWidget(self.foreground_clusters)
        layout.addLayout(foreground_row)

        self.foreground_mean = QCheckBox("Use Mean")
        self.foreground_std = QCheckBox("Use Std")
        layout.addWidget(self.foreground_mean)
        layout.addWidget(self.foreground_std)

        # Element Exclusion
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
        self.scale_button.clicked.connect(self.run_scaling)
        scale_row.addWidget(self.scale_button)
        scale_row.addWidget(QLabel("Method:"))
        scale_row.addWidget(self.scale_combo)
        layout.addLayout(scale_row)

        # Create Augmented Dataset
        augment_row = QHBoxLayout()
        self.augment_button = QPushButton("Create Augmented Dataset")
        self.augment_button.clicked.connect(self.run_augment_dataset)
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

        group.setLayout(layout)
        return group

    def create_clustering_section(self):
        group = QGroupBox("4. Clustering")
        layout = QFormLayout()

        self.cluster_checkbox = QCheckBox("Run KMeans Clustering")
        self.k_input = QSpinBox()
        self.k_input.setRange(2, 20)
        self.k_input.setValue(4)
        layout.addRow(self.cluster_checkbox, QLabel("K:"))
        layout.addRow("", self.k_input)

        group.setLayout(layout)
        return group

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)")
        if file_path:
            self.log_output.append(f"Loaded dataset: {file_path}")
            self.dataset = f"DummyData({file_path})"

    def run_moran_filter(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded. Please load a dataset first.")
            return
        threshold = self.moran_threshold.value()
        self.dataset = f"MoranFiltered({self.dataset}, threshold={threshold})"
        self.log_output.append(f"Applied Moran's Filter with threshold {threshold}")
        self.left_canvas.show_dummy("Moran filtered image")
        self.mark_button_done(self.moran_button)

    def run_foreground_clustering(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded. Please load a dataset first.")
            return
        k = self.foreground_clusters.value()
        use_mean = self.foreground_mean.isChecked()
        use_std = self.foreground_std.isChecked()
        self.dataset = f"ForegroundClustered({self.dataset}, k={k}, mean={use_mean}, std={use_std})"
        self.log_output.append(f"Applied Foreground Clustering (k={k}, mean={use_mean}, std={use_std})")
        self.left_canvas.show_dummy("Foreground clustering result")
        self.mark_button_done(self.foreground_button)

    def run_element_exclusion(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded. Please load a dataset first.")
            return
        self.dataset = f"ElementExcluded({self.dataset})"
        self.log_output.append("Manually excluded elements from dataset")
        self.left_canvas.show_dummy("Element exclusion result")
        self.mark_button_done(self.element_button)

    def run_add_xy(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded. Please load a dataset first.")
            return
        self.dataset = f"XYAdded({self.dataset})"
        self.log_output.append("Added XY coordinates to dataset")
        self.mark_button_done(self.xy_button)

    def run_scaling(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded. Please load a dataset first.")
            return
        method = self.scale_combo.currentText()
        self.dataset = f"Scaled({self.dataset}, method={method})"
        self.log_output.append(f"Applied {method} scaling to dataset")
        self.mark_button_done(self.scale_button)

    def run_augment_dataset(self):
        if self.dataset is None:
            self.log_output.append("No dataset loaded. Please load a dataset first.")
            return
        element_scaler = self.augment_element_scale.currentText()
        xy_scaler = self.augment_xy_scale.currentText()
        weight_e = self.weight_elements.value()
        weight_xy = self.weight_xy.value()
        self.dataset = f"Augmented({self.dataset}, el_scale={element_scaler}, xy_scale={xy_scaler}, w_el={weight_e}, w_xy={weight_xy})"
        self.log_output.append(f"Created augmented dataset (element scale={element_scaler}, xy scale={xy_scaler}, w_el={weight_e}, w_xy={weight_xy})")
        self.mark_button_done(self.augment_button)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XRFGui()
    window.show()
    sys.exit(app.exec())
