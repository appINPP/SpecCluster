# functions_pyside.py
import io
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from PyMca5.PyMcaIO.OutputBuffer import OutputBuffer
from PyMca5.PyMcaPhysics.xrf.FastXRFLinearFit import FastXRFLinearFit
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, 
    QDialogButtonBox, QFileDialog, QInputDialog, QMessageBox,
    QCheckBox, QScrollArea, QWidget, QGridLayout,
    QTreeWidget, QTreeWidgetItem, QHeaderView, 
    QHBoxLayout, QAbstractItemView, QTableWidget,
    QTableWidgetItem, QComboBox, QDoubleSpinBox,
    QGraphicsView, QGraphicsScene,QToolBar, QTabWidget,
    QFormLayout, QSpinBox, QSizePolicy
    )
from PySide6.QtGui import QPixmap, QImage, QPainter, QFont, QGuiApplication, QAction
from PySide6.QtCore import Qt, Signal, QSignalBlocker, QTimer,QEvent, QPointF, QSize, QCoreApplication
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from libpysal.weights import lat2W
from esda.moran import Moran
from sklearn.cluster import KMeans
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib.backend_bases import MouseButton
import time

def _process_images_df(df_raw):
    # identical to what you do after reading IMAGES/images.csv
    df = df_raw.loc[:, ~df_raw.columns.astype(str).str.contains("1")].copy()
    df.rename(columns={df.columns[0]: "Y", df.columns[1]: "X"}, inplace=True)
    cols = df.columns.tolist()
    return df[["X", "Y"] + cols[2:]]

def import_images_csv_processed_qt(parent=None):
    path, _ = QFileDialog.getOpenFileName(parent, "Import raw images.csv", "", "CSV Files (*.csv)")
    if not path:
        return None
    # try semicolon first (what PyMca wrote), then comma, then sniff
    for sep in [";", ",", None]:
        try:
            if sep is None:
                df_raw = pd.read_csv(path, sep=None, engine="python")
            else:
                df_raw = pd.read_csv(path, sep=sep)
            break
        except Exception:
            df_raw = None
    if df_raw is None:
        QMessageBox.critical(parent, "Import error", "Could not read images.csv (tried ; , and auto-detect).")
        return None
    try:
        return _process_images_df(df_raw)
    except Exception as e:
        QMessageBox.critical(parent, "Import error", f"CSV format not as expected: {e}")
        return None
    
def elemental_conversion_qt(parent=None, save_raw: bool = False):
    try:
        # --- file picking (unchanged) ---
        hdf_file, _ = QFileDialog.getOpenFileName(parent, "Select HDF file", "", "HDF5 Files (*.h5 *.hdf)")
        if not hdf_file:
            return None

        cfg_file, _ = QFileDialog.getOpenFileName(parent, "Select .cfg file", "", "CFG Files (*.cfg)")
        if not cfg_file:
            return None

        picker = DatasetPickerDialog(hdf_file, parent=parent)
        if picker.exec() != QDialog.Accepted:
            return None
        dataset_path = picker.selected_path()
        if not dataset_path:
            return None

        # --- temp & run (unchanged) ---
        import tempfile, shutil, os
        temp_dir = tempfile.mkdtemp(prefix="pymca_output_")

        output = OutputBuffer(outputDir=temp_dir, csv=True, overwrite=True)
        output.saveDataDiagnostics = False
        output.saveFOM = False
        output.saveResiduals = False
        output.saveFit = False
        output.saveImages = False
        output.saveData = True

        with h5py.File(hdf_file, 'r') as f:
            data = np.array(f[dataset_path])
            if parent is not None:
                arr = data.reshape(-1, data.shape[-1])
                parent.pipeline_config["raw_spectra"] = arr         # (N_pixels, n_channels)
                # Try to discover an energy axis; otherwise we’ll plot by channel index
                eng = None
                try:
                    ds = f[dataset_path]
                    for key in ("energy", "energies", "xrf_energies"):
                        if key in ds.attrs:
                            e = np.asarray(ds.attrs[key]).astype(float)
                            # guess units (eV vs keV)
                            eng = e / 1000.0 if np.nanmax(e) > 5000 else e
                            break
                except Exception:
                    pass

        parent.pipeline_config["energy_keV"] = eng
        fast_fit = FastXRFLinearFit()
        fast_fit.setFitConfigurationFile(cfg_file)
        fast_fit.fitMultipleSpectra(y=data, weight=0, outbuffer=output)

        # --- paths to raw CSV & read once ---
        result_path = os.path.join(temp_dir, "IMAGES", "images.csv")
        df_raw = pd.read_csv(result_path, sep=";")

        # --- optionally let user save the RAW file NOW (no caching) ---
        if save_raw and parent is not None:
            to_path, _ = QFileDialog.getSaveFileName(parent, "Save raw images.csv", "images.csv", "CSV Files (*.csv)")
            if to_path:
                try:
                    shutil.copyfile(result_path, to_path)
                    parent.pipeline_config["last_saved_raw_images_csv"] = to_path
                except Exception as e:
                    QMessageBox.warning(parent, "Save raw failed", str(e))

        # --- process once via helper ---
        df_elemental = _process_images_df(df_raw)

        # --- cleanup temp ---
        shutil.rmtree(temp_dir, ignore_errors=True)

        # --- small provenance note (unchanged) ---
        if parent is not None:
            parent.pipeline_config.setdefault("raw_hdf_source", {})
            parent.pipeline_config["raw_hdf_source"]["file"] = hdf_file
            parent.pipeline_config["raw_hdf_source"]["dataset_path"] = dataset_path

        return df_elemental

    except Exception as e:
        QMessageBox.critical(parent, "Conversion Error", f"Error during conversion: {str(e)}")
        return None

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

# replace the whole DataPreviewDialog with this

class DataPreviewDialog(QDialog):
    def __init__(self, title, df_preview, full_df=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 600)
        self.df_preview = df_preview
        self.df_full = full_df if full_df is not None else df_preview

        layout = QVBoxLayout(self)

        label = QLabel("Preview:")
        layout.addWidget(label)

        # Table view (no sorting)
        table = QTableWidget(self.df_preview.shape[0], self.df_preview.shape[1], self)
        table.setAlternatingRowColors(True)
        table.setHorizontalHeaderLabels([str(c) for c in self.df_preview.columns])
        table.verticalHeader().setVisible(False)  # keep off; preserves index order anyway
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setSortingEnabled(False)  # <- critical: NO sorting

        # Fill cells (fast, read-only)
        for i in range(self.df_preview.shape[0]):
            for j in range(self.df_preview.shape[1]):
                item = QTableWidgetItem(str(self.df_preview.iat[i, j]))
                # keep read-only (default flags)
                table.setItem(i, j, item)

        layout.addWidget(table)

        # Buttons: Export CSV… + Close
        buttons = QDialogButtonBox(self)
        export_btn = buttons.addButton("Export CSV…", QDialogButtonBox.ActionRole)
        close_btn = buttons.addButton(QDialogButtonBox.Close)
        export_btn.clicked.connect(self._export_csv)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)

    def _export_csv(self):
        fd = QFileDialog(self)
        fd.setOption(QFileDialog.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptSave)
        fd.selectFile("dataset.csv")
        fd.setNameFilter("CSV Files (*.csv)")
        if fd.exec() != QDialog.Accepted:
            return
        path = fd.selectedFiles()[0]
        if not path:
            return
        try:
            # Export the FULL dataset, not just the preview
            self.df_full.to_csv(path, index=False)
            QMessageBox.information(self, "Export complete", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

class DatasetPickerDialog(QDialog):
    def __init__(self, hdf_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select HDF5 Dataset")
        self.setMinimumSize(800, 500)
        self._selected = None
        self._hdf_path = hdf_path

        main = QVBoxLayout(self)

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Type to filter by path...")
        filter_row.addWidget(self.filter_edit)
        main.addLayout(filter_row)

        # Tree with columns
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Path", "Shape", "Dtype", "Compression", "Size"])
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setAlternatingRowColors(True)
        main.addWidget(self.tree)

        # Selected path display
        self.selected_label = QLabel("Selected: —")
        main.addWidget(self.selected_label)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setEnabled(False)
        main.addWidget(btns)

        # Wire signals
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.tree.itemSelectionChanged.connect(
            lambda: self._on_select(btns.button(QDialogButtonBox.Ok))
        )
        self.tree.itemDoubleClicked.connect(lambda *_: self.accept())
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # Populate
        self._load_datasets()

    def _load_datasets(self):
        import h5py, numpy as np
        self.tree.clear()
        with h5py.File(self._hdf_path, "r") as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    path = "/" + name if not name.startswith("/") else name
                    shape = tuple(obj.shape)
                    dtype = str(obj.dtype)
                    comp = str(obj.compression) if obj.compression else "-"
                    try:
                        size = int(np.prod(shape))
                    except Exception:
                        size = "-"
                    item = QTreeWidgetItem([
                        path,
                        str(shape),
                        dtype,
                        comp,
                        str(size)
                    ])
                    # store path
                    item.setData(0, Qt.UserRole, path)
                    self.tree.addTopLevelItem(item)
            f.visititems(visitor)
        self.tree.sortItems(0, Qt.AscendingOrder)

    def _apply_filter(self, text):
        # simple contains filter on path column
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            path = item.text(0)
            item.setHidden(text.lower() not in path.lower())

    def _on_select(self, ok_button):
        items = self.tree.selectedItems()
        if items:
            self._selected = items[0].data(0, Qt.UserRole)
            self.selected_label.setText(f"Selected: {self._selected}")
            ok_button.setEnabled(True)
        else:
            self._selected = None
            self.selected_label.setText("Selected: —")
            ok_button.setEnabled(False)

    def selected_path(self):
        return self._selected

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

class ElementImageSelectionDialog(QDialog):
    """
    Element gallery with:
    - Search / Show enabled only
    - All / None / Invert (fast)
    - Settings… (colormap, scaling incl Global percentile, Log10 + ε, color scale mode)
    - Apply (emits selection without closing) + OK/Cancel
    """
    applied = Signal(list)

    def __init__(self, element_list, checked_elements, df, width, height, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Elements")
        self.setMinimumSize(1000, 700)
        self._df = df
        self._W, self._H = int(width), int(height)
        self._elements = list(element_list)

        # Caches/state
        self._thumb_cache = {}            # (el, settings_id) -> QPixmap
        self._cbar_base_cache = {}        # cmap -> base 1x256 pixmap
        self._range_dirty = True
        self._rendered = set()
        self._render_queue = []

        # Display settings
        self._cmap = "inferno"
        self._norm_mode = "Per image"     # Per image | Global min/max | Global percentile
        self._pct_lo, self._pct_hi = 2.0, 98.0
        self._scale_mode = "Linear"       # Linear | Log10
        self._log_eps = 1e-12
        self._cbar_mode = "Zoom only"     # Off | Zoom only | Thumbnails + zoom

        # --- Header ---
        top = QHBoxLayout()
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText("Search (e.g., Fe|Cu|Ni)")
        self.only_enabled_box = QCheckBox("Show enabled only")
        btn_all = QPushButton("All"); btn_none = QPushButton("None"); btn_inv = QPushButton("Invert")
        self.settings_btn = QPushButton("Settings…")

        self.save_grid_btn = QPushButton("Save grid…")
        top.addSpacing(8)
        top.addWidget(self.save_grid_btn)

        # Debounced search
        self._filter_timer = QTimer(self); self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self._apply_filter_fast)
        self.search_edit.textChanged.connect(lambda: self._filter_timer.start(60))

        top.addWidget(QLabel("Search:")); top.addWidget(self.search_edit, 2)
        top.addSpacing(12); top.addWidget(self.only_enabled_box); top.addStretch(1)
        top.addWidget(btn_all); top.addWidget(btn_none); top.addWidget(btn_inv)
        top.addSpacing(12); top.addWidget(self.settings_btn)

        # --- Grid ---
        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget(); self.grid = QGridLayout(self.grid_host)
        self.grid.setHorizontalSpacing(12); self.grid.setVerticalSpacing(8)
        self.scroll.setWidget(self.grid_host)

        # Lazy renderer timer (start once, from queue)
        self._render_timer = QTimer(self); self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._drain_render_queue)
        self.scroll.verticalScrollBar().valueChanged.connect(self._enqueue_visible_needed)

        # --- Footer ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        apply_btn = buttons.addButton("Apply", QDialogButtonBox.ActionRole)

        lay = QVBoxLayout(self)
        lay.addLayout(top); lay.addWidget(self.scroll); lay.addWidget(buttons)

        # Maps
        self._cache  = getattr(parent, "pipeline_config", {}).setdefault("_element_images", {})
        self._thumbs = {}     # el -> QLabel (image)
        self._cbars  = {}     # el -> QLabel (colorbar)
        self._checks = {}     # el -> QCheckBox
        self._tiles  = {}     # el -> QWidget (tile)
        self._raw    = {}     # el -> np.ndarray
        self._global_min = None; self._global_max = None

        # Initial selection
        self._initial_checked = set(checked_elements)

        # Signals
        self.only_enabled_box.toggled.connect(self._apply_filter_fast)
        btn_all.clicked.connect(self._select_all_visible)
        btn_none.clicked.connect(self._select_none_visible)
        btn_inv.clicked.connect(self._invert_visible)
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        apply_btn.clicked.connect(self._apply_without_close)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.save_grid_btn.clicked.connect(self._export_grid_png)

        # Build
        self._prepare_raw_images()
        self._compute_global_range()
        self._build_grid_once()
        self._apply_filter_fast()

    # ---------- Data ----------
    def _prepare_raw_images(self):
        xs = self._df["X"].astype(int).values
        ys = self._df["Y"].astype(int).values
        for el in self._elements:
            if el in self._raw: continue
            if el in self._cache:
                self._raw[el] = self._cache[el]; continue
            img = np.zeros((self._H, self._W), dtype=float)
            img[ys, xs] = self._df[el].values
            self._cache[el] = img; self._raw[el] = img

    # ---------- Ranges ----------
    def _compute_global_range(self):
        if not self._range_dirty:
            return
        self._range_dirty = False

        if not self._raw or self._norm_mode == "Per image":
            self._global_min = self._global_max = None
            return

        # Always build stack in LINEAR domain (clip for log)
        eps = float(self._log_eps)
        arrays = [ (np.clip(a, eps, None) if self._scale_mode == "Log10" else a).ravel()
                for a in self._raw.values() ]
        stack = np.concatenate(arrays)

        if self._norm_mode == "Global min/max":
            self._global_min = float(np.nanmin(stack))
            self._global_max = float(np.nanmax(stack))
        else:  # Global percentile
            self._global_min = float(np.nanpercentile(stack, self._pct_lo))
            self._global_max = float(np.nanpercentile(stack, self._pct_hi))

    # ---------- Build grid once ----------
    def _build_grid_once(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget(): item.widget().setParent(None)
        self._thumbs.clear(); self._checks.clear(); self._tiles.clear(); self._cbars.clear()
        self._rendered.clear(); self._render_queue.clear()

        cols = 6; row = col = 0
        for el in self._elements:
            tile = QWidget(); v = QVBoxLayout(tile); v.setContentsMargins(0,0,0,0); v.setSpacing(4)
            thumb_lbl = ClickableLabel(); thumb_lbl.setMinimumWidth(140)
            cbar_lbl = QLabel(); cbar_lbl.setFixedWidth(self._cbar_pixel_width())
            cbar_lbl.setVisible(self._cbar_mode == "Thumbnails + zoom")

            row_img = QHBoxLayout(); row_img.setContentsMargins(0,0,0,0); row_img.setSpacing(6)
            row_img.addWidget(thumb_lbl, 1); row_img.addWidget(cbar_lbl)

            chk = QCheckBox(el); chk.setChecked(el in self._initial_checked)
            chk.stateChanged.connect(lambda _st, el=el: self._on_checkbox_changed(el))
            thumb_lbl.clicked.connect(lambda el=el: self._open_quicklook(el))

            v.addLayout(row_img); v.addWidget(chk)

            self._thumbs[el] = thumb_lbl; self._cbars[el] = cbar_lbl
            self._checks[el] = chk;       self._tiles[el]  = tile
            self.grid.addWidget(tile, row, col); col += 1
            if col >= cols: col = 0; row += 1

            if tile.isVisibleTo(self.scroll.viewport()):
                self._render_queue.append(el)

        # kick lazy rendering ONCE (outside the loop)
        if self._render_queue and not self._render_timer.isActive():
            self._render_timer.start(0)

    # ---------- Filtering ----------
    def _apply_filter_fast(self):
        bar = self.scroll.verticalScrollBar(); pos = bar.value()
        rx = self.search_edit.text().strip()
        pat = None
        if rx:
            try: pat = re.compile(rx, re.I)
            except re.error: pat = None

        show_enabled_only = self.only_enabled_box.isChecked()
        for el, tile in self._tiles.items():
            if pat and not pat.search(el): tile.setVisible(False); continue
            if show_enabled_only and not self._checks[el].isChecked(): tile.setVisible(False); continue
            tile.setVisible(True)

        bar.setValue(pos)
        self._enqueue_visible_needed()

    def _on_checkbox_changed(self, el):
        if self.only_enabled_box.isChecked():
            self._tiles[el].setVisible(self._checks[el].isChecked())

    # ---------- Thumbnails ----------
    def _v_range_for(self, arr_in_domain):
        if self._norm_mode == "Per image":
            lo, hi = np.nanpercentile(arr_in_domain, 2), np.nanpercentile(arr_in_domain, 98)
            if lo == hi: lo, hi = float(np.nanmin(arr_in_domain)), float(np.nanmax(arr_in_domain))
            return lo, hi
        return self._global_min, self._global_max

    def _v_range_lin(self, img_linear):
        """Return vmin/vmax in LINEAR domain for current norm mode."""
        if self._norm_mode == "Per image":
            lo, hi = np.nanpercentile(img_linear, 2), np.nanpercentile(img_linear, 98)
            if lo == hi:
                lo, hi = float(np.nanmin(img_linear)), float(np.nanmax(img_linear))
            return lo, hi
        else:
            return self._global_min, self._global_max
        
    def _render_thumb(self, el):
        img_lin = self._raw[el].astype(float, copy=False)

        # vmin/vmax in LINEAR domain
        eps = float(self._log_eps)
        base = np.clip(img_lin, eps, None) if self._scale_mode == "Log10" else img_lin
        vmin, vmax = self._v_range_lin(base)

        # cache key only on settings; data/range are reproducible given settings
        settings_id = (self._cmap, self._norm_mode, round(self._pct_lo,2), round(self._pct_hi,2),
                    self._scale_mode, round(eps, 12), 140)
        key = (el, settings_id)
        pm = self._thumb_cache.get(key)
        if pm is None:
            # --- Matplotlib path (matches QuickLook) ---
            fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=120)
            if self._scale_mode == "Log10":
                im = ax.imshow(np.clip(img_lin, eps, None), cmap=self._cmap,
                            norm=LogNorm(vmin=max(eps, vmin), vmax=vmax))
            else:
                im = ax.imshow(img_lin, cmap=self._cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
            plt.close(fig)
            qimg = QImage.fromData(buf.getvalue())
            pm = QPixmap.fromImage(qimg).scaledToWidth(140, Qt.SmoothTransformation)
            self._thumb_cache[key] = pm

        self._thumbs[el].setPixmap(pm)

        # Thumbnail colorbar (use LINEAR vmin/vmax)
        if self._cbar_mode == "Thumbnails + zoom":
            self._render_cbar(el, pm.height(), vmin, vmax)  # pass LINEAR bounds
        else:
            self._cbars[el].setVisible(False)

    def _refresh_thumbs(self):
        self._compute_global_range()
        for el, lbl in self._thumbs.items():
            if lbl.isVisible(): self._render_thumb(el)

    # ---------- Colorbar helpers ----------
    def _cbar_pixel_width(self): return 40 if self._cbar_mode == "Thumbnails + zoom" else 14

    def _get_cbar_base(self):
        pm = self._cbar_base_cache.get(self._cmap)
        if pm is None:
            import matplotlib.cm as cm
            lut = (mpl.colormaps[self._cmap](np.linspace(0,1,256)) * 255.0).astype(np.uint8)
            rgba = np.ascontiguousarray(lut[::-1, :])
            qimg = QImage(rgba.data, 1, 256, 4*1, QImage.Format_RGBA8888).copy()
            pm = QPixmap.fromImage(qimg)
            self._cbar_base_cache[self._cmap] = pm
        return pm

    def _render_cbar(self, el, thumb_height, vmin, vmax):
        base = self._get_cbar_base()
        bar = base.scaled(14, thumb_height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        if self._cbar_mode != "Thumbnails + zoom":
            self._cbars[el].setPixmap(bar); self._cbars[el].setVisible(False); return

        total_w = 40
        qimg = QImage(total_w, thumb_height, QImage.Format_ARGB32); qimg.fill(Qt.transparent)
        p = QPainter(qimg); p.setRenderHint(QPainter.TextAntialiasing, True)
        p.drawPixmap(total_w - 14, 0, bar)

        # label values in linear domain
        if self._scale_mode == "Log10":
            lo_val = 0.0 if vmin is None else float(vmin)
            hi_val = 1.0 if vmax is None else float(vmax)
        else:
            lo_val = float(vmin) if vmin is not None else 0.0
            hi_val = float(vmax) if vmax is not None else 1.0

        def _fmt(val):
            if not np.isfinite(val) or val == 0: return "0"
            e = int(np.floor(np.log10(abs(val))))
            return f"{val:.0e}" if e >= 3 or e <= -3 else f"{val:.3g}"

        f = QFont(); f.setPointSize(8); p.setFont(f); p.setPen(Qt.black)
        margin = 2; text_w = total_w - 14 - 2*margin
        p.drawText(0, 0, text_w, 14, Qt.AlignLeft | Qt.AlignTop, _fmt(hi_val))
        p.drawText(0, thumb_height - 14, text_w, 14, Qt.AlignLeft | Qt.AlignBottom, _fmt(lo_val))
        p.end()

        self._cbars[el].setPixmap(QPixmap.fromImage(qimg)); self._cbars[el].setVisible(True)

    # ---------- Lazy rendering ----------
    def _enqueue_visible_needed(self):
        for el, tile in self._tiles.items():
            if tile.isVisible() and tile.isVisibleTo(self.scroll.viewport()) and el not in self._rendered:
                self._render_queue.append(el)
        if self._render_queue and not self._render_timer.isActive():
            self._render_timer.start(0)

    def _drain_render_queue(self, batch=24):
        n = min(batch, len(self._render_queue))
        for _ in range(n):
            el = self._render_queue.pop(0)
            self._render_thumb(el); self._rendered.add(el)
        if self._render_queue: self._render_timer.start(0)

    # ---------- Quick look ----------
    def _open_quicklook(self, el):
        img = self._raw[el]  # linear data
        # use the same domain as thumbnails to compute vmin/vmax
        arr = np.log10(np.clip(img, float(self._log_eps), None)) if self._scale_mode == "Log10" else img.astype(float, copy=False)
        vmin, vmax = self._v_range_for(arr)

        dlg = QuickLookDialog(
            element_name=el,
            img_linear=img,
            cmap=self._cmap,
            scale_mode=self._scale_mode,   # "Linear" or "Log10"
            vmin=vmin, vmax=vmax,
            eps=float(self._log_eps),
            show_colorbar=(self._cbar_mode in ("Zoom only", "Thumbnails + zoom")),
            parent=self
        )
        dlg.exec()

    # ---------- Bulk actions ----------
    def _visible_elements(self):
        rx = self.search_edit.text().strip(); pat = None
        if rx:
            try: pat = re.compile(rx, re.I)
            except re.error: pat = None
        return [el for el, tile in self._tiles.items() if (pat is None or pat.search(el))]

    def _select_all_visible(self):
        with self._batch_checks():
            for el in self._visible_elements():
                cb = self._checks[el]; QSignalBlocker(cb); cb.setChecked(True)
        self._apply_filter_fast()

    def _select_none_visible(self):
        with self._batch_checks():
            for el in self._visible_elements():
                cb = self._checks[el]; QSignalBlocker(cb); cb.setChecked(False)
        self._apply_filter_fast()

    def _invert_visible(self):
        with self._batch_checks():
            for el in self._visible_elements():
                cb = self._checks[el]; QSignalBlocker(cb); cb.setChecked(not cb.isChecked())
        self._apply_filter_fast()

    def _batch_checks(self):
        class _Batch:
            def __init__(self, w): self.w = w
            def __enter__(self): self.w.setUpdatesEnabled(False)
            def __exit__(self, *args): self.w.setUpdatesEnabled(True)
        return _Batch(self.grid_host)

    # ---------- Settings ----------
    def _open_settings_dialog(self):
        dlg = QDialog(self); dlg.setWindowTitle("Element view settings")
        lay = QVBoxLayout(dlg)

        row1 = QHBoxLayout(); row1.addWidget(QLabel("Colormap:"))
        cmap = QComboBox(); cmap.addItems(["inferno","magma","viridis","cividis","plasma","gray"]); cmap.setCurrentText(self._cmap)
        row1.addWidget(cmap); lay.addLayout(row1)

        row2 = QHBoxLayout(); row2.addWidget(QLabel("Scaling:"))
        norm = QComboBox(); norm.addItems(["Per image","Global min/max","Global percentile"]); norm.setCurrentText(self._norm_mode)
        row2.addWidget(norm); lay.addLayout(row2)

        row3 = QHBoxLayout()
        lo = QDoubleSpinBox(); lo.setRange(0.0, 20.0); lo.setDecimals(2); lo.setValue(self._pct_lo)
        hi = QDoubleSpinBox(); hi.setRange(80.0, 100.0); hi.setDecimals(2); hi.setValue(self._pct_hi)
        row3.addWidget(QLabel("Clip %:")); row3.addWidget(lo); row3.addWidget(hi); lay.addLayout(row3)

        row4 = QHBoxLayout(); row4.addWidget(QLabel("Display:"))
        scale = QComboBox(); scale.addItems(["Linear", "Log10"]); scale.setCurrentText(self._scale_mode)
        eps = QDoubleSpinBox(); eps.setRange(1e-20, 1e-2); eps.setDecimals(12); eps.setValue(self._log_eps)
        row4.addWidget(scale); row4.addWidget(QLabel("ε:")); row4.addWidget(eps); lay.addLayout(row4)

        row5 = QHBoxLayout(); row5.addWidget(QLabel("Color scale:"))
        cbar_mode = QComboBox(); cbar_mode.addItems(["Off", "Zoom only (default)", "Thumbnails + zoom"])
        current = {"Off":"Off","Zoom only":"Zoom only (default)","Thumbnails + zoom":"Thumbnails + zoom"}[self._cbar_mode]
        cbar_mode.setCurrentText(current); row5.addWidget(cbar_mode); row5.addStretch(1); lay.addLayout(row5)

        def _toggle_pct():
            use = norm.currentText() == "Global percentile"
            lo.setEnabled(use); hi.setEnabled(use)
        _toggle_pct(); norm.currentTextChanged.connect(_toggle_pct)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply"); lay.addWidget(buttons)

        def _apply_and_close():
            m = {"Off":"Off","Zoom only (default)":"Zoom only","Thumbnails + zoom":"Thumbnails + zoom"}
            self._apply_settings_and_close(
                cmap.currentText(), norm.currentText(), lo.value(), hi.value(),
                scale.currentText(), eps.value(), m[cbar_mode.currentText()]
            )
            dlg.accept()

        buttons.accepted.connect(_apply_and_close)
        buttons.rejected.connect(dlg.reject)
        dlg.exec()

    def _apply_without_close(self):
        self.applied.emit(self.get_selected_elements())

    def get_selected_elements(self):
        return [el for el, cb in self._checks.items() if cb.isChecked()]

    def _apply_settings_and_close(self, cmap, norm_mode, lo, hi, scale_mode, eps, cbar_mode):
        changed = (
            self._cmap != cmap or self._norm_mode != norm_mode or
            round(self._pct_lo,2) != round(lo,2) or round(self._pct_hi,2) != round(hi,2) or
            self._scale_mode != scale_mode or round(float(self._log_eps),12) != round(float(eps),12) or
            self._cbar_mode != cbar_mode
        )
        if not changed: return
        self._cmap, self._norm_mode = cmap, norm_mode
        self._pct_lo, self._pct_hi = float(lo), float(hi)
        self._scale_mode, self._log_eps = scale_mode, float(eps)
        self._cbar_mode = cbar_mode

        # invalidate + lazy re-render
        self._range_dirty = True
        self._thumb_cache.clear()
        self._cbar_base_cache.clear()
        self._rendered.clear()
        self._render_queue.clear()

        # adjust cbar widgets
        for el, lbl in self._cbars.items():
            lbl.setFixedWidth(self._cbar_pixel_width())
            lbl.setVisible(self._cbar_mode == "Thumbnails + zoom")

        self._compute_global_range()
        self._apply_filter_fast()

    def _visible_element_list(self):
        """Elements that are currently visible under the search + 'enabled only' filters."""
        rx = self.search_edit.text().strip()
        pat = None
        if rx:
            try: pat = re.compile(rx, re.I)
            except re.error: pat = None
        only_enabled = self.only_enabled_box.isChecked()
        out = []
        for el, tile in self._tiles.items():
            if pat and not pat.search(el):
                continue
            if only_enabled and not self._checks[el].isChecked():
                continue
            out.append(el)
        return out

    def _make_cbar_pixmap(self, height, vmin, vmax, with_labels):
        """Return a vertical colorbar QPixmap (14px or 40px wide)."""
        base = self._get_cbar_base().scaled(14, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        if not with_labels:
            return base
        total_w = 40
        qimg = QImage(total_w, height, QImage.Format_ARGB32)
        qimg.fill(Qt.transparent)
        p = QPainter(qimg)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.drawPixmap(total_w - 14, 0, base)

        # label values in linear domain
        if self._scale_mode == "Log10":
            lo_val = 10.0 ** float(vmin) if vmin is not None else self._log_eps
            hi_val = 10.0 ** float(vmax) if vmax is not None else self._log_eps
        else:
            lo_val = float(vmin) if vmin is not None else 0.0
            hi_val = float(vmax) if vmax is not None else 1.0

        def _fmt(val):
            if not np.isfinite(val) or val == 0:
                return "0"
            e = int(np.floor(np.log10(abs(val))))
            return f"{val:.0e}" if e >= 3 or e <= -3 else f"{val:.3g}"

        f = QFont(); f.setPointSize(8)
        p.setFont(f); p.setPen(Qt.black)
        margin = 2; text_w = total_w - 14 - 2*margin
        p.drawText(0, 0, text_w, 14, Qt.AlignLeft | Qt.AlignTop, _fmt(hi_val))
        p.drawText(0, height - 14, text_w, 14, Qt.AlignLeft | Qt.AlignBottom, _fmt(lo_val))
        p.end()
        return QPixmap.fromImage(qimg)

    def _export_grid_png(self):
        els = self._visible_element_list()
        if not els:
            QMessageBox.information(self, "Nothing to export", "No elements are currently visible under the filters.")
            return

        # Ensure thumbs for these elements exist (also gives us vmin/vmax per tile)
        tiles = []
        for el in els:
            # build array in current display domain to get ranges
            img = self._raw[el]
            arr = np.log10(np.clip(img, float(self._log_eps), None)) if self._scale_mode == "Log10" else img.astype(float, copy=False)
            vmin, vmax = self._v_range_for(arr)
            # get (or build) the thumbnail pixmap at 140px width
            settings_id = (self._cmap, self._norm_mode, round(self._pct_lo,2), round(self._pct_hi,2),
                        self._scale_mode, round(float(self._log_eps), 12), 140)
            key = (el, settings_id)
            pm = self._thumb_cache.get(key)
            if pm is None:
                # render once, then read back
                self._render_thumb(el)
                pm = self._thumb_cache.get(key)
            tiles.append((el, pm, vmin, vmax))

        # Layout constants
        cols = 6
        label_h = 18
        pad = 12
        gap = 8
        show_cbar = (self._cbar_mode == "Thumbnails + zoom")
        cbar_w = 40 if show_cbar else 0

        # Determine per-tile size (they can vary slightly in height depending on data)
        tile_w = 140 + (6 if show_cbar else 0) + cbar_w
        tile_hs = [pm.height() + label_h for (_, pm, _, _) in tiles]
        # Compute rows
        rows = (len(tiles) + cols - 1) // cols
        # Height per row = max tile_h in that row
        row_heights = []
        for r in range(rows):
            row_heights.append(max(tile_hs[r*cols:(r+1)*cols]))
        total_w = pad*2 + cols*tile_w + (cols-1)*gap
        total_h = pad*2 + sum(row_heights) + (rows-1)*gap

        # Compose
        canvas = QImage(total_w, total_h, QImage.Format_ARGB32)
        canvas.fill(Qt.white)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        y = pad
        t = 0
        for r in range(rows):
            x = pad
            row_h = row_heights[r]
            for c in range(cols):
                if t >= len(tiles):
                    break
                el, pm, vmin, vmax = tiles[t]
                # center vertically within this row
                yy = y + (row_h - (pm.height() + label_h))//2

                # image
                p.drawPixmap(x, yy, pm)
                xx = x + pm.width() + 6
                # cbar
                if show_cbar:
                    cpm = self._make_cbar_pixmap(pm.height(), vmin, vmax, with_labels=True)
                    p.drawPixmap(xx, yy, cpm)
                    xx += cpm.width()

                # label under the image (span image+cbar width)
                p.drawText(x, yy + pm.height() + 14, el)

                x += tile_w + gap
                t += 1
            y += row_h + gap
        p.end()

        # Save dialog
        fd = QFileDialog(self)
        fd.setOption(QFileDialog.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptSave)
        fd.setNameFilter("PNG Images (*.png)")
        fd.selectFile("elements_grid.png")
        if fd.exec() != QFileDialog.Accepted:
            return
        path = fd.selectedFiles()[0]
        if not path.lower().endswith(".png"): path += ".png"
        if not canvas.save(path, "PNG"):
            QMessageBox.critical(self, "Save failed", "Could not write PNG.")
        else:
            QMessageBox.information(self, "Saved", f"Grid saved to:\n{path}")

class _ZoomView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            # ctrl+wheel zoom
            factor = 1.15 if e.angleDelta().y() > 0 else 1/1.15
            self.scale(factor, factor)
            self._zoom += 1 if factor > 1 else -1
        else:
            super().wheelEvent(e)

    def zoom_in(self):
        self.scale(1.15, 1.15); self._zoom += 1

    def zoom_out(self):
        self.scale(1/1.15, 1/1.15); self._zoom -= 1

    def reset_zoom(self):
        self.resetTransform(); self._zoom = 0

class ClickableLabel(QLabel):
    clicked = Signal()
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)

class QuickLookDialog(QDialog):
    """
    Rich quick-look: wheel/drag zoom, toolbar zoom buttons, save PNG, pixel value on hover.
    Build a static pixmap (matplotlib) once, then view it in a QGraphicsView.
    """
    def __init__(self, *, element_name: str, img_linear: np.ndarray,
                 cmap: str, scale_mode: str, vmin: float, vmax: float,
                 eps: float = 1e-12, show_colorbar: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle(element_name)
        self._img = img_linear               # linear-domain data for value readout
        self._h, self._w = img_linear.shape  # for coordinate mapping

        # --- render one PNG via matplotlib (with optional colorbar) ---
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=170)
        if scale_mode == "Log10":
            im = ax.imshow(np.clip(img_linear, eps, None), cmap=cmap,
                           norm=LogNorm(vmin=max(eps, vmin), vmax=vmax))
        else:
            im = ax.imshow(img_linear, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=8)

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        qimg = QImage.fromData(buf.getvalue())
        pm = QPixmap.fromImage(qimg)

        # --- scene/view ---
        self.view = _ZoomView()
        self.scene = QGraphicsScene(self.view)
        self._pix = self.scene.addPixmap(pm)
        self.view.setScene(self.scene)

        # --- toolbar ---
        tb = QToolBar()
        act_in  = QAction("Zoom in", self);  act_in.triggered.connect(self.view.zoom_in)
        act_out = QAction("Zoom out", self); act_out.triggered.connect(self.view.zoom_out)
        act_100 = QAction("Reset", self);    act_100.triggered.connect(self.view.reset_zoom)
        act_save = QAction("Save PNG…", self)
        act_save.triggered.connect(lambda: self._save_png(pm))
        tb.addAction(act_in); tb.addAction(act_out); tb.addAction(act_100); tb.addSeparator(); tb.addAction(act_save)

        # value label
        self._val_lbl = QLabel("x: -, y: -, value: -")
        self._val_lbl.setStyleSheet("color: #444")
        tb.addSeparator(); tb.addWidget(self._val_lbl)

        # map mouse move to value readout
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        # layout
        lay = QVBoxLayout(self)
        lay.addWidget(tb)
        lay.addWidget(self.view)
        self.resize(min(1400, pm.width()+48), min(1000, pm.height()+96))

    # map cursor→array index through the pixmap item geometry
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.MouseMove:
            pos = self.view.mapToScene(ev.pos())
            if self._pix.contains(pos):
                br = self._pix.boundingRect()
                x_norm = (pos.x() - br.left()) / br.width()
                y_norm = (pos.y() - br.top()) / br.height()
                xi = int(np.clip(round(x_norm * (self._w - 1)), 0, self._w - 1))
                yi = int(np.clip(round(y_norm * (self._h - 1)), 0, self._h - 1))
                val = float(self._img[yi, xi])
                self._val_lbl.setText(f"x: {xi}, y: {yi}, value: {val:.6g}")
            else:
                self._val_lbl.setText("x: -, y: -, value: -")
        return super().eventFilter(obj, ev)

    def _save_png(self, pm: QPixmap):
        fd = QFileDialog(self)
        fd.setOption(QFileDialog.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptSave)
        fd.selectFile("element.png")
        fd.setNameFilter("PNG Files (*.png)")
        if fd.exec() != QDialog.Accepted:
            return
        path = fd.selectedFiles()[0]
        if path:
            pm.save(path, "PNG")

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

def compute_moran_scores(df_features, coordinates_df):
    """
    Return full Moran's I scores for all element columns (no thresholding).
    Output: DataFrame indexed by element with column 'Morans_I', sorted desc.
    """
    if "X" not in coordinates_df.columns or "Y" not in coordinates_df.columns:
        raise ValueError("Coordinates must include 'X' and 'Y' columns")

    width = int(coordinates_df["X"].max()) + 1
    height = int(coordinates_df["Y"].max()) + 1
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
    return moran_df

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

class SpectraRoiDialog(QDialog):
    """
    Spectra & ROI with ONE global x-range (channel/energy span) and ONE global pixel ROI mask.

    Spectra tab (one column, multiple rows):
      • Each row overlays a chosen subset of clusters (keeps your cluster colors).
      • Row settings via “Edit Row…” (clusters per row; Auto Y or manual Y).
      • Global: Aggregate (mean/sum), Y-scale (linear/log), Show % in legend.
      • LEFT-drag on any row or ROI-bottom spectra → set global span (per-row bands update).
      • RIGHT-drag on any row / ROI-bottom spectra → zoom; RIGHT double-click → reset zoom.
      • Save PNG, Export CSV/Excel.

    ROI Imaging tab:
      • Left: heatmap for ENABLED clusters; Right: heatmap for ALL clusters (reference).
      • Rectangle ROI LEFT-drag on left heatmap: zooms both heatmaps, sets global pixel mask.
      • Bottom: spectra of enabled clusters (shares global span; shows its own band).
    """
    spanChanged = Signal(tuple)   # (lo, hi)
    maskChanged = Signal(object)  # bool mask

    # ----------------------------- init -----------------------------
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.resize(1150, 820)
        self.setMinimumSize(900, 720)
        self.setSizeGripEnabled(True)

        # ---- inputs ----
        self.pcfg    = getattr(parent, "pipeline_config", {})
        self.labels  = np.asarray(self.pcfg.get("clustering", {}).get("labels", []))
        self.k       = int(self.pcfg.get("clustering", {}).get(
                        "num_clusters", (int(self.labels.max())+1) if self.labels.size else 0))
        self.xy      = self.pcfg.get("original_coordinates", None)
        self.spectra = self.pcfg.get("raw_spectra", None)   # (N, C)
        self.energy  = self.pcfg.get("energy_keV", None)    # (C,) optional

        if self.labels.size == 0 or self.xy is None:
            QMessageBox.critical(self, "Missing data", "Run loading & clustering first.")
            self.reject(); return

        # If spectra are missing (CSV-only path), use the same HDF dataset picker you use elsewhere.
        if self.spectra is None:
            if not self._ensure_spectra_loaded_from_hdf():
                self.reject(); return

        self.N, self.C = self.spectra.shape
        self.current_roi_mask = np.ones(self.N, dtype=bool)   # global pixel mask
        self._span_range = None                               # global x-span (lo, hi)
        self._eps = 1e-12                                     # log floor
        self._span_idx_range = None   # (i0, i1) in channel index space

        # rows model: list of dicts: {"clusters": set[int], "ylim": None|(ymin,ymax)}
        self.rows = [{"clusters": set(range(self.k)), "ylim": None}]

        # per-row zoom + span bookkeeping
        self._row_zoomers   = []      # RectangleSelector per row (RIGHT-drag)
        self._row_defaults  = []      # (xlim_default, ylim_default) per row (for right-dblclick)
        self._roi_zoom_sel  = None    # RIGHT-drag zoom on ROI spectra

        # persistent bands: one per row + one on ROI spectra
        self._row_span_patches   = [] # patch per row
        self._roi_span_patch     = None

        # span selectors (LEFT-drag). We keep them around but hide their visuals after each selection.
        self._row_span_selectors = []
        self._span_roi = None

        # ---- tabs ----
        tabs = QTabWidget(self)

        # ===================== Spectra tab =====================
        spec_tab = QWidget(); spec_v = QVBoxLayout(spec_tab)

        top = QHBoxLayout()
        top.addWidget(QLabel("Aggregate:"))
        self.agg_combo = QComboBox(); self.agg_combo.addItems(["mean", "sum"])
        top.addWidget(self.agg_combo)

        top.addSpacing(12); top.addWidget(QLabel("Y-scale:"))
        self.yscale = QComboBox(); self.yscale.addItems(["linear","log"]); self.yscale.setCurrentText("log")
        top.addWidget(self.yscale)

        top.addSpacing(12)
        self.show_pct = QCheckBox("Show % in legend"); self.show_pct.setChecked(True)
        top.addWidget(self.show_pct)

        top.addStretch(1)
        top.addWidget(QLabel("Row:"))
        self.row_select = QComboBox(); self._refresh_row_select(); top.addWidget(self.row_select)

        self.btn_edit_row  = QPushButton("Edit Row…")
        self.btn_add_row   = QPushButton("Add row")
        self.btn_remove    = QPushButton("Remove selected row")
        self.btn_one_per   = QPushButton("One-per-cluster")
        self.btn_clear     = QPushButton("Clear → 1 row (all)")
        top.addWidget(self.btn_edit_row); top.addWidget(self.btn_add_row)
        top.addWidget(self.btn_remove);   top.addWidget(self.btn_one_per); top.addWidget(self.btn_clear)

        spec_v.addLayout(top)

        self.spec_fig = Figure(figsize=(7.6, 5.6), constrained_layout=True)
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        spec_v.addWidget(self.spec_canvas, 1)

        span_row = QHBoxLayout()
        self.span_info = QLabel("Selected range: (LEFT-drag on any row or ROI spectra)")
        span_row.addWidget(self.span_info); span_row.addStretch(1)
        self.btn_save_png  = QPushButton("Save PNG…")
        self.btn_export    = QPushButton("Export spectra…")
        span_row.addWidget(self.btn_save_png); span_row.addWidget(self.btn_export)
        spec_v.addLayout(span_row)

        tabs.addTab(spec_tab, "Spectra")

        # ===================== ROI Imaging tab =====================
        roi_tab = QWidget(); roi_v = QVBoxLayout(roi_tab)

        cl_row = QHBoxLayout(); cl_row.addWidget(QLabel("Visible clusters (left & spectra):"))
        self.cluster_checks = []
        for cid in range(self.k):
            cb = QCheckBox(f"{cid}"); cb.setChecked(True)
            cb.stateChanged.connect(lambda *_: (self._update_heatmaps(),
                                                self._render_roi_spectra(update_only=False),
                                                self._update_all_row_span_overlays(),
                                                ))
            self.cluster_checks.append(cb); cl_row.addWidget(cb)
        cl_row.addStretch(1)
        roi_v.addLayout(cl_row)

        self.roi_fig = Figure(figsize=(5.2, 4.6), constrained_layout=True)
        self.roi_ax  = self.roi_fig.add_subplot(111)
        self.roi_canvas = FigureCanvas(self.roi_fig)

        self.map_fig = Figure(figsize=(5.2, 4.6), constrained_layout=True)
        self.map_ax  = self.map_fig.add_subplot(111)
        self.map_canvas = FigureCanvas(self.map_fig)

        canv_row = QHBoxLayout()
        canv_row.addWidget(self.roi_canvas, 1)
        canv_row.addWidget(self.map_canvas, 1)
        roi_v.addLayout(canv_row, 1)

        tools = QHBoxLayout()
        self.btn_rect        = QPushButton("Rectangle ROI")
        self.btn_zoom_roi    = QPushButton("Zoom to ROI")
        self.btn_reset_bars  = QPushButton("Reset ROI bars")
        self.btn_clear_roi   = QPushButton("Clear ROI")

        tools.addWidget(self.btn_rect)
        tools.addWidget(self.btn_zoom_roi)
        tools.addWidget(self.btn_reset_bars)
        tools.addWidget(self.btn_clear_roi)
        tools.addStretch(1)
        roi_v.addLayout(tools)

        self.roi_spec_fig = Figure(figsize=(9.0, 3.3), constrained_layout=True)
        self.roi_spec_ax  = self.roi_spec_fig.add_subplot(111)
        self.roi_spec_canvas = FigureCanvas(self.roi_spec_fig)
        roi_v.addWidget(self.roi_spec_canvas, 0)

        exp_row = QHBoxLayout()
        self.btn_save_map   = QPushButton("Save heatmap PNG…")
        self.btn_export_map = QPushButton("Export heatmap CSV…")
        exp_row.addStretch(1); exp_row.addWidget(self.btn_save_map); exp_row.addWidget(self.btn_export_map)
        roi_v.addLayout(exp_row)

        tabs.addTab(roi_tab, "ROI Imaging")

        lay = QVBoxLayout(self); lay.addWidget(tabs)

        # optional hook for external color updates
        if hasattr(parent, "cluster_colors_changed"):
            try: parent.cluster_colors_changed.connect(self._on_colors_changed)
            except Exception: pass

        # wire actions
        self.agg_combo.currentIndexChanged.connect(lambda *_: (self._render_spectra_rows(),
                                                               self._render_roi_spectra(update_only=False)))
        self.yscale.currentIndexChanged.connect(lambda *_: (self._render_spectra_rows(),
                                                            self._render_roi_spectra(update_only=False)))
        self.show_pct.stateChanged.connect(self._render_spectra_rows)

        self.row_select.currentIndexChanged.connect(self._render_spectra_rows)
        self.btn_edit_row.clicked.connect(self._edit_row_dialog)
        self.btn_add_row.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_selected_row)
        self.btn_one_per.clicked.connect(self._make_one_per_cluster)
        self.btn_clear.clicked.connect(self._clear_to_one_row)

        self.btn_save_png.clicked.connect(self._save_spectra_png)
        self.btn_export.clicked.connect(self._export_spectra)

        # existing
        self.btn_rect.clicked.connect(self._activate_rectangle_roi)
        self.btn_zoom_roi.clicked.connect(self._zoom_to_roi)
        self.btn_reset_bars.clicked.connect(self._reset_roi_bars)
        self.btn_clear_roi.clicked.connect(self._clear_roi)


        self.btn_save_map.clicked.connect(self._save_heatmap_png)
        self.btn_export_map.clicked.connect(self._export_heatmap_csv)

        # sync span between tabs (and keep zoom)
        self.spanChanged.connect(self._on_span_changed)

        # RIGHT dbl-click reset hooks (rows + ROI spectra)
        self._press_cid_spec = self.spec_canvas.mpl_connect("button_press_event", self._maybe_reset_row_zoom)
        self._press_cid_roi  = self.roi_spec_canvas.mpl_connect("button_press_event", self._maybe_reset_roi_zoom)
        self._press_cid_map1 = self.roi_canvas.mpl_connect("button_press_event", self._maybe_reset_map_zoom)
        self._press_cid_map2 = self.map_canvas.mpl_connect("button_press_event", self._maybe_reset_map_zoom)

        # ---- initial draws ----
        self._cluster_rgba_img = self._build_cluster_image()
        self._update_heatmaps()
        self._render_roi_spectra(update_only=False)
        self._render_spectra_rows()
        self._init_span_selectors()
        self._install_map_zooms()

    # ---------------------- HDF loader (CSV-only path) ----------------------
    def _ensure_spectra_loaded_from_hdf(self) -> bool:
        """When only images.csv was loaded, ask for the HDF and let the user pick a dataset."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("HDF missing")
        box.setText("This dataset has no raw spectra.\nImport the matching HDF5 now?")
        import_btn = box.addButton("Import…", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is not import_btn:
            return False

        h5_path, _ = QFileDialog.getOpenFileName(self, "Select HDF5 file", "", "HDF5 Files (*.h5 *.hdf *.hdf5)")
        if not h5_path:
            return False

        # Use the same dataset picker UI you use elsewhere (tree view)
        picker = DatasetPickerDialog(h5_path, parent=self)
        if picker.exec() != QDialog.Accepted:
            return False
        ds_path = picker.selected_path()
        if not ds_path:
            return False

        # Read and sanity-check against XY
        try:
            with h5py.File(h5_path, "r") as f:
                data = np.asarray(f[ds_path])
                if data.ndim == 3:
                    H, W, C = data.shape
                    xs = self.xy["X"].astype(int).to_numpy()
                    ys = self.xy["Y"].astype(int).to_numpy()
                    if (H, W) != (int(ys.max())+1, int(xs.max())+1):
                        QMessageBox.critical(self, "Size mismatch",
                                             f"Dataset spatial size {H}×{W} does not match XY.")
                        return False
                    arr = data.reshape(H*W, C)
                elif data.ndim == 2:
                    arr = data
                    if arr.shape[0] != len(self.xy):
                        QMessageBox.critical(self, "Size mismatch",
                                             f"Dataset length {arr.shape[0]} != number of pixels {len(self.xy)}.")
                        return False
                else:
                    QMessageBox.critical(self, "Unsupported dataset", f"Shape {data.shape} not 2D/3D.")
                    return False

                self.spectra = arr.astype(float, copy=False)
                self.N, self.C = self.spectra.shape

                # best-effort energy axis from attrs
                eng = None
                ds = f[ds_path]
                for key in ("energy", "energies", "xrf_energies", "energy_keV"):
                    if key in ds.attrs:
                        e = np.asarray(ds.attrs[key]).astype(float)
                        eng = e/1000.0 if np.nanmax(e) > 5000 else e
                        break
                if eng is not None and eng.size == self.C:
                    self.energy = eng

            # persist back
            self.pcfg["raw_spectra"] = self.spectra
            if self.energy is not None:
                self.pcfg["energy_keV"] = self.energy
            return True

        except Exception as e:
            QMessageBox.critical(self, "Import error", str(e))
            return False

    # ----------------------------- helpers -----------------------------
    def _x_axis(self):
        if self.energy is not None and np.size(self.energy) == self.C:
            return np.asarray(self.energy), "Energy (keV)", True
        return np.arange(self.C), "Channels", False

    def _get_cluster_colors(self):
        rgba = (self.pcfg or {}).get("cluster_rgba", None)
        if rgba is None:
            cmap = mpl.colormaps["tab20"].resampled(max(1, self.k))
            rgba = list(map(tuple, cmap.colors))
        out = []
        for c in rgba:
            if len(c) == 3: r,g,b = c; a = 1.0
            else: r,g,b,a = c[:4]
            if max(r,g,b,a) > 1.01:
                r,g,b,a = r/255.0, g/255.0, b/255.0, a/255.0
            out.append((float(r), float(g), float(b), float(a)))
        if len(out) < self.k:
            out += [out[i % len(out)] for i in range(self.k - len(out))]
        return out[:self.k]

    def _on_colors_changed(self):
        self._cluster_rgba_img = self._build_cluster_image()
        self._render_spectra_rows(preserve_limits=True)
        self._render_roi_spectra(update_only=False)
        self._update_heatmaps()

    def set_cluster_colors(self, rgba_list):
        def _norm_one(c):
            if len(c) == 3: r,g,b = c; a = 1.0
            else: r,g,b,a = c[:4]
            if max(r,g,b,a) > 1.01:
                r,g,b,a = r/255.0, g/255.0, b/255.0, a/255.0
            return float(r), float(g), float(b), float(a)
        rgba = [_norm_one(c) for c in rgba_list]
        if len(rgba) < self.k:
            rgba += [rgba[i % len(rgba)] for i in range(self.k - len(rgba))]
        self.pcfg["cluster_rgba"] = rgba[:self.k]
        self._on_colors_changed()

    def _refresh_row_select(self):
        self.row_select.blockSignals(True)
        self.row_select.clear()
        for i in range(len(self.rows)):
            self.row_select.addItem(f"Row {i+1}")
        self.row_select.setCurrentIndex(0)
        self.row_select.blockSignals(False)
    
    # --- SpanSelector cleanup helpers (remove red bars, keep gray band) ---
    def _clear_selector_visuals(self, sel):
        """Remove the selection artists drawn by a SpanSelector/RectangleSelector without disabling it."""
        if sel is None:
            return
        # Matplotlib changed internals across versions, so try many names safely.
        for attr in ("_selection_artist", "_rect", "_span", "_lineleft", "_lineright"):
            art = getattr(sel, attr, None)
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    try:
                        art.set_visible(False)
                    except Exception:
                        pass

    def _clear_all_selector_visuals(self):
        """Remove all currently visible red bars/rectangles from all selectors."""
        for sel in getattr(self, "_row_span_selectors", []):
            self._clear_selector_visuals(sel)
        self._clear_selector_visuals(getattr(self, "_span_roi", None))
        
    # ---------------- Spectra drawing (per-row) ----------------
    def _render_spectra_rows(self, preserve_limits: bool=False):
        # save current limits if requested
        prev_limits = []
        if preserve_limits and self.spec_fig.axes:
            for ax in self.spec_fig.axes:
                prev_limits.append((ax.get_xlim(), ax.get_ylim()))
        else:
            prev_limits = None

        # tear down old zoomers & span selectors (also delete their artists)
        for z in self._row_zoomers:
            try: z.disconnect_events()
            except Exception: pass
        self._row_zoomers = []

        for s in self._row_span_selectors:
            try:
                self._clear_selector_visuals(s)  # <— NEW: remove the red bars
                s.disconnect_events()
            except Exception:
                pass
        self._row_span_selectors = []

        self._row_defaults = []
        self._row_span_patches = []

        self.spec_fig.clear()
        colors = self._get_cluster_colors()
        x, xlabel, _ = self._x_axis()
        agg    = self.agg_combo.currentText()
        yscale = self.yscale.currentText()
        pct    = self.pcfg.get("cluster_pct", {})
        mask   = np.ones(self.N, dtype=bool)  # Spectra tab ignores rectangle ROI

        nrows = max(1, len(self.rows))
        axs = [self.spec_fig.add_subplot(nrows, 1, i+1) for i in range(nrows)]

        for r, ax in enumerate(axs):
            for cid in sorted(self.rows[r]["clusters"]):
                idx = (self.labels == cid) & mask
                if not np.any(idx):
                    continue
                Y = self.spectra[idx]
                y = np.nanmean(Y, axis=0) if agg == "mean" else np.nansum(Y, axis=0)
                if yscale == "log": y = np.where(y <= 0, self._eps, y)
                label = f"Cluster {cid}"
                if self.show_pct.isChecked(): label += f"  {pct.get(int(cid), 0.0):.1f}%"
                ax.plot(x, y, color=colors[cid], lw=1.2, label=label)

            ax.set_ylabel("Intensity" if agg == "mean" else "Counts (sum)")
            ax.grid(True, which=("both" if yscale == "log" else "major"), alpha=0.25)
            ax.set_yscale(yscale)
            if r == nrows - 1: ax.set_xlabel(xlabel)

            # force x edges to exactly data edges
            ax.set_xlim(float(x.min()), float(x.max()))

            # manual Y or autorange, store defaults
            ylim = self.rows[r].get("ylim")
            if isinstance(ylim, tuple) and ylim is not None:
                ymin, ymax = ylim
                if yscale == "log":
                    ymin = max(self._eps, ymin)
                    ymax = max(ymin * 1.001, ymax)
                ax.set_ylim(ymin, ymax)
            else:
                ax.relim(); ax.autoscale_view()

            self._row_defaults.append(((float(x.min()), float(x.max())), ax.get_ylim()))

            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc="best", fontsize=8)

            # persistent band for this axes
            self._row_span_patches.append(self._ensure_row_span_patch(ax))

        # reapply previous limits if needed
        if preserve_limits and prev_limits and len(prev_limits) == len(axs):
            for ax, (xl, yl) in zip(axs, prev_limits):
                ax.set_xlim(*xl)
                r = self.spec_fig.axes.index(ax)
                row_yl = self.rows[r].get("ylim")
                if isinstance(row_yl, tuple) and row_yl is not None:
                    ax.set_ylim(*row_yl)
                else:
                    ax.set_ylim(*yl)

        self.spec_canvas.draw_idle()
        self._update_span_label()

        # RIGHT-drag zoom for each axes
        for r, ax in enumerate(axs):
            self._row_zoomers.append(
                RectangleSelector(
                    ax, lambda e0, e1, rr=r: self._on_row_zoom(rr, e0, e1),
                    useblit=True, button=[MouseButton.RIGHT],
                    minspanx=5, minspany=5, spancoords="data",
                    interactive=False, drag_from_anywhere=False
                )
            )

        # LEFT-drag span selectors for each row (non-interactive; we hide the visual right after)
        for ax in axs:
            sel = SpanSelector(
                ax, lambda a,b, s=None: self._on_span_select(a, b, ax_selector=sel),
                "horizontal", useblit=True, props=dict(alpha=0.15),
                interactive=False, grab_range=5,
                button=[MouseButton.LEFT], ignore_event_outside=False
            )
            self._row_span_selectors.append(sel)
            
    def _span_lohi_in_axis_units(self):
        """
        Return (lo, hi) for the current x-axis units (energy or channels),
        computed from the canonical index range if available.
        """
        # If you've already got this helper in your class, keep a single copy.
        if getattr(self, "_span_idx_range", None) is None:
            # Fallback for older projects: use raw values (already in current units)
            return getattr(self, "_span_range", None)
        x, _, _ = self._x_axis()
        i0, i1 = self._span_idx_range
        i0 = max(0, min(self.C - 1, int(i0)))
        i1 = max(0, min(self.C - 1, int(i1)))
        lo = float(x[min(i0, i1)])
        hi = float(x[max(i0, i1)])
        return (lo, hi)

    # persist band on a given axes
    def _ensure_row_span_patch(self, ax):
        lohi = self._span_lohi_in_axis_units()
        if lohi is None:
            if getattr(ax, "_roi_span_patch", None):
                try: ax._roi_span_patch.remove()
                except Exception: pass
                ax._roi_span_patch = None
            return None

        lo, hi = lohi
        p = getattr(ax, "_roi_span_patch", None)
        if p is None:
            ax._roi_span_patch = ax.axvspan(
                lo, hi, ymin=0.0, ymax=1.0,
                transform=ax.get_xaxis_transform(),
                color="k", alpha=0.08, zorder=0
            )
            return ax._roi_span_patch

        try:
            p.set_xy([[lo, 0.0], [lo, 1.0], [hi, 1.0], [hi, 0.0], [lo, 0.0]])
            return p
        except Exception:
            pass
        try:
            if hasattr(p, "set_x") and hasattr(p, "set_width"):
                p.set_x(lo); p.set_width(hi - lo)
                return p
        except Exception:
            pass

        try: p.remove()
        except Exception: pass
        ax._roi_span_patch = ax.axvspan(
            lo, hi, ymin=0.0, ymax=1.0,
            transform=ax.get_xaxis_transform(),
            color="k", alpha=0.08, zorder=0
        )
        return ax._roi_span_patch

    def _update_all_row_span_overlays(self):
        for ax in self.spec_fig.axes:
            self._ensure_row_span_patch(ax)
        self._ensure_roi_span_patch()
        self.spec_canvas.draw_idle()
        self.roi_spec_canvas.draw_idle()

    # RIGHT dbl-click reset (rows)
    def _maybe_reset_row_zoom(self, event):
        if event is None or event.inaxes is None:
            return
        if event.dblclick and event.button == MouseButton.RIGHT and event.inaxes in self.spec_fig.axes:
            ax = event.inaxes
            r = self.spec_fig.axes.index(ax)
            xlim_def, ylim_def = self._row_defaults[r] if r < len(self._row_defaults) else (ax.get_xlim(), ax.get_ylim())
            ax.set_xlim(*xlim_def)
            ylim = self.rows[r].get("ylim")
            if isinstance(ylim, tuple) and ylim is not None:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(*ylim_def)
            self.spec_canvas.draw_idle()

    def _on_row_zoom(self, r, e0, e1):
        if r < 0 or r >= len(self.spec_fig.axes) or e0 is None or e1 is None:
            return
        ax = self.spec_fig.axes[r]
        if e0.xdata is None or e1.xdata is None or e0.ydata is None or e1.ydata is None:
            return
        x0, x1 = float(min(e0.xdata, e1.xdata)), float(max(e0.xdata, e1.xdata))
        y0, y1 = float(min(e0.ydata, e1.ydata)), float(max(e0.ydata, e1.ydata))
        if x0 == x1:
            xr = ax.get_xlim(); pad = max(1e-9, 0.02*(xr[1]-xr[0])); x0, x1 = (x0 - pad, x1 + pad)
        if y0 == y1:
            yr = ax.get_ylim(); pad = max(1e-12, 0.02*abs(yr[1]-yr[0])); y0, y1 = (y0 - pad, y1 + pad)
        if self.yscale.currentText() == "log":
            y0 = max(self._eps, y0); y1 = max(y0*1.001, y1)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        self.spec_canvas.draw_idle()

    # ---------------- Row editing ----------------
    def _add_row(self):
        base = set(self.rows[0]["clusters"]) if self.rows else set(range(self.k))
        self.rows.append({"clusters": base, "ylim": None})
        self._refresh_row_select()
        self._render_spectra_rows(preserve_limits=True)

    def _remove_selected_row(self):
        r = self.row_select.currentIndex()
        if len(self.rows) <= 1:
            QMessageBox.information(self, "Remove row", "Keep at least one row.")
            return
        if 0 <= r < len(self.rows):
            self.rows.pop(r)
            self._refresh_row_select()
            self._render_spectra_rows(preserve_limits=True)

    def _clear_to_one_row(self):
        self.rows = [{"clusters": set(range(self.k)), "ylim": None}]
        self._refresh_row_select()
        self._render_spectra_rows(preserve_limits=False)

    def _auto_limits_for_row(self, r):
        agg    = self.agg_combo.currentText()
        yscale = self.yscale.currentText()
        mask = np.ones(self.N, dtype=bool)  # Spectra tab ignores rectangle ROI
        ymin, ymax = np.inf, -np.inf
        for cid in self.rows[r]["clusters"]:
            idx = (self.labels == cid) & mask
            if not np.any(idx): continue
            Y = self.spectra[idx]
            y = np.nanmean(Y, axis=0) if agg == "mean" else np.nansum(Y, axis=0)
            if yscale == "log": y = np.where(y <= 0, self._eps, y)
            ymin = min(ymin, float(np.nanmin(y)))
            ymax = max(ymax, float(np.nanmax(y)))
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = (self._eps if yscale=="log" else 0.0, 1.0)
        if yscale == "log":
            ymin = max(self._eps, ymin)
            ymax = max(ymin*1.001, ymax)
        return ymin, ymax

    def _edit_row_dialog(self):
        r = self.row_select.currentIndex()
        if not (0 <= r < len(self.rows)): return

        dlg = QDialog(self); dlg.setWindowTitle(f"Edit Row {r+1}")
        v = QVBoxLayout(dlg)

        grid = QGridLayout(); ncol = 6
        checks = []
        for i in range(self.k):
            cb = QCheckBox(f"Cluster {i}")
            cb.setChecked(i in self.rows[r]["clusters"])
            grid.addWidget(cb, i // ncol, i % ncol)
            checks.append((i, cb))
        v.addLayout(grid)

        yl = QHBoxLayout()
        auto_box = QCheckBox("Auto Y"); yl.addWidget(auto_box)
        ymin_spin = QDoubleSpinBox(); ymax_spin = QDoubleSpinBox()
        ymin_spin.setDecimals(6); ymax_spin.setDecimals(6)
        ymin_spin.setRange(-1e30, 1e30); ymax_spin.setRange(-1e30, 1e30)
        yl.addSpacing(12); yl.addWidget(QLabel("Ymin:")); yl.addWidget(ymin_spin)
        yl.addSpacing(8);  yl.addWidget(QLabel("Ymax:")); yl.addWidget(ymax_spin)
        v.addLayout(yl)

        if self.rows[r]["ylim"] is None:
            y0, y1 = self._auto_limits_for_row(r)
            auto_box.setChecked(True)
            ymin_spin.setValue(y0); ymax_spin.setValue(y1)
            ymin_spin.setEnabled(False); ymax_spin.setEnabled(False)
        else:
            y0, y1 = self.rows[r]["ylim"]
            auto_box.setChecked(False)
            ymin_spin.setValue(float(y0)); ymax_spin.setValue(float(y1))
            ymin_spin.setEnabled(True);  ymax_spin.setEnabled(True)

        def _toggle_auto(state):
            use_auto = (state == Qt.Checked)
            if use_auto:
                y0, y1 = self._auto_limits_for_row(r)
                ymin_spin.setValue(y0); ymax_spin.setValue(y1)
            ymin_spin.setEnabled(not use_auto); ymax_spin.setEnabled(not use_auto)
        auto_box.stateChanged.connect(_toggle_auto)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        v.addWidget(btns)

        def _apply():
            cl = {i for i, cb in checks if cb.isChecked()}
            self.rows[r]["clusters"] = cl
            if auto_box.isChecked():
                self.rows[r]["ylim"] = None
            else:
                y0, y1 = float(ymin_spin.value()), float(ymax_spin.value())
                self.rows[r]["ylim"] = (y0, y1)
            dlg.accept()
            self._render_spectra_rows(preserve_limits=True)

        btns.accepted.connect(_apply)
        btns.rejected.connect(dlg.reject)
        dlg.exec()

    # ---------------- span selectors ----------------
    def _init_span_selectors(self):
        # Kill previous ROI selector artists (red bars) but keep it active
        if hasattr(self, "_span_roi") and self._span_roi is not None:
            try:
                self._clear_selector_visuals(self._span_roi)
                self._span_roi.disconnect_events()
            except Exception:
                pass
            self._span_roi = None

        # Kill row selectors & their artists
        for s in getattr(self, "_row_span_selectors", []):
            try:
                self._clear_selector_visuals(s)
                s.disconnect_events()
            except Exception:
                pass
        self._row_span_selectors = []

        # Install LEFT-drag span selectors on every spectra row
        for ax in self.spec_fig.axes:
            sel = SpanSelector(
                ax, lambda a, b: self._set_span(min(a, b), max(a, b)),
                "horizontal", useblit=True, props=dict(alpha=0.15),
                interactive=True, grab_range=5,
                button=[MouseButton.LEFT], ignore_event_outside=False
            )
            self._row_span_selectors.append(sel)

        # ROI tab bottom spectra (LEFT-drag)
        self._span_roi = SpanSelector(
            self.roi_spec_ax, lambda a, b: self._set_span(min(a, b), max(a, b)),
            "horizontal", useblit=True, props=dict(alpha=0.15),
            interactive=True, grab_range=5,
            button=[MouseButton.LEFT], ignore_event_outside=False
        )

        # ROI spectra RIGHT-drag zoomer
        if self._roi_zoom_sel is not None:
            try: self._roi_zoom_sel.disconnect_events()
            except Exception: pass
        self._roi_zoom_sel = RectangleSelector(
            self.roi_spec_ax, self._on_roi_zoom,
            useblit=True, button=[MouseButton.RIGHT],
            minspanx=5, minspany=5, spancoords="data",
            interactive=False, drag_from_anywhere=False
        )

    def _hide_selector_visual(self, selector):
        """Hide the transient red handles/rectangle drawn by a SpanSelector."""
        try:
            selector.set_visible(False)
        except Exception:
            pass

    def _on_span_select(self, a, b, ax_selector=None):
        lo, hi = float(min(a, b)), float(max(a, b))
        self._set_span(lo, hi)
        if ax_selector is not None:
            self._hide_selector_visual(ax_selector)  # remove the red handles immediately
        # patches get updated via spanChanged → _on_span_changed

    def _set_span(self, lo, hi, source="any"):
        # Always keep the original values around (for backwards compat/UI), but
        # canonically persist the selection in INDEX space.
        self._span_range = (float(lo), float(hi))

        # Compute index bounds under current axis mapping
        x, _, is_energy = self._x_axis()
        if is_energy:
            # map energy → nearest channel indices
            i0 = int(np.argmin(np.abs(x - float(lo))))
            i1 = int(np.argmin(np.abs(x - float(hi))))
        else:
            i0 = int(np.floor(min(lo, hi)))
            i1 = int(np.ceil (max(lo, hi)))

        i0 = max(0, min(self.C - 1, i0))
        i1 = max(0, min(self.C - 1, i1))
        self._span_idx_range = (min(i0, i1), max(i0, i1))

        # Hide transient selector visuals so only the persistent grey band remains
        self._clear_all_selector_visuals()
        self.spanChanged.emit(self._span_range)

    def _on_span_changed(self, span):
        self._span_range = (float(span[0]), float(span[1]))
        self._refresh_span_bands()

    def _refresh_span_bands(self):
        self._update_span_label()
        self._update_heatmaps()
        self._render_roi_spectra(update_only=True)
        self._update_all_row_span_overlays()
        # also ensure any transient selector visuals are hidden
        for s in self._row_span_selectors:
            self._hide_selector_visual(s)
        if self._span_roi is not None:
            self._hide_selector_visual(self._span_roi)

    def _update_span_label(self):
        if self._span_idx_range is None and self._span_range is None:
            self.span_info.setText("Selected range: (LEFT-drag on any row or ROI spectra)")
            return
        x, _, is_energy = self._x_axis()

        if self._span_idx_range is not None:
            i0, i1 = self._span_idx_range
            i0 = max(0, min(self.C - 1, int(i0)))
            i1 = max(0, min(self.C - 1, int(i1)))
            lo = float(x[min(i0, i1)])
            hi = float(x[max(i0, i1)])
        else:
            # fallback for old sessions
            lo, hi = self._span_range
            lo = max(float(x.min()), min(float(x.max()), lo))
            hi = max(float(x.min()), min(float(x.max()), hi))

        if is_energy:
            self.span_info.setText(f"Selected range: {lo:.4g} – {hi:.4g} keV")
        else:
            self.span_info.setText(f"Selected range: {int(round(lo))} – {int(round(hi))} channels")

    # ---------------- ROI (rectangle + heatmaps + spectra) ----------------
    def _build_cluster_image(self):
        xs = self.xy["X"].astype(int).to_numpy()
        ys = self.xy["Y"].astype(int).to_numpy()
        W = int(xs.max()) + 1; H = int(ys.max()) + 1
        rgba = self._get_cluster_colors()
        img = np.ones((H, W, 4), dtype=float); img[..., 3] = 0.0
        for i, (x, y) in enumerate(zip(xs, ys)):
            lbl = int(self.labels[i])
            if 0 <= x < W and 0 <= y < H:
                img[y, x] = rgba[lbl]
        return img

    def _heatmap_array(self, keep_clusters=None):
        idx_cols = self._current_span_indices()
        xs = self.xy["X"].astype(int).to_numpy()
        ys = self.xy["Y"].astype(int).to_numpy()
        W = int(xs.max()) + 1; H = int(ys.max()) + 1
        img = np.zeros((H, W), dtype=float)

        if keep_clusters is None:
            mask = np.ones(self.N, dtype=bool)
        else:
            mask = np.isin(self.labels, list(keep_clusters))

        if np.any(mask):
            vals = self.spectra[mask][:, idx_cols].sum(axis=1)
            img[ys[mask], xs[mask]] = vals
        return img

    def _apply_zoom_to_maps(self, redraw=True):
        H, W, _ = self._cluster_rgba_img.shape
        for ax in (self.roi_ax, self.map_ax):
            for p in list(ax.patches):
                p.remove()

        if hasattr(self, "_last_rect"):
            x0, y0, x1, y1 = self._last_rect
            for ax in (self.roi_ax, self.map_ax):
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)  # imshow origin='upper'
                ax.add_patch(mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.4))
        else:
            for ax in (self.roi_ax, self.map_ax):
                ax.set_xlim(0, W - 1)
                ax.set_ylim(H - 1, 0)

        if redraw:
            self.roi_canvas.draw_idle()
            self.map_canvas.draw_idle()

    def _update_roi_overlays(self, clear_only: bool = False):
        """Draw/clear ROI overlays (focus rect + hatched dimming outside)."""
        # track and remove previous overlays safely
        for attr in ("_roi_overlay_left", "_roi_overlay_right"):
            patches = getattr(self, attr, [])
            for p in patches:
                try: p.remove()
                except Exception: pass
            setattr(self, attr, [])

        if clear_only or not hasattr(self, "_last_rect"):
            # nothing else to draw
            if hasattr(self, "roi_canvas"): self.roi_canvas.draw_idle()
            if hasattr(self, "map_canvas"): self.map_canvas.draw_idle()
            return

        x0, y0, x1, y1 = self._last_rect
        H, W, _ = self._cluster_rgba_img.shape

        def _draw(ax):
            drawn = []
            # focus rectangle (thin white outline)
            try:
                r = mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.6, ec="w")
                ax.add_patch(r); drawn.append(r)
            except Exception:
                pass
            # four “outside” rectangles with a light hatch + translucent fill
            try:
                hatch = "////"
                fc = (0, 0, 0, 0.06)  # subtle dim
                ec = (0, 0, 0, 0.15)
                # left
                if x0 > 0:
                    p = mpl.patches.Rectangle((0, 0), x0, H, facecolor=fc, edgecolor=ec, hatch=hatch, lw=0)
                    ax.add_patch(p); drawn.append(p)
                # right
                if x1 < W:
                    p = mpl.patches.Rectangle((x1, 0), W - x1, H, facecolor=fc, edgecolor=ec, hatch=hatch, lw=0)
                    ax.add_patch(p); drawn.append(p)
                # top
                if y0 > 0:
                    p = mpl.patches.Rectangle((x0, 0), x1 - x0, y0, facecolor=fc, edgecolor=ec, hatch=hatch, lw=0)
                    ax.add_patch(p); drawn.append(p)
                # bottom
                if y1 < H:
                    p = mpl.patches.Rectangle((x0, y1), x1 - x0, H - y1, facecolor=fc, edgecolor=ec, hatch=hatch, lw=0)
                    ax.add_patch(p); drawn.append(p)
            except Exception:
                pass
            return drawn

        self._roi_overlay_left  = _draw(self.roi_ax)
        self._roi_overlay_right = _draw(self.map_ax)
        self.roi_canvas.draw_idle(); self.map_canvas.draw_idle()


    def _install_map_zooms(self):
        """Right-drag zoom on both heatmaps."""
        # disconnect previous
        for attr in ("_map_zoom_left", "_map_zoom_right"):
            sel = getattr(self, attr, None)
            if sel is not None:
                try: sel.disconnect_events()
                except Exception: pass
                setattr(self, attr, None)

        self._map_zoom_left = RectangleSelector(
            self.roi_ax, lambda e0, e1, ax=self.roi_ax: self._on_map_zoom(ax, e0, e1),
            useblit=True, button=[MouseButton.RIGHT],
            minspanx=5, minspany=5, spancoords="data",
            interactive=False, drag_from_anywhere=False
        )
        self._map_zoom_right = RectangleSelector(
            self.map_ax, lambda e0, e1, ax=self.map_ax: self._on_map_zoom(ax, e0, e1),
            useblit=True, button=[MouseButton.RIGHT],
            minspanx=5, minspany=5, spancoords="data",
            interactive=False, drag_from_anywhere=False
        )

    def _on_map_zoom(self, ax, e0, e1):
        """Right-drag rectangle zoom: apply to BOTH heatmaps (origin='upper')."""
        if e0 is None or e1 is None: 
            return
        if None in (e0.xdata, e1.xdata, e0.ydata, e1.ydata):
            return

        x0, x1 = float(min(e0.xdata, e1.xdata)), float(max(e0.xdata, e1.xdata))
        y0, y1 = float(min(e0.ydata, e1.ydata)), float(max(e0.ydata, e1.ydata))

        # pad degenerate selections
        def _pad_if_needed(ax_, x0_, x1_, y0_, y1_):
            if x0_ == x1_:
                xr = ax_.get_xlim(); pad = max(1e-9, 0.02*(xr[1]-xr[0])); x0_, x1_ = (x0_ - pad, x1_ + pad)
            if y0_ == y1_:
                yr = ax_.get_ylim(); pad = max(1e-9, 0.02*abs(yr[1]-yr[0])); y0_, y1_ = (y0_ - pad, y1_ + pad)
            return x0_, x1_, y0_, y1_

        x0, x1, y0, y1 = _pad_if_needed(ax, x0, x1, y0, y1)

        # Apply to BOTH heatmaps; y is inverted for imshow(origin='upper')
        for target_ax in (self.roi_ax, self.map_ax):
            target_ax.set_xlim(x0, x1)
            target_ax.set_ylim(y1, y0)

        self.roi_canvas.draw_idle()
        self.map_canvas.draw_idle()

    def _maybe_reset_map_zoom(self, event):
        """Double-right-click on either heatmap resets BOTH to default extents."""
        if event is None or event.inaxes is None:
            return
        if not (getattr(event, "dblclick", False) and event.button == MouseButton.RIGHT):
            return
        if event.inaxes not in (self.roi_ax, self.map_ax):
            return

        # Defaults are stored on each draw; compute if missing
        if not hasattr(self, "_map_default_xlim") or not hasattr(self, "_map_default_ylim"):
            H, W, _ = self._cluster_rgba_img.shape
            self._map_default_xlim = (0, W - 1)
            self._map_default_ylim = (H - 1, 0)

        for target_ax in (self.roi_ax, self.map_ax):
            target_ax.set_xlim(*self._map_default_xlim)
            target_ax.set_ylim(*self._map_default_ylim)

        self.roi_canvas.draw_idle()
        self.map_canvas.draw_idle()

    def _update_heatmaps(self):
        keep = {i for i, cb in enumerate(self.cluster_checks) if cb.isChecked()}
        img_focus  = self._heatmap_array(keep_clusters=keep)
        img_global = self._heatmap_array(keep_clusters=None)

        self.roi_ax.clear()
        im1 = self.roi_ax.imshow(img_focus, cmap="inferno")
        self.roi_ax.set_title("Heatmap (enabled clusters)"); self.roi_ax.axis("off")

        self.map_ax.clear()
        im2 = self.map_ax.imshow(img_global, cmap="inferno")
        self.map_ax.set_title("Heatmap (all clusters)"); self.map_ax.axis("off")

        for attr in ("_focus_cbar", "_global_cbar"):
            if hasattr(self, attr) and getattr(self, attr):
                try: getattr(self, attr).remove()
                except Exception: pass
                setattr(self, attr, None)

        self._focus_cbar  = self.roi_fig.colorbar(im1, ax=self.roi_ax, fraction=0.046, pad=0.03)
        self._global_cbar = self.map_fig.colorbar(im2, ax=self.map_ax, fraction=0.046, pad=0.03)

        # self._apply_zoom_to_maps(redraw=False)
        self.roi_canvas.draw_idle(); self.map_canvas.draw_idle()

        # --- NEW: remember default extents for heatmap unzoom ---
        H, W, _ = self._cluster_rgba_img.shape
        self._map_default_xlim = (0, W - 1)
        self._map_default_ylim = (H - 1, 0)  # origin='upper'

        # --- NEW: re-apply ROI overlays after clearing axes ---
        self._update_roi_overlays()

    def _activate_rectangle_roi(self):
        try:
            if hasattr(self, "_rect_sel") and self._rect_sel is not None:
                self._rect_sel.set_visible(False); self._rect_sel.disconnect_events()
        except Exception:
            pass

        def _on_select(eclick, erelease):
            if eclick is None or erelease is None: return
            if None in (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata): return
            x0 = int(np.floor(min(eclick.xdata, erelease.xdata)))
            x1 = int(np.ceil (max(eclick.xdata, erelease.xdata)))
            y0 = int(np.floor(min(eclick.ydata, erelease.ydata)))
            y1 = int(np.ceil (max(eclick.ydata, erelease.ydata)))
            self._last_rect = (x0, y0, x1, y1)

            xs = self.xy["X"].to_numpy(); ys = self.xy["Y"].to_numpy()
            in_rect = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
            self.current_roi_mask = in_rect

            # Do NOT zoom; just update overlays + spectra/rows
            self._update_roi_overlays()
            self._render_roi_spectra(update_only=False)
            # self._render_spectra_rows(preserve_limits=True)

        self._rect_sel = RectangleSelector(
            self.roi_ax, _on_select,
            useblit=True, button=[MouseButton.LEFT],
            minspanx=2, minspany=2, spancoords="pixels",
            interactive=False, drag_from_anywhere=False,
        )

    def _clear_roi(self):
        """Clear the pixel ROI + focus overlays and UNZOOM the heatmaps to defaults.
        Does NOT touch the ROI channel 'shadow' bands."""
        self.current_roi_mask = np.ones(self.N, dtype=bool)
        if hasattr(self, "_last_rect"):
            delattr(self, "_last_rect")

        self._clear_all_selector_visuals()
        self._update_roi_overlays(clear_only=True)

        self._update_heatmaps()

        if not hasattr(self, "_map_default_xlim") or not hasattr(self, "_map_default_ylim"):
            H, W, _ = self._cluster_rgba_img.shape
            self._map_default_xlim = (0, W - 1)
            self._map_default_ylim = (H - 1, 0)
        for ax in (self.roi_ax, self.map_ax):
            ax.set_xlim(*self._map_default_xlim)
            ax.set_ylim(*self._map_default_ylim)
        self.roi_canvas.draw_idle()
        self.map_canvas.draw_idle()

        self._render_roi_spectra(update_only=False)
        self._update_all_row_span_overlays()

    def _current_span_indices(self):
        # Use canonical index range if available
        if self._span_idx_range is not None:
            i0, i1 = self._span_idx_range
            i0 = max(0, min(self.C - 1, int(i0)))
            i1 = max(0, min(self.C - 1, int(i1)))
            lo, hi = (min(i0, i1), max(i0, i1))
            return np.arange(lo, hi + 1)

        # Fallback to old behavior if no index span is stored yet
        x, _, is_energy = self._x_axis()
        if self._span_range is None:
            return np.arange(self.C)
        lo, hi = self._span_range
        if is_energy:
            return np.where((x >= lo) & (x <= hi))[0]
        a = int(np.floor(min(lo, hi))); b = int(np.ceil(max(lo, hi)))
        a = max(0, min(self.C - 1, a)); b = max(0, min(self.C - 1, b))
        return np.arange(min(a, b), max(a, b) + 1)

    # ---------------- ROI bottom spectra ----------------
    def _ensure_roi_span_patch(self):
        """
        Ensure the persistent gray band on the ROI-spectra (bottom) axis matches
        the current span, recreating it if the axes were cleared or changed.
        """
        ax = self.roi_spec_ax
        lohi = self._span_lohi_in_axis_units()
        if lohi is None:
            # No span selected: remove any stale patch
            p = getattr(self, "_roi_span_patch", None)
            if p is not None:
                try: p.remove()
                except Exception: pass
            self._roi_span_patch = None
            return None

        lo, hi = lohi
        p = getattr(self, "_roi_span_patch", None)

        # If we don't have a patch yet, or the old patch was detached by ax.clear(),
        # recreate a fresh one.
        if (p is None) or (getattr(p, "axes", None) is not ax) or (p not in ax.patches):
            self._roi_span_patch = ax.axvspan(
                lo, hi, ymin=0.0, ymax=1.0,
                transform=ax.get_xaxis_transform(),
                color="k", alpha=0.08, zorder=0
            )
            return self._roi_span_patch

        # Try to update in place
        try:
            p.set_xy([[lo, 0.0], [lo, 1.0], [hi, 1.0], [hi, 0.0], [lo, 0.0]])
            return p
        except Exception:
            pass
        try:
            if hasattr(p, "set_x") and hasattr(p, "set_width"):
                p.set_x(lo); p.set_width(hi - lo)
                return p
        except Exception:
            pass

        # Fallback: recreate
        try: p.remove()
        except Exception: pass
        self._roi_span_patch = ax.axvspan(
            lo, hi, ymin=0.0, ymax=1.0,
            transform=ax.get_xaxis_transform(),
            color="k", alpha=0.08, zorder=0
        )
        return self._roi_span_patch

    def _render_roi_spectra(self, update_only=False):
        """
        Draw (or update) the spectra at the bottom of the ROI tab. If we clear the
        axes, we invalidate the stored patch so it will be recreated on this axes.
        """
        if not update_only:
            self.roi_spec_ax.clear()
            # IMPORTANT: invalidate the old patch because ax.clear() detached it.
            self._roi_span_patch = None

        colors = self._get_cluster_colors()
        x, xlabel, _ = self._x_axis()
        agg    = self.agg_combo.currentText()
        yscale = self.yscale.currentText()
        mask   = self.current_roi_mask
        keep = {i for i, cb in enumerate(self.cluster_checks) if cb.isChecked()}

        if not update_only:
            for cid in sorted(keep):
                idx = (self.labels == cid) & mask
                if not np.any(idx): 
                    continue
                Y = self.spectra[idx]
                y = np.nanmean(Y, axis=0) if agg == "mean" else np.nansum(Y, axis=0)
                if yscale == "log": 
                    y = np.where(y <= 0, self._eps, y)
                self.roi_spec_ax.plot(x, y, color=colors[cid], lw=1.1, label=f"Cluster {cid}")
            self.roi_spec_ax.set_xlabel(xlabel)
            self.roi_spec_ax.set_ylabel("Intensity" if agg == "mean" else "Counts (sum)")
            self.roi_spec_ax.set_yscale(yscale)
            self.roi_spec_ax.grid(True, which=("both" if yscale=="log" else "major"), alpha=0.25)
            # keep x edges exact
            self.roi_spec_ax.set_xlim(float(x.min()), float(x.max()))
            if self.roi_spec_ax.get_legend_handles_labels()[1]:
                self.roi_spec_ax.legend(loc="best", fontsize=8)

        # Always (re)ensure the band AFTER any axis changes
        self._ensure_roi_span_patch()
        self.roi_spec_canvas.draw_idle()

        # Remember defaults once (for dbl-click reset)
        if not hasattr(self, "_roi_default_xlim"):
            self._roi_default_xlim = (float(x.min()), float(x.max()))
        if not hasattr(self, "_roi_default_ylim"):
            self.roi_spec_ax.relim(); self.roi_spec_ax.autoscale_view()
            self._roi_default_ylim = self.roi_spec_ax.get_ylim()
        
    # ROI spectra zoom handlers
    def _on_roi_zoom(self, e0, e1):
        if e0 is None or e1 is None or e0.xdata is None or e1.xdata is None or e0.ydata is None or e1.ydata is None:
            return
        ax = self.roi_spec_ax
        x0, x1 = float(min(e0.xdata, e1.xdata)), float(max(e0.xdata, e1.xdata))
        y0, y1 = float(min(e0.ydata, e1.ydata)), float(max(e0.ydata, e1.ydata))
        if x0 == x1:
            xr = ax.get_xlim(); pad = max(1e-9, 0.02*(xr[1]-xr[0])); x0, x1 = (x0-pad, x1+pad)
        if y0 == y1:
            yr = ax.get_ylim(); pad = max(1e-12, 0.02*abs(yr[1]-yr[0])); y0, y1 = (y0-pad, y1+pad)
        if self.yscale.currentText() == "log":
            y0 = max(self._eps, y0); y1 = max(y0*1.001, y1)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        self.roi_spec_canvas.draw_idle()

    def _reset_roi_spec_zoom(self):
        """Reset ROI spectra axes limits to full X and autorange Y."""
        x, _, _ = self._x_axis()
        self.roi_spec_ax.set_xlim(float(x.min()), float(x.max()))
        self.roi_spec_ax.relim(); self.roi_spec_ax.autoscale_view()
        self.roi_spec_canvas.draw_idle()

    def _maybe_reset_roi_zoom(self, event):
        if event is None or event.inaxes is None:
            return
        if event.inaxes is self.roi_spec_ax and event.dblclick and event.button == MouseButton.RIGHT:
            # Temporarily disable the right-drag zoomer to avoid conflicts with the dblclick
            try:
                if self._roi_zoom_sel is not None:
                    self._roi_zoom_sel.set_active(False)
            except Exception:
                pass

            # Make sure we have stored defaults (done on first draw in _render_roi_spectra)
            x, _, _ = self._x_axis()
            if not hasattr(self, "_roi_default_xlim"):
                self._roi_default_xlim = (float(x.min()), float(x.max()))
            if not hasattr(self, "_roi_default_ylim"):
                self.roi_spec_ax.relim(); self.roi_spec_ax.autoscale_view()
                self._roi_default_ylim = self.roi_spec_ax.get_ylim()

            # Restore BOTH X and Y to true defaults (no fresh autoscale)
            self.roi_spec_ax.set_xlim(*self._roi_default_xlim)
            self.roi_spec_ax.set_ylim(*self._roi_default_ylim)

            # Ensure log constraints if needed
            if self.yscale.currentText() == "log":
                lo, hi = self.roi_spec_ax.get_ylim()
                lo = max(self._eps, lo); hi = max(lo * 1.001, hi)
                self.roi_spec_ax.set_ylim(lo, hi)

            self.roi_spec_canvas.draw_idle()

            # Re-enable the right-drag zoomer
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._roi_zoom_sel and self._roi_zoom_sel.set_active(True))

    def _zoom_to_roi_if_exists(self):
        """Used by cluster toggles: zoom to ROI only if a rectangle already exists."""
        if hasattr(self, "_last_rect"):
            self._zoom_to_roi()

    def _zoom_to_roi(self):
        """Zoom BOTH heatmaps to the current rectangle ROI (if any), preserving overlays."""
        if not hasattr(self, "_last_rect"):
            return
        x0, y0, x1, y1 = self._last_rect
        # set limits on BOTH canvases (origin='upper')
        for ax in (self.roi_ax, self.map_ax):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
        # keep the focus hatched overlays aligned with current rect
        self._update_roi_overlays(clear_only=False)
        self.roi_canvas.draw_idle()
        self.map_canvas.draw_idle()

    def _reset_roi_bars(self):
        """Set the ROI channel bars to cover ALL channels (keep the gray band visible)."""
        x, _, _ = self._x_axis()
        lo, hi = float(x.min()), float(x.max())
        # this emits spanChanged → updates heatmaps + bands everywhere
        self._set_span(lo, hi)
        # Ensure the persistent gray bands are displayed everywhere
        self._update_all_row_span_overlays()

    # ---------------- Exports ----------------
    def _save_spectra_png(self):
        fd = QFileDialog(self, "Save spectra PNG", "spectra.png", "PNG Image (*.png)")
        fd.setOption(QFileDialog.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptSave)
        if fd.exec() != QFileDialog.Accepted: return
        path = fd.selectedFiles()[0]
        if not path.lower().endswith(".png"): path += ".png"
        try: self.spec_fig.savefig(path, dpi=220)
        except Exception as e: QMessageBox.critical(self, "Export error", str(e))

    def _export_spectra(self):
        try:
            dlg = SpectraExportDialog(self, default_mode=self.agg_combo.currentText())
        except Exception:
            dlg = None
        if dlg and dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.get_opts() if dlg else {"format":"csv","log_safe":False,"epsilon":1e-12,"include_meta":False}

        x, _, _ = self._x_axis()
        out = {"x": x}
        mask = np.ones(self.N, dtype=bool)  # Export from Spectra tab uses ALL pixel
        agg = self.agg_combo.currentText()
        for r, row in enumerate(self.rows, start=1):
            for cid in sorted(row["clusters"]):
                idx = (self.labels == cid) & mask
                if not np.any(idx): continue
                Y = self.spectra[idx]
                y = np.nanmean(Y, axis=0) if agg=="mean" else np.nansum(Y, axis=0)
                out[f"row{r}_cluster{cid}"] = y

        df = pd.DataFrame(out)
        if opts.get("log_safe", False):
            eps = float(opts.get("epsilon", 1e-12))
            for col in df.columns[1:]:
                df[col] = np.where(df[col] <= 0, eps, df[col])

        if opts.get("format","csv") == "csv":
            fd = QFileDialog(self); fd.setOption(QFileDialog.DontUseNativeDialog, True)
            fd.setAcceptMode(QFileDialog.AcceptSave)
            fd.setNameFilter("CSV Files (*.csv)"); fd.selectFile("spectra.csv")
            if fd.exec() != QFileDialog.Accepted: return
            path = fd.selectedFiles()[0]
            if not path.lower().endswith(".csv"): path += ".csv"
            try: df.to_csv(path, index=False)
            except Exception as e: QMessageBox.critical(self, "Export error", str(e))
        else:
            fd = QFileDialog(self); fd.setOption(QFileDialog.DontUseNativeDialog, True)
            fd.setAcceptMode(QFileDialog.AcceptSave)
            fd.setNameFilter("Excel Files (*.xlsx)"); fd.selectFile("spectra.xlsx")
            if fd.exec() != QFileDialog.Accepted: return
            path = fd.selectedFiles()[0]
            if not path.lower().endswith(".xlsx"): path += ".xlsx"
            try:
                with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Spectra")
                    if opts.get("include_meta", False):
                        meta = pd.DataFrame({
                            "N":[self.N], "C":[self.C], "Y-scale":[self.yscale.currentText()],
                            "Aggregate":[self.agg_combo.currentText()]
                        })
                        meta.to_excel(writer, index=False, sheet_name="Meta")
            except Exception as e:
                QMessageBox.critical(self, "Export error", str(e))

    def _save_heatmap_png(self):
        fd = QFileDialog(self); fd.setOption(QFileDialog.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptSave)
        fd.setNameFilter("PNG Images (*.png)"); fd.selectFile("heatmap.png")
        if fd.exec() != QFileDialog.Accepted: return
        path = fd.selectedFiles()[0]
        if not path.lower().endswith(".png"): path += ".png"
        try: self.map_fig.savefig(path, dpi=200)
        except Exception as e: QMessageBox.critical(self, "Export error", str(e))

    def _export_heatmap_csv(self):
        fd = QFileDialog(self); fd.setOption(QFileDialog.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptSave)
        fd.setNameFilter("CSV Files (*.csv)"); fd.selectFile("heatmap.csv")
        if fd.exec() != QFileDialog.Accepted: return
        path = fd.selectedFiles()[0]
        if not path.lower().endswith(".csv"): path += ".csv"
        try:
            img = self._heatmap_array(keep_clusters=None)
            yy, xx = np.indices(img.shape)
            df = pd.DataFrame({"X":xx.ravel(), "Y":yy.ravel(), "Value":img.ravel()})
            df.to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def _make_one_per_cluster(self):
        self.rows = [{"clusters": {cid}, "ylim": None} for cid in range(self.k)]
        self._refresh_row_select()
        self._render_spectra_rows()

class SpectraExportDialog(QDialog):
    def __init__(self, parent=None, default_mode="sum"):
        super().__init__(parent)
        self.setWindowTitle("Export spectra")
        self.setModal(True)
        lay = QVBoxLayout(self)

        form = QFormLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Excel (.xlsx)", "CSV (.csv)"])
        form.addRow("Format:", self.format_combo)

        self.safe_box = QCheckBox("Make log-plot friendly (replace ≤ 0 by ε)")
        self.safe_box.setChecked(True)
        form.addRow("", self.safe_box)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setDecimals(12)
        self.eps_spin.setRange(1e-20, 1e-3)
        self.eps_spin.setSingleStep(1e-12)
        self.eps_spin.setValue(1e-12)
        form.addRow("ε value:", self.eps_spin)

        self.meta_box = QCheckBox("Include metadata sheet (Excel)")
        self.meta_box.setChecked(True)

        self.chart_box = QCheckBox("Embed chart (log y-axis) (Excel)")
        self.chart_box.setChecked(True)

        form.addRow("", self.meta_box)
        form.addRow("", self.chart_box)

        lay.addLayout(form)

        # Ok/Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        # Enable/disable Excel-only opts
        def toggle_excel_opts():
            excel = self.format_combo.currentIndex() == 0
            self.meta_box.setEnabled(excel)
            self.chart_box.setEnabled(excel)
        self.format_combo.currentIndexChanged.connect(toggle_excel_opts)
        toggle_excel_opts()

    def get_opts(self):
        return {
            "format": "xlsx" if self.format_combo.currentIndex() == 0 else "csv",
            "log_safe": self.safe_box.isChecked(),
            "epsilon": float(self.eps_spin.value()),
            "include_meta": self.meta_box.isChecked(),
            "embed_chart": self.chart_box.isChecked(),
        }
