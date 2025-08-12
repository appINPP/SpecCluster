# functions_pyside.py
import io
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    QGraphicsView, QGraphicsScene,QToolBar
    )
from PySide6.QtGui import QPixmap, QImage, QPainter, QFont, QGuiApplication, QAction
from PySide6.QtCore import Qt, Signal, QSignalBlocker, QTimer,QEvent, QPointF, QSize, QCoreApplication
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from libpysal.weights import lat2W
from esda.moran import Moran
from sklearn.cluster import KMeans
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator

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
        if not self._range_dirty: return
        self._range_dirty = False
        if not self._raw or self._norm_mode == "Per image":
            self._global_min = self._global_max = None; return
        eps = float(self._log_eps)
        if self._scale_mode == "Log10":
            stack = np.concatenate([np.log10(np.clip(a, eps, None)).ravel() for a in self._raw.values()])
        else:
            stack = np.concatenate([a.ravel() for a in self._raw.values()])
        if self._norm_mode == "Global min/max":
            self._global_min, self._global_max = float(np.nanmin(stack)), float(np.nanmax(stack))
        else:
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

    def _render_thumb(self, el):
        img = self._raw[el]

        # choose domain (linear/log) for range + labels
        if self._scale_mode == "Log10":
            arr = np.log10(np.clip(img, float(self._log_eps), None))
        else:
            arr = img.astype(float, copy=False)

        # compute vmin/vmax in that domain
        vmin, vmax = self._v_range_for(arr)

        # fetch/create pixmap
        settings_id = (self._cmap, self._norm_mode, round(self._pct_lo,2), round(self._pct_hi,2),
                       self._scale_mode, round(float(self._log_eps), 12), 140)
        key = (el, settings_id)
        pm = self._thumb_cache.get(key)
        if pm is None:
            # normalize to 0..255
            if vmin is None or vmax is None or not np.isfinite([vmin, vmax]).all() or vmin == vmax:
                vmin, vmax = 0.0, 1.0
            norm = (arr - vmin) / (vmax - vmin + 1e-12)
            idx = np.clip((norm * 255.0).round().astype(np.uint8), 0, 255)

            # colormap lookup
            import matplotlib.cm as cm
            lut = (cm.get_cmap(self._cmap)(np.linspace(0,1,256)) * 255.0).astype(np.uint8)
            rgba = lut[idx]  # (H,W,4)
            h, w = rgba.shape[:2]
            qimg = QImage(rgba.data, w, h, 4*w, QImage.Format_RGBA8888).copy()
            pm = QPixmap.fromImage(qimg).scaledToWidth(140, Qt.SmoothTransformation)
            self._thumb_cache[key] = pm

        self._thumbs[el].setPixmap(pm)

        if self._cbar_mode == "Thumbnails + zoom":
            self._render_cbar(el, pm.height(), vmin, vmax)
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
            lut = (cm.get_cmap(self._cmap)(np.linspace(0,1,256)) * 255.0).astype(np.uint8)
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
            lo_val = 10.0 ** float(vmin) if vmin is not None else self._log_eps
            hi_val = 10.0 ** float(vmax) if vmax is not None else self._log_eps
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

class QuickLookDialog(QDialog):
    """
    Zoomable, savable viewer with hover/click value readout.
    - Wheel zoom, Right-click -> zoom out, Ctrl+0 -> reset
    - 'Save PNG…' button renders a high-quality export (no blur)
    - Hover shows (x,y) and value; left-click pins it
    """
    def __init__(self, element_name, img_linear, cmap, scale_mode, vmin, vmax, eps=1e-12, show_colorbar=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle(element_name)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1000, 800)

        self.img_linear = np.asarray(img_linear, dtype=float)  # (H,W)
        self.cmap = cmap
        self.scale_mode = scale_mode
        self.vmin = float(vmin) if vmin is not None else None
        self.vmax = float(vmax) if vmax is not None else None
        self.eps  = float(eps)
        self.show_cbar = bool(show_colorbar)

        # Build display pixmap at native data resolution (1:1 mapping)
        disp_arr = np.log10(np.clip(self.img_linear, self.eps, None)) if self.scale_mode == "Log10" else self.img_linear
        if self.vmin is None or self.vmax is None or not np.isfinite([self.vmin, self.vmax]).all() or self.vmin == self.vmax:
            self.vmin, self.vmax = float(np.nanmin(disp_arr)), float(np.nanmax(disp_arr))
            if self.vmin == self.vmax:
                self.vmin, self.vmax = 0.0, 1.0

        # LUT map -> QImage
        import matplotlib.cm as cm
        lut = (cm.get_cmap(self.cmap)(np.linspace(0,1,256)) * 255.0).astype(np.uint8)
        norm = (disp_arr - self.vmin) / (self.vmax - self.vmin + 1e-12)
        idx = np.clip((norm * 255.0).round().astype(np.uint8), 0, 255)
        rgba = lut[idx]  # (H,W,4)
        H, W = rgba.shape[:2]
        qimg = QImage(np.ascontiguousarray(rgba).data, W, H, 4*W, QImage.Format_RGBA8888).copy()
        pm = QPixmap.fromImage(qimg)

        # Optional colorbar (compose next to image)
        if self.show_cbar:
            bar = self._make_cbar_pixmap(H)
            comp = QImage(W + 8 + bar.width(), H, QImage.Format_ARGB32)
            comp.fill(Qt.white)
            p = QPainter(comp)
            p.drawPixmap(0, 0, pm)
            p.drawPixmap(W + 8, 0, bar)
            p.end()
            pm = QPixmap.fromImage(comp)
            self._img_rect = (0, 0, W, H)  # where the image sits in the composite
        else:
            self._img_rect = (0, 0, W, H)

        # Scene/View
        self.scene = QGraphicsScene(self)
        self.pixitem = self.scene.addPixmap(pm)
        self.view = _ZoomView(self.scene, self._value_from_pos, parent=self)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Top toolbar
        tb = QToolBar()
        act_save = QAction("Save PNG…", self); act_save.triggered.connect(self._save_png)
        act_zoom_in = QAction("Zoom In (+)", self); act_zoom_in.triggered.connect(lambda: self.view.zoom(1.25))
        act_zoom_out = QAction("Zoom Out (–)", self); act_zoom_out.triggered.connect(lambda: self.view.zoom(0.8))
        act_reset = QAction("Reset (Ctrl+0)", self); act_reset.triggered.connect(self.view.reset_view)
        tb.addAction(act_save); tb.addSeparator(); tb.addAction(act_zoom_in); tb.addAction(act_zoom_out); tb.addAction(act_reset)

        # Status label (value readout)
        self.status = QLabel("Move mouse over the image…")
        self.status.setStyleSheet("color: #444;")

        # Layout
        lay = QVBoxLayout(self)
        lay.addWidget(tb)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.status)

        # Fit initially
        self.view.fitInView(self.pixitem, Qt.KeepAspectRatio)

    # value callback used by the view
    def _value_from_pos(self, scene_pos: QPointF):
        x, y, W, H = self._img_rect
        px = scene_pos.x() - x
        py = scene_pos.y() - y
        if px < 0 or py < 0 or px >= W or py >= H:
            self.status.setText("—")
            return None
        ix = int(px)
        iy = int(py)
        # Clamp
        H0, W0 = self.img_linear.shape
        if ix >= W0 or iy >= H0:
            self.status.setText("—")
            return None
        val = self.img_linear[iy, ix]
        self.status.setText(f"x={ix}, y={iy}, value={val:.6g}")
        return val

    def _make_cbar_pixmap(self, height):
        # Create labeled colorbar matching vmin/vmax, linear labels even for Log10
        import matplotlib.cm as cm
        lut = (cm.get_cmap(self.cmap)(np.linspace(0,1,256)) * 255.0).astype(np.uint8)
        rgba = np.ascontiguousarray(lut[::-1, :])
        base = QPixmap.fromImage(QImage(rgba.data, 1, 256, 4*1, QImage.Format_RGBA8888).copy())
        bar = base.scaled(20, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        # Label bar
        total_w = 60
        qimg = QImage(total_w, height, QImage.Format_ARGB32)
        qimg.fill(Qt.white)
        p = QPainter(qimg)
        p.drawPixmap(total_w - bar.width(), 0, bar)

        def _fmt(v):
            if not np.isfinite(v) or v == 0: return "0"
            e = int(np.floor(np.log10(abs(v))))
            return f"{v:.0e}" if e >= 3 or e <= -3 else f"{v:.3g}"

        if self.scale_mode == "Log10":
            lo_val = 10.0 ** float(self.vmin)
            hi_val = 10.0 ** float(self.vmax)
        else:
            lo_val, hi_val = float(self.vmin), float(self.vmax)

        f = QFont(); f.setPointSize(9); p.setFont(f); p.setPen(Qt.black)
        p.drawText(0, 0, total_w- bar.width() - 4, 16, Qt.AlignLeft | Qt.AlignTop, _fmt(hi_val))
        p.drawText(0, height-16, total_w- bar.width() - 4, 16, Qt.AlignLeft | Qt.AlignBottom, _fmt(lo_val))
        p.end()
        return QPixmap.fromImage(qimg)

    def _save_png(self):
        fd = QFileDialog(self)
        fd.setOption(QFileDialog.DontUseNativeDialog, True)
        fd.setAcceptMode(QFileDialog.AcceptSave)
        fd.setNameFilter("PNG Images (*.png)")
        fd.selectFile(f"{self.windowTitle()}.png")
        if fd.exec() != QFileDialog.Accepted:
            return
        path = fd.selectedFiles()[0]
        if not path.lower().endswith(".png"):
            path += ".png"

        # Re-render at high quality (no UI chrome)
        disp_arr = np.log10(np.clip(self.img_linear, self.eps, None)) if self.scale_mode == "Log10" else self.img_linear
        import matplotlib.cm as cm
        lut = (cm.get_cmap(self.cmap)(np.linspace(0,1,256)) * 255.0).astype(np.uint8)
        norm = (disp_arr - self.vmin) / (self.vmax - self.vmin + 1e-12)
        idx = np.clip((norm * 255.0).round().astype(np.uint8), 0, 255)
        rgba = lut[idx]
        H, W = rgba.shape[:2]
        img = QImage(np.ascontiguousarray(rgba).data, W, H, 4*W, QImage.Format_RGBA8888).copy()

        if self.show_colorbar:
            bar = self._make_cbar_pixmap(H)
            comp = QImage(W + 8 + bar.width(), H, QImage.Format_ARGB32)
            comp.fill(Qt.white)
            p = QPainter(comp)
            p.drawImage(0, 0, img)
            p.drawPixmap(W + 8, 0, bar)
            p.end()
            comp.save(path, "PNG")
        else:
            img.save(path, "PNG")

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


