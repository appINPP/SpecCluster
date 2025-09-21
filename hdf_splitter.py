# hdf_splitter.py — RAM-safe splitter/visualizer for large HDF5 imaging cubes (H, W, C).
#
# GUI highlights:
# - Preview: channel-summed heatmap with stride {1,2,4,8} for speed.
# - Scaling: Robust (p1–p99, default), Auto (min–max), Fixed (vmin/vmax shown only when selected).
# - ROI mode: rectangle selection (cleared when leaving the mode).
# - Grid mode: Nx×Ny grid with **draggable** lines; safeguards prevent crossovers (<1 px tiles).
# - Estimator: **compression-aware** via dataset storage size; per-tile min/avg/max for custom grids.
# - Export: ROI → .h5 (+ overview PNG); Grid tiles → dir (+ overview PNG). PNGs use active scaling.
# - Zoom: right-drag to zoom, right double-click to reset.
#
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import h5py

# Headless matplotlib for non-GUI PNGs
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg

# Optional GUI imports
try:
    from PySide6.QtCore import Qt, QThread, Signal, QObject
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
        QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox,
        QDoubleSpinBox, QGroupBox, QRadioButton, QGridLayout, QLineEdit, QCheckBox
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvasQt
    from matplotlib.widgets import RectangleSelector
    HAVE_GUI = True
except Exception:
    HAVE_GUI = False


# ============================= Core splitter =============================

class HDFSplitter:
    """HDF5 splitter/preview helper for 3D imaging datasets (H, W, C). RAM-safe."""

    def __init__(self, path: str | os.PathLike, ds_path: Optional[str] = None):
        self.path = str(path)
        self.file = h5py.File(self.path, "r")
        self.ds: Optional[h5py.Dataset] = None
        if ds_path is not None:
            self.use_dataset(ds_path)

    # ---- discovery ----
    def list_3d_datasets(self) -> List[str]:
        out = []

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                out.append("/" + name if not name.startswith("/") else name)

        self.file.visititems(lambda name, obj: _visit(name, obj))
        return out

    def use_dataset(self, ds_path: str):
        if ds_path not in self.file:
            raise KeyError(f"Dataset not found: {ds_path}")
        ds = self.file[ds_path]
        if not isinstance(ds, h5py.Dataset) or ds.ndim != 3:
            raise ValueError(f"Dataset {ds_path} must be 3D (H, W, C). Found shape={ds.shape}.")
        self.ds = ds

    # ---- basic props ----
    @property
    def shape(self) -> Tuple[int, int, int]:
        if self.ds is None:
            raise RuntimeError("Dataset not selected.")
        return tuple(int(x) for x in self.ds.shape)  # (H, W, C)

    @property
    def dtype(self):
        if self.ds is None:
            raise RuntimeError("Dataset not selected.")
        return self.ds.dtype

    # ---- preview (RAM-safe, stridable) ----
    def preview_sum(self, ch_lo: int = 0, ch_hi: int = -1, block_c: int = 256, stride: int = 1) -> np.ndarray:
        """
        Returns float32 (ceil(H/stride), ceil(W/stride)) heatmap = sum over channels [ch_lo, ch_hi] inclusive,
        reading channels in blocks and pixels with strides for speed.
        """
        H, W, C = self.shape
        if ch_hi < 0:
            ch_hi = C - 1
        ch_lo = max(0, min(C - 1, ch_lo))
        ch_hi = max(0, min(C - 1, ch_hi))
        if ch_lo > ch_hi:
            ch_lo, ch_hi = ch_hi, ch_lo

        s = max(1, int(stride))
        Hs = math.ceil(H / s)
        Ws = math.ceil(W / s)
        out = np.zeros((Hs, Ws), dtype=np.float32)

        for c0 in range(ch_lo, ch_hi + 1, block_c):
            c1 = min(ch_hi + 1, c0 + block_c)
            slab = self.ds[0:H:s, 0:W:s, c0:c1]  # strided hyperslab
            out += np.asarray(slab, dtype=np.float32).sum(axis=2, dtype=np.float32)

        return out

    # ---- compression awareness ----
    def approx_compression_ratio(self) -> float:
        """
        Returns ~compressed_size/raw_size for the **source dataset on disk**.
        If dataset is uncompressed or unknown, returns 1.0.
        """
        if self.ds is None:
            return 1.0
        H, W, C = self.shape
        itemsize = np.dtype(self.dtype).itemsize
        raw = H * W * C * itemsize
        if raw <= 0:
            return 1.0
        try:
            used = float(self.ds.id.get_storage_size())  # bytes on disk (compressed + overhead)
            return max(used / raw, 1e-6)
        except Exception:
            return 1.0

    # ---- export helpers ----
    def _copy_attrs(self, src: h5py.Dataset, dst: h5py.Dataset):
        for k, v in src.attrs.items():
            try:
                dst.attrs[k] = v
            except Exception:
                pass

    def _dst_kwargs_like(self, src: h5py.Dataset, *, shape: Optional[Tuple[int, int, int]] = None):
        """
        Carry over compression/shuffle/fletcher32. If copying chunking, clamp each
        chunk dimension to the corresponding output dimension so tiny tiles/ROIs remain valid.
        """
        kw = {}
        for k in ("compression", "compression_opts", "shuffle", "fletcher32"):
            try:
                v = getattr(src, k)
            except Exception:
                v = None
            if v is not None:
                kw[k] = v

        # chunks handled separately so we can clamp
        try:
            ch = getattr(src, "chunks")
        except Exception:
            ch = None
        if ch is not None:
            if shape is not None:
                sh = tuple(int(x) for x in shape)
                ch = tuple(max(1, min(int(c), int(s))) for c, s in zip(ch, sh))
            kw["chunks"] = ch
        return kw

    def _roi_slices(self, x0, x1, y0, y1) -> Tuple[slice, slice]:
        H, W, _ = self.shape
        x0, x1 = int(max(0, min(W, x0))), int(max(0, min(W, x1)))
        y0, y1 = int(max(0, min(H, y0))), int(max(0, min(H, y1)))
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        return slice(y0, y1), slice(x0, x1)  # inclusive→exclusive via ceil/floor in GUI

    def export_roi(
        self,
        out_path: str | os.PathLike,
        x0: int, x1: int, y0: int, y1: int,
        ch_lo: int = 0, ch_hi: int = -1,
        block_c: int = 256, block_y: int = 512,
        ds_name: Optional[str] = None,
        keep_layout: bool = True,
    ):
        """Write a rectangular subset to out_path with shape (roiH, roiW, Csel)."""
        if self.ds is None:
            raise RuntimeError("Dataset not selected.")

        H, W, C = self.shape
        if ch_hi < 0:
            ch_hi = C - 1
        ch_lo = max(0, min(C - 1, ch_lo))
        ch_hi = max(0, min(C - 1, ch_hi))
        if ch_lo > ch_hi:
            ch_lo, ch_hi = ch_hi, ch_lo

        sly, slx = self._roi_slices(x0, x1, y0, y1)
        roiH = sly.stop - sly.start
        roiW = slx.stop - slx.start
        Csel = ch_hi - ch_lo + 1
        if roiH <= 0 or roiW <= 0:
            raise ValueError("Empty ROI.")

        ds_name = ds_name or self.ds.name.split("/")[-1]

        with h5py.File(str(out_path), "w") as fout:
            kw = self._dst_kwargs_like(self.ds, shape=(roiH, roiW, Csel)) if keep_layout else {}
            dso = fout.create_dataset(f"{ds_name}", shape=(roiH, roiW, Csel), dtype=self.dtype, **kw)
            self._copy_attrs(self.ds, dso)

            for y0b in range(0, roiH, block_y):
                y1b = min(roiH, y0b + block_y)
                for c0 in range(ch_lo, ch_hi + 1, block_c):
                    c1 = min(ch_hi + 1, c0 + block_c)
                    src = self.ds[sly.start + y0b : sly.start + y1b, slx, c0:c1]
                    dso[y0b:y1b, :, c0 - ch_lo : c1 - ch_lo] = src

    def export_grid(
        self,
        out_dir: str | os.PathLike,
        nx: int,
        ny: int,
        ch_lo: int = 0,
        ch_hi: int = -1,
        base: Optional[str] = None,
        x_edges: Optional[np.ndarray] = None,
        y_edges: Optional[np.ndarray] = None,
    ):
        """
        Split the full field into tiles. If x_edges/y_edges are given, they are used as custom
        boundaries (length nx+1 / ny+1, integers, monotonic). Otherwise, tiles are equal-sized.
        """
        H, W, C = self.shape
        if ch_hi < 0:
            ch_hi = C - 1

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = base or Path(self.path).stem

        if x_edges is None or y_edges is None:
            tile_w = math.ceil(W / nx)
            tile_h = math.ceil(H / ny)
            x_edges = np.array([min(W, i * tile_w) for i in range(nx)] + [W], dtype=int)
            y_edges = np.array([min(H, j * tile_h) for j in range(ny)] + [H], dtype=int)
        else:
            x_edges = np.asarray(x_edges, dtype=int)
            y_edges = np.asarray(y_edges, dtype=int)

        for j in range(ny):
            for i in range(nx):
                x0 = int(x_edges[i]); x1 = int(x_edges[i+1])
                y0 = int(y_edges[j]); y1 = int(y_edges[j+1])
                if x0 >= x1 or y0 >= y1:
                    continue
                out_path = out_dir / f"{base}_x{i:02d}y{j:02d}.h5"
                self.export_roi(out_path, x0, x1, y0, y1, ch_lo=ch_lo, ch_hi=ch_hi)

    def suggest_grid_from_target_mb(self, target_mb: float, ch_lo: int, ch_hi: int) -> Tuple[int, int]:
        """
        Return (nx, ny) such that each tile ≈ target_mb **on disk**, using the source dataset's
        measured compression ratio and the **selected channel range**.
        """
        H, W, C = self.shape
        itemsize = np.dtype(self.dtype).itemsize
        ch_lo = max(0, min(C - 1, int(ch_lo)))
        ch_hi = C - 1 if int(ch_hi) < 0 else max(0, min(C - 1, int(ch_hi)))
        if ch_lo > ch_hi:
            ch_lo, ch_hi = ch_hi, ch_lo
        Csel = (ch_hi - ch_lo + 1)

        bytes_total_raw = H * W * Csel * itemsize
        ratio = self.approx_compression_ratio()  # ~compressed/raw
        bytes_total_est = bytes_total_raw * ratio

        target_bytes = max(1.0, float(target_mb) * (1024**2))
        tiles = max(1, int(math.ceil(bytes_total_est / target_bytes)))
        # Aspect-aware: split roughly proportional to W/H
        nx = int(math.ceil(math.sqrt(tiles * (W / max(1.0, H)))))
        ny = int(math.ceil(tiles / max(1, nx)))
        return max(1, nx), max(1, ny)

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


# ============================= Helpers for overview PNGs =============================

def _save_overview_png(
    heatmap: np.ndarray,
    path: os.PathLike,
    *,
    title: str = "",
    grid: Optional[Tuple[int, int]] = None,
    grid_edges: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    clim: Optional[Tuple[float, float]] = None,
    dpi: int = 170,
    full_shape: Optional[Tuple[int, int]] = None,  # (H_full, W_full)
):
    """
    Save a PNG with optional grid/ROI overlay.

    If preview is strided/downsampled, pass `full_shape=(H_full, W_full)` so
    overlays (which are in full-res pixel coords) line up.
    """
    Hh, Wh = heatmap.shape  # downsampled heatmap size
    if full_shape is None:
        H, W = Hh, Wh
    else:
        H, W = full_shape

    fig = Figure(figsize=(6, 5), constrained_layout=True)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        heatmap, cmap="inferno", origin="upper",
        interpolation="nearest", extent=(0, W, H, 0)  # key: full-res extent
    )
    if clim is not None:
        im.set_clim(*clim)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if clim is not None:
        cbar.update_normal(im)

    # Grid overlay
    if grid_edges is not None:
        x_edges, y_edges = grid_edges
        x_edges = np.asarray(x_edges); y_edges = np.asarray(y_edges)
        nx = len(x_edges) - 1; ny = len(y_edges) - 1
        for i in range(1, nx):
            ax.vlines(float(x_edges[i]), 0, H, colors=(1, 1, 1, 0.55), linewidth=1.0)
        for j in range(1, ny):
            ax.hlines(float(y_edges[j]), 0, W, colors=(1, 1, 1, 0.55), linewidth=1.0)
        for j in range(ny):
            for i in range(nx):
                cx = 0.5 * (x_edges[i] + x_edges[i + 1])
                cy = 0.5 * (y_edges[j] + y_edges[j + 1])
                ax.text(cx, cy, f"x{i}y{j}", color="w", fontsize=8, ha="center", va="center",
                        bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", pad=1.5))
    elif grid is not None:
        nx, ny = grid
        tile_w = W / nx; tile_h = H / ny
        for i in range(1, nx):
            ax.vlines(i * tile_w, 0, H, colors=(1, 1, 1, 0.55), linewidth=1.0)
        for j in range(1, ny):
            ax.hlines(j * tile_h, 0, W, colors=(1, 1, 1, 0.55), linewidth=1.0)
        for j in range(ny):
            for i in range(nx):
                cx = (i + 0.5) * tile_w; cy = (j + 0.5) * tile_h
                ax.text(cx, cy, f"x{i}y{j}", color="w", fontsize=8, ha="center", va="center",
                        bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", pad=1.5))

    if roi is not None:
        x0, x1, y0, y1 = roi
        import matplotlib.patches as patches
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.8, ec=(1, 1, 1, 0.8)))

    fig.savefig(str(path), dpi=dpi)
    fig.clear()

# ============================= GUI (optional) =============================

if HAVE_GUI:

    class Worker(QObject):
        """Generic worker that runs a function in a QThread and emits finished(result, error)."""
        finished = Signal(object, str)  # (result, error)

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs

        def run(self):
            try:
                res = self.fn(*self.args, **self.kwargs)
                self.finished.emit(res, "")
            except Exception as e:
                self.finished.emit(None, str(e))

    class SplitterApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("HDF5 Tiler / ROI Export")
            self.resize(1280, 860)

            self.splitter: Optional[HDFSplitter] = None
            self.ds_path: Optional[str] = None

            self.rect_sel: Optional[RectangleSelector] = None
            self.zoom_sel: Optional[RectangleSelector] = None
            self.roi_xy: Optional[Tuple[int, int, int, int]] = None  # (x0,x1,y0,y1) ints
            self.C = 0

            # Grid overlay state
            self._grid_artists = []   # generic list to clear
            self._grid_vlines = []    # vertical Line2D (for dragging)
            self._grid_hlines = []    # horizontal Line2D (for dragging)
            self._grid_edges_x: Optional[np.ndarray] = None  # len nx+1
            self._grid_edges_y: Optional[np.ndarray] = None  # len ny+1
            self._dragging_kind = None  # ('v', idx) or ('h', idx) where idx in 1..N-1

            # Matplotlib artists/state
            self._im = None
            self._cbar = None
            self._last_preview = None
            self._preview_stride = 1
            self._full_xlim = None
            self._full_ylim = None

            # overlays
            self._roi_artist = None      # Rectangle patch for ROI
            self._roi_rect = None        # (x0, y0, x1, y1) floats

            # Threads
            self._threads: List[QThread] = []
            self._current_worker: Optional[Worker] = None
            self._current_thread: Optional[QThread] = None
            self._busy: bool = False

            # Build UI
            central = QWidget()
            self.setCentralWidget(central)
            v = QVBoxLayout(central)

            # --- Top: file/dataset ---
            top = QHBoxLayout()
            self.in_edit = QLineEdit()
            self.in_edit.setPlaceholderText("Open an .h5/.hdf5 file…")
            btn_open = QPushButton("Open…")
            btn_open.clicked.connect(self._open_file)
            self.ds_combo = QComboBox()
            self.ds_combo.currentTextChanged.connect(self._choose_ds)
            top.addWidget(QLabel("File:"))
            top.addWidget(self.in_edit, 1)
            top.addWidget(btn_open)
            top.addSpacing(10)
            top.addWidget(QLabel("Dataset:"))
            top.addWidget(self.ds_combo, 1)
            v.addLayout(top)

            # --- Preview controls ---
            pr = QHBoxLayout()
            self.ch_lo = QSpinBox(); self.ch_lo.setRange(0, 0); self.ch_lo.setValue(0)
            self.ch_hi = QSpinBox(); self.ch_hi.setRange(0, 0); self.ch_hi.setValue(0)
            self.btn_preview = QPushButton("Compute preview")
            self.btn_preview.clicked.connect(self._compute_preview)

            pr.addWidget(QLabel("Chan lo:")); pr.addWidget(self.ch_lo)
            pr.addWidget(QLabel("hi:")); pr.addWidget(self.ch_hi)

            pr.addSpacing(10)
            pr.addWidget(QLabel("Preview stride:"))
            self.stride_combo = QComboBox()
            self.stride_combo.addItems(["1 (full)", "2", "4", "8"])
            self.stride_combo.setCurrentIndex(0)
            self.stride_combo.currentIndexChanged.connect(self._on_stride_changed)
            pr.addWidget(self.stride_combo)

            pr.addSpacing(10)
            pr.addWidget(QLabel("Scale:"))
            self.scale_mode = QComboBox()
            self.scale_mode.addItems(["Auto (min–max)", "Robust (p1–p99)", "Fixed"])
            self.scale_mode.setCurrentIndex(1)  # Robust by default
            self.scale_mode.currentIndexChanged.connect(self._apply_scale_to_current_preview)
            pr.addWidget(self.scale_mode)

            self.vmin_label = QLabel("vmin:")
            self.vmin_spin = QDoubleSpinBox(); self.vmin_spin.setDecimals(6); self.vmin_spin.setRange(-1e30, 1e30)
            self.vmax_label = QLabel("vmax:")
            self.vmax_spin = QDoubleSpinBox(); self.vmax_spin.setDecimals(6); self.vmax_spin.setRange(-1e30, 1e30)
            self.vmin_spin.valueChanged.connect(self._apply_scale_to_current_preview)
            self.vmax_spin.valueChanged.connect(self._apply_scale_to_current_preview)
            pr.addWidget(self.vmin_label); pr.addWidget(self.vmin_spin)
            pr.addWidget(self.vmax_label); pr.addWidget(self.vmax_spin)

            pr.addSpacing(10)
            pr.addWidget(self.btn_preview)
            pr.addStretch(1)
            v.addLayout(pr)
            self._sync_fixed_boxes_visibility()

            # --- Canvas ---
            from matplotlib.figure import Figure as FigureQt
            self.fig = FigureQt(figsize=(6.8, 5.6), constrained_layout=True)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasQt(self.fig)
            v.addWidget(self.canvas, 1)

            # Mouse connections
            self._press_cid = self.canvas.mpl_connect("button_press_event", self._on_button_press)
            self._release_cid = self.canvas.mpl_connect("button_release_event", self._on_button_release)
            self._motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

            # --- Mode/overlay box ---
            box = QGroupBox("Export / Overlay")
            grid = QGridLayout(box)
            self.rb_roi = QRadioButton("Rectangle ROI")
            self.rb_roi.setChecked(True)
            self.rb_grid = QRadioButton("Equal grid (Nx×Ny)")
            self.rb_roi.toggled.connect(self._refresh_overlay_visibility)
            self.rb_grid.toggled.connect(self._refresh_overlay_visibility)

            self.nx = QSpinBox(); self.nx.setRange(1, 4096); self.nx.setValue(4)
            self.nx.valueChanged.connect(self._on_grid_params_changed)
            self.ny = QSpinBox(); self.ny.setRange(1, 4096); self.ny.setValue(4)
            self.ny.valueChanged.connect(self._on_grid_params_changed)
            self.target_mb = QDoubleSpinBox(); self.target_mb.setRange(1, 32768); self.target_mb.setValue(400.0); self.target_mb.setDecimals(1)

            self.btn_auto_grid = QPushButton("Auto grid from target MB")
            self.btn_auto_grid.clicked.connect(self._auto_grid)

            self.cb_show_grid = QCheckBox("Show grid overlay"); self.cb_show_grid.setChecked(True)
            self.cb_show_grid.toggled.connect(self._draw_grid_overlay)

            grid.addWidget(self.rb_roi, 0, 0, 1, 2)
            grid.addWidget(self.rb_grid, 0, 2, 1, 2)
            grid.addWidget(self.cb_show_grid, 1, 2, 1, 2)

            grid.addWidget(QLabel("Nx:"), 2, 2); grid.addWidget(self.nx, 2, 3)
            grid.addWidget(QLabel("Ny:"), 3, 2); grid.addWidget(self.ny, 3, 3)
            grid.addWidget(QLabel("Target MB/tile:"), 4, 2); grid.addWidget(self.target_mb, 4, 3)
            grid.addWidget(self.btn_auto_grid, 5, 2, 1, 2)
            v.addWidget(box)

            # --- Estimator + Actions ---
            bottom = QHBoxLayout()
            self.est_label = QLabel("Est. output: –")
            self.btn_export = QPushButton("Export…")
            self.btn_export.clicked.connect(self._export)
            bottom.addWidget(self.est_label, 1)
            bottom.addStretch(1)
            bottom.addWidget(self.btn_export)
            v.addLayout(bottom)

            self._set_rect_selector(active=False)
            self._set_zoom_selector(active=False)
            self._refresh_overlay_visibility()

        # ===================== File/dataset =====================

        def _open_file(self):
            path, _ = QFileDialog.getOpenFileName(self, "Open HDF5 file", "", "HDF5 files (*.h5 *.hdf5 *.hdf)")
            if not path:
                return
            self.in_edit.setText(path)
            try:
                if self.splitter:
                    self.splitter.close()
                self.splitter = HDFSplitter(path)
                ds_list = self.splitter.list_3d_datasets()
                self.ds_combo.blockSignals(True)
                self.ds_combo.clear()
                self.ds_combo.addItems(ds_list)
                self.ds_combo.blockSignals(False)
                if ds_list:
                    self.ds_combo.setCurrentIndex(0)
                    self._choose_ds(ds_list[0])
                else:
                    QMessageBox.warning(self, "No 3D datasets", "No 3D (H,W,C) datasets found in this file.")
            except Exception as e:
                QMessageBox.critical(self, "Open error", str(e))

        def _choose_ds(self, path: str):
            if not path:
                return
            try:
                self.splitter.use_dataset(path)
                H, W, C = self.splitter.shape
                self.C = C
                self.ch_lo.setRange(0, C - 1); self.ch_lo.setValue(0)
                self.ch_hi.setRange(0, C - 1); self.ch_hi.setValue(C - 1)

                self.ax.clear()
                self.ax.set_title(f"{path}  (H={H}, W={W}, C={C})")
                self.ax.set_axis_off()
                self._full_xlim = (0, W)
                self._full_ylim = (H, 0)
                self.canvas.draw_idle()

                self._set_rect_selector(active=False)
                self._set_zoom_selector(active=False)
                self._clear_grid_overlay()
                self._reset_grid_edges()
                self._update_estimator()
            except Exception as e:
                QMessageBox.critical(self, "Dataset error", str(e))

        # ===================== Preview =====================

        def _on_stride_changed(self):
            idx = self.stride_combo.currentIndex()
            self._preview_stride = [1, 2, 4, 8][idx]

        def _compute_preview(self):
            if not self.splitter or self._busy:
                return
            ch_lo = int(self.ch_lo.value())
            ch_hi = int(self.ch_hi.value())
            stride = int(self._preview_stride)
            self._run_in_thread(self.splitter.preview_sum, ch_lo, ch_hi, 256, stride, callback=self._show_preview)

        def _robust_limits(self, arr: np.ndarray, p_lo=1.0, p_hi=99.0, max_samples=200_000) -> Tuple[float, float]:
            a = arr
            n = a.size
            if n > max_samples:
                step = max(1, n // max_samples)
                a = a.ravel()[::step]
            vmin = float(np.nanmin(a))
            vmax = float(np.nanmax(a))
            try:
                qlo = float(np.percentile(a, p_lo))
                qhi = float(np.percentile(a, p_hi))
                if qhi > qlo:
                    vmin, vmax = qlo, qhi
            except Exception:
                pass
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1.0
            return vmin, vmax

        def _current_clim_for_array(self, arr: np.ndarray) -> Tuple[float, float]:
            mode = self.scale_mode.currentIndex()
            if mode == 0:  # Auto min-max
                vmin = float(np.nanmin(arr))
                vmax = float(np.nanmax(arr))
            elif mode == 1:  # Robust
                vmin, vmax = self._robust_limits(arr, 1.0, 99.0)
            else:  # Fixed
                vmin = float(self.vmin_spin.value())
                vmax = float(self.vmax_spin.value())
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin0 = float(np.nanmin(arr)); vmax0 = float(np.nanmax(arr))
                    vmin, vmax = vmin0, max(vmin0 + 1e-9, vmax0)
            return vmin, vmax

        def _apply_scale_to_current_preview(self):
            self._sync_fixed_boxes_visibility()
            if self._im is None or self._last_preview is None:
                return
            vmin, vmax = self._current_clim_for_array(self._last_preview)
            self._im.set_clim(vmin, vmax)
            if self._cbar is not None:
                self._cbar.update_normal(self._im)
            self.canvas.draw_idle()

        def _sync_fixed_boxes_visibility(self):
            fixed = (self.scale_mode.currentIndex() == 2)
            for w in (self.vmin_label, self.vmin_spin, self.vmax_label, self.vmax_spin):
                w.setVisible(fixed)

        def _show_preview(self, arr: Optional[np.ndarray], err: str):
            if err:
                QMessageBox.critical(self, "Preview error", err)
                return
            self._last_preview = arr
            # extents (full-res coords even if preview downsampled)
            try:
                H, W, _ = self.splitter.shape
            except Exception:
                H, W = arr.shape[0], arr.shape[1]
            self._full_xlim = (0, W)
            self._full_ylim = (H, 0)

            self.ax.clear()
            self._im = self.ax.imshow(
                arr, cmap="inferno", origin="upper", interpolation="nearest", extent=(0, W, H, 0)
            )
            self.ax.set_xlim(*self._full_xlim)
            self.ax.set_ylim(*self._full_ylim)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_title(f"Preview (stride={self._preview_stride})")

            # single persistent colorbar
            if self._cbar is None:
                self._cbar = self.fig.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.03)
            else:
                self._cbar.update_normal(self._im)

            self.ax.set_axis_on()

            # Initialize fixed inputs if Fixed selected and blank
            if self.scale_mode.currentIndex() == 2 and (self.vmin_spin.value() == 0 and self.vmax_spin.value() == 0):
                vmin0 = float(np.nanmin(arr)); vmax0 = float(np.nanmax(arr))
                self.vmin_spin.blockSignals(True); self.vmax_spin.blockSignals(True)
                self.vmin_spin.setValue(vmin0); self.vmax_spin.setValue(max(vmin0 + 1e-9, vmax0))
                self.vmin_spin.blockSignals(False); self.vmax_spin.blockSignals(False)
            self._apply_scale_to_current_preview()

            # re-enable selectors
            self._set_rect_selector(active=self.rb_roi.isChecked())
            self._set_zoom_selector(active=True)
            self._draw_grid_overlay()  # if grid is on
            self.canvas.draw_idle()
            self._update_estimator()

        # ===================== ROI selector =====================

        def _set_rect_selector(self, active: bool):
            # tear down previous
            try:
                if self.rect_sel is not None:
                    self.rect_sel.set_active(False)
                    self.rect_sel.disconnect_events()
            except Exception:
                pass
            self.rect_sel = None
            if not active or self._im is None:
                return
            self.rect_sel = RectangleSelector(
                self.ax, self._on_roi_select, useblit=True, button=[1],
                minspanx=2, minspany=2, spancoords="data", interactive=False, drag_from_anywhere=False
            )

        def _on_roi_select(self, e0, e1):
            if any(v is None for v in (e0.xdata, e0.ydata, e1.xdata, e1.ydata)):
                return
            x0, x1 = sorted([float(e0.xdata), float(e1.xdata)])
            y0, y1 = sorted([float(e0.ydata), float(e1.ydata)])
            self._roi_rect = (x0, y0, x1, y1)
            # store integer ROI for export/estimator
            xi0 = int(max(self._full_xlim[0], math.floor(x0)))
            xi1 = int(min(self._full_xlim[1], math.ceil (x1)))
            yi0 = int(max(self._full_ylim[1], math.floor(y0)))
            yi1 = int(min(self._full_ylim[0], math.ceil (y1)))
            self.roi_xy = (xi0, xi1, yi0, yi1)
            self._draw_roi_artist()
            self._update_estimator()

        def _draw_roi_artist(self):
            # clear old
            if self._roi_artist is not None:
                try: self._roi_artist.remove()
                except Exception: pass
                self._roi_artist = None
            if not self._roi_rect:
                self.canvas.draw_idle(); return
            import matplotlib as mpl
            x0, y0, x1, y1 = self._roi_rect
            if x1 <= x0 or y1 <= y0:
                return
            self._roi_artist = mpl.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.6, ec=(1, 1, 1, 0.9), zorder=9
            )
            self.ax.add_patch(self._roi_artist)
            self.canvas.draw_idle()

        def _remove_roi_artist(self):
            art = self._roi_artist
            if art is not None:
                try: art.remove()
                except Exception:
                    try: art.set_visible(False)
                    except Exception: pass
            self._roi_artist = None
            self._roi_rect = None
            self.canvas.draw_idle()

        # ===================== Zoom selector (right-drag) =====================

        def _set_zoom_selector(self, active: bool):
            try:
                if self.zoom_sel is not None:
                    self.zoom_sel.set_active(False)
                    self.zoom_sel.disconnect_events()
            except Exception:
                pass
            self.zoom_sel = None
            if not active or self._im is None:
                return
            self.zoom_sel = RectangleSelector(
                self.ax, self._on_zoom_rect, useblit=True, button=[3],
                minspanx=2, minspany=2, spancoords="data", interactive=False, drag_from_anywhere=False
            )

        def _on_zoom_rect(self, e0, e1):
            if e0 is None or e1 is None: return
            if None in (e0.xdata, e1.xdata, e0.ydata, e1.ydata): return
            x0, x1 = sorted([e0.xdata, e1.xdata])
            y0, y1 = sorted([e0.ydata, e1.ydata])
            if x0 == x1:
                xr = self.ax.get_xlim(); pad = max(1e-9, 0.02*(xr[1]-xr[0])); x0, x1 = (x0-pad, x1+pad)
            if y0 == y1:
                yr = self.ax.get_ylim(); pad = max(1e-9, 0.02*abs(yr[1]-yr[0])); y0, y1 = (y0-pad, y1+pad)
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y1, y0)  # origin='upper'
            self.canvas.draw_idle()

        def _maybe_reset_zoom(self, event):
            if event is None or event.inaxes is not self.ax: return
            if getattr(event, "dblclick", False) and event.button == 3:
                if self._full_xlim and self._full_ylim:
                    self.ax.set_xlim(*self._full_xlim)
                    self.ax.set_ylim(*self._full_ylim)
                    self.canvas.draw_idle()

        # hook right double-click + grid-line picking
        def _on_button_press(self, event):
            self._maybe_reset_zoom(event)
            if event is None or event.inaxes is not self.ax:
                return
            if event.button != 1:
                return
            if not (self._im and self.rb_grid.isChecked() and self.cb_show_grid.isChecked()):
                return
            # try pick a grid line (we set picker=6 on lines)
            for i, line in enumerate(self._grid_vlines):
                if line is not None and line.contains(event)[0]:
                    self._dragging_kind = ('v', i + 1)  # corresponds to interior edge index
                    return
            for j, line in enumerate(self._grid_hlines):
                if line is not None and line.contains(event)[0]:
                    self._dragging_kind = ('h', j + 1)
                    return

        def _on_button_release(self, event):
            self._dragging_kind = None

        def _on_mouse_move(self, event):
            # motion events have event.button=None; rely on _dragging_kind
            if self._dragging_kind is None:
                return
            if event is None or event.inaxes is not self.ax:
                return
            kind, idx = self._dragging_kind  # idx is interior edge index (1..N-1)
            if kind == 'v':
                x = int(round(max(self._grid_edges_x[idx - 1] + 1,
                                  min(self._grid_edges_x[idx + 1] - 1, event.xdata))))
                if x != self._grid_edges_x[idx]:
                    self._grid_edges_x[idx] = x
            else:
                y = int(round(max(self._grid_edges_y[idx - 1] + 1,
                                  min(self._grid_edges_y[idx + 1] - 1, event.ydata))))
                if y != self._grid_edges_y[idx]:
                    self._grid_edges_y[idx] = y
            self._draw_grid_overlay(update_only=True)
            self._update_estimator()

        # ===================== Grid overlay =====================

        def _reset_grid_edges(self):
            if not self.splitter:
                self._grid_edges_x = None
                self._grid_edges_y = None
                return
            H, W, _ = self.splitter.shape
            nx, ny = int(self.nx.value()), int(self.ny.value())
            tile_w = math.ceil(W / max(1, nx))
            tile_h = math.ceil(H / max(1, ny))
            self._grid_edges_x = np.array([min(W, i * tile_w) for i in range(nx)] + [W], dtype=int)
            self._grid_edges_y = np.array([min(H, j * tile_h) for j in range(ny)] + [H], dtype=int)

        def _clear_grid_overlay(self):
            for a in self._grid_artists:
                try: a.remove()
                except Exception: pass
            self._grid_artists = []
            self._grid_vlines = []
            self._grid_hlines = []
            self.canvas.draw_idle()

        def _draw_grid_overlay(self, update_only: bool = False):
            if not self._im:
                return
            if not self.cb_show_grid.isChecked() or not self.rb_grid.isChecked():
                self._clear_grid_overlay()
                return

            if self._grid_edges_x is None or self._grid_edges_y is None:
                self._reset_grid_edges()

            x_edges = np.clip(self._grid_edges_x, 0, self._full_xlim[1])
            y_edges = np.clip(self._grid_edges_y, 0, self._full_ylim[0])
            for i in range(1, len(x_edges)):
                if x_edges[i] <= x_edges[i - 1]:
                    x_edges[i] = x_edges[i - 1] + 1
            for j in range(1, len(y_edges)):
                if y_edges[j] <= y_edges[j - 1]:
                    y_edges[j] = y_edges[j - 1] + 1
            self._grid_edges_x = x_edges
            self._grid_edges_y = y_edges

            nx, ny = int(self.nx.value()), int(self.ny.value())

            if not update_only:
                self._clear_grid_overlay()

                # create new lines (skip outer borders)
                self._grid_vlines = []
                for i in range(1, nx):
                    x = float(x_edges[i])
                    l = self.ax.axvline(x, color=(1, 1, 1, 0.85), lw=1.4, zorder=3, picker=6)
                    self._grid_artists.append(l); self._grid_vlines.append(l)
                self._grid_hlines = []
                for j in range(1, ny):
                    y = float(y_edges[j])
                    l = self.ax.axhline(y, color=(1, 1, 1, 0.85), lw=1.4, zorder=3, picker=6)
                    self._grid_artists.append(l); self._grid_hlines.append(l)

                # labels
                for j in range(ny):
                    for i in range(nx):
                        cx = 0.5 * (x_edges[i] + x_edges[i + 1])
                        cy = 0.5 * (y_edges[j] + y_edges[j + 1])
                        t = self.ax.text(cx, cy, f"x{i}y{j}", color="w", fontsize=9, ha="center", va="center",
                                         zorder=4, bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", pad=1.5))
                        self._grid_artists.append(t)
            else:
                # update positions only (no duplicates)
                for i, line in enumerate(self._grid_vlines, start=1):
                    if line is not None and i < len(x_edges):
                        x = float(x_edges[i])
                        line.set_xdata([x, x])
                for j, line in enumerate(self._grid_hlines, start=1):
                    if line is not None and j < len(y_edges):
                        y = float(y_edges[j])
                        line.set_ydata([y, y])
                # update label centers
                k = 0
                for j in range(ny):
                    for i in range(nx):
                        # the first len(labels) artists at end of _grid_artists are labels, but we
                        # re-create on full draw; during update, iterate and update where possible
                        # Find next Text in _grid_artists
                        while k < len(self._grid_artists) and not hasattr(self._grid_artists[k], "set_position"):
                            k += 1
                        if k >= len(self._grid_artists):
                            break
                        cx = 0.5 * (x_edges[i] + x_edges[i + 1])
                        cy = 0.5 * (y_edges[j] + y_edges[j + 1])
                        try:
                            self._grid_artists[k].set_position((cx, cy))
                        except Exception:
                            pass
                        k += 1

            self.canvas.draw_idle()

        def _refresh_overlay_visibility(self):
            grid_mode = self.rb_grid.isChecked()
            # Toggle grid controls
            self.cb_show_grid.setEnabled(grid_mode)
            self.nx.setEnabled(grid_mode)
            self.ny.setEnabled(grid_mode)
            self.target_mb.setEnabled(grid_mode)
            self.btn_auto_grid.setEnabled(grid_mode)

            # Clear opposite overlay
            if grid_mode:
                self._set_rect_selector(active=False)
                self.roi_xy = None
                self._remove_roi_artist()
                self._reset_grid_edges()
                self._draw_grid_overlay()
            else:
                self._clear_grid_overlay()
                self._set_rect_selector(active=(self._im is not None))

            self._update_estimator()

        # ---- Grid helper ----
        def _auto_grid(self):
            if not self.splitter:
                return
            c_lo = int(self.ch_lo.value())
            c_hi = int(self.ch_hi.value())
            nx, ny = self.splitter.suggest_grid_from_target_mb(float(self.target_mb.value()), c_lo, c_hi)
            self.nx.setValue(nx)
            self.ny.setValue(ny)
            self._reset_grid_edges()
            self._draw_grid_overlay()
            self._update_estimator()

        def _on_grid_params_changed(self):
            self._reset_grid_edges()
            self._draw_grid_overlay()
            self._update_estimator()

        # ---- Estimator ----
        def _update_estimator(self):
            if not self.splitter:
                self.est_label.setText("Est. output: –")
                return
            H, W, C = self.splitter.shape
            itemsize = np.dtype(self.splitter.dtype).itemsize
            c_lo = int(self.ch_lo.value())
            c_hi = int(self.ch_hi.value())
            if c_hi < c_lo:
                c_lo, c_hi = c_hi, c_lo
            Csel = (c_hi - c_lo + 1)

            ratio = self.splitter.approx_compression_ratio()

            if self.rb_roi.isChecked() and self.roi_xy:
                x0, x1, y0, y1 = self.roi_xy
                roiW = max(0, min(W, x1) - max(0, x0))
                roiH = max(0, min(H, y1) - max(0, y0))
                bytes_raw = roiH * roiW * Csel * itemsize
                bytes_cmp = bytes_raw * ratio
                self.est_label.setText(
                    f"Est. output (ROI): ~{bytes_cmp / (1024**2):.1f} MB (raw {bytes_raw / (1024**2):.1f} MB)  "
                    f"({roiW}×{roiH}×{Csel})"
                )
            elif self.rb_grid.isChecked():
                nx, ny = int(self.nx.value()), int(self.ny.value())
                x_edges = self._grid_edges_x
                y_edges = self._grid_edges_y
                if x_edges is None or y_edges is None:
                    tile_w = math.ceil(W / max(1, nx))
                    tile_h = math.ceil(H / max(1, ny))
                    bytes_tile_raw = tile_h * tile_w * Csel * itemsize
                    bytes_tile_cmp = bytes_tile_raw * ratio
                    tiles = max(1, nx * ny)
                    self.est_label.setText(
                        f"Est. per tile: ~{bytes_tile_cmp / (1024**2):.1f} MB (raw {bytes_tile_raw / (1024**2):.1f})  "
                        f"• tiles: {tiles}  • tile size ≈ {tile_w}×{tile_h}×{Csel}"
                    )
                else:
                    sizes_raw = []
                    for j in range(ny):
                        for i in range(nx):
                            tw = int(x_edges[i+1] - x_edges[i])
                            th = int(y_edges[j+1] - y_edges[j])
                            sizes_raw.append(th * tw * Csel * itemsize)
                    sizes_raw = np.asarray(sizes_raw, dtype=float)
                    if sizes_raw.size == 0:
                        self.est_label.setText("Est. output: –"); return
                    sizes_cmp = sizes_raw * ratio
                    tiles = max(1, nx * ny)
                    txt = (
                        f"Est. per tile (compressed): min {sizes_cmp.min()/(1024**2):.1f} MB, "
                        f"avg {sizes_cmp.mean()/(1024**2):.1f} MB, max {sizes_cmp.max()/(1024**2):.1f} MB  "
                        f"(raw avg {sizes_raw.mean()/(1024**2):.1f} MB)  • tiles: {tiles}"
                    )
                    self.est_label.setText(txt)
            else:
                self.est_label.setText("Est. output: –")

        # ===================== Export =====================

        def _export(self):
            if not self.splitter or self._busy:
                return
            ch_lo = int(self.ch_lo.value())
            ch_hi = int(self.ch_hi.value())

            # Preview (may be strided) for overview PNG + clim from current scaling
            try:
                preview = self.splitter.preview_sum(ch_lo, ch_hi, 256, stride=self._preview_stride)
            except Exception:
                preview = None
            
            H, W, _ = self.splitter.shape
            clim = self._current_clim_for_array(preview) if preview is not None else None

            if self.rb_roi.isChecked():
                if not self.roi_xy:
                    QMessageBox.information(self, "ROI", "Draw a rectangle on the preview first."); return
                x0, x1, y0, y1 = self.roi_xy
                path, _ = QFileDialog.getSaveFileName(self, "Save ROI as", "roi.h5", "HDF5 files (*.h5 *.hdf5)")
                if not path:
                    return

                def job():
                    self.splitter.export_roi(path, x0, x1, y0, y1, ch_lo=ch_lo, ch_hi=ch_hi)
                    if preview is not None:
                        png = Path(path).with_suffix("").as_posix() + "_roi_preview.png"
                        _save_overview_png(preview, png, title="ROI export preview", roi=(x0, x1, y0, y1), clim=clim, full_shape=(H, W))

                def _after_export(res, err):
                    if err:
                        QMessageBox.critical(self, "Export error", err)
                    else:
                        QMessageBox.information(self, "Done", "ROI export completed.")

                self._run_in_thread(job, callback=_after_export)
                
            else:
                outdir = QFileDialog.getExistingDirectory(self, "Choose output directory")
                if not outdir:
                    return
                nx, ny = int(self.nx.value()), int(self.ny.value())
                x_edges = self._grid_edges_x
                y_edges = self._grid_edges_y

                def job():
                    self.splitter.export_grid(outdir, nx, ny, ch_lo=ch_lo, ch_hi=ch_hi,
                                              x_edges=x_edges, y_edges=y_edges)
                    if preview is not None:
                        base = Path(self.splitter.path).stem
                        png = Path(outdir) / f"{base}_grid_overview.png"
                        _save_overview_png( preview, png, title="Grid export overview",
                                            grid_edges=(x_edges, y_edges) if (x_edges is not None and y_edges is not None) else None,
                                            grid=None if (x_edges is not None and y_edges is not None) else (nx, ny),
                                            clim=clim, full_shape=(H, W))
                        
                def _after_export(res, err):
                    if err:
                        QMessageBox.critical(self, "Export error", err)
                    else:
                        QMessageBox.information(self, "Done", "ROI export completed.")

                self._run_in_thread(job, callback=_after_export)

        # ===================== Threading helper (robust) =====================

        def _run_in_thread(self, fn, *args, callback=None):
            """
            Run fn(*args) off the GUI thread. Calls `callback(result, err)` AFTER the thread
            has fully finished and the window is re-enabled.
            """
            if self._busy:
                return
            self._busy = True
            self.setEnabled(False)
            if self.statusBar():
                self.statusBar().showMessage("Working…")

            result_box = {"res": None, "err": ""}

            thread = QThread()
            worker = Worker(fn, *args)
            worker.moveToThread(thread)
            thread.started.connect(worker.run, Qt.QueuedConnection)

            def _worker_finished(res, err):
                result_box["res"] = res
                result_box["err"] = err
                thread.quit()

            worker.finished.connect(_worker_finished, Qt.QueuedConnection)

            def _thread_finished():
                try: self._threads.remove(thread)
                except ValueError: pass
                try: worker.deleteLater()
                except Exception: pass
                try: thread.deleteLater()
                except Exception: pass

                self.setEnabled(True)
                if self.statusBar(): self.statusBar().clearMessage()
                self._busy = False

                if callback:
                    try:
                        callback(result_box["res"], result_box["err"])
                    except Exception as e:
                        QMessageBox.critical(self, "Callback error", str(e))

            thread.finished.connect(_thread_finished, Qt.QueuedConnection)

            self._current_thread = thread
            self._current_worker = worker
            self._threads.append(thread)
            thread.start()

        # ===================== Close: clean shutdown =====================

        def closeEvent(self, e):
            for t in list(self._threads):
                if t and t.isRunning():
                    t.requestInterruption()
                    t.quit()
                    if not t.wait(3000):
                        t.terminate()
                        t.wait(1000)
            self._threads.clear()
            try:
                if self.splitter:
                    self.splitter.close()
            except Exception:
                pass
            super().closeEvent(e)


# ============================= CLI =============================

def _cli():
    p = argparse.ArgumentParser(description="HDF5 tiler/ROI exporter (3D datasets, shape = H×W×C).")
    p.add_argument("--in", dest="in_path", help="Input HDF5 file")
    p.add_argument("--list", action="store_true", help="List 3D datasets and exit")
    p.add_argument("--ds", dest="ds_path", help="Dataset path inside the file")
    p.add_argument("--roi", nargs=4, type=int, metavar=("x0", "x1", "y0", "y1"), help="Rectangular ROI (integers).")
    p.add_argument("--crange", nargs=2, type=int, default=[0, -1], metavar=("c_lo", "c_hi"),
                   help="Channel range inclusive (defaults to full).")
    p.add_argument("--out", help="Output .h5 path (for ROI).")
    p.add_argument("--grid", nargs=2, type=int, metavar=("nx", "ny"), help="Split into nx×ny tiles; requires --outdir.")
    p.add_argument("--auto-grid-mb", type=float, metavar="MB",
                   help="Suggest nx×ny aiming at this **compressed** size per tile; requires --outdir.")
    p.add_argument("--outdir", help="Directory to write tiles.")
    p.add_argument("--gui", action="store_true", help="Launch GUI.")
    args = p.parse_args()

    if args.gui or (not any([args.in_path, args.list, args.ds_path, args.roi, args.grid, args.out, args.outdir]) and HAVE_GUI):
        if not HAVE_GUI:
            raise SystemExit("GUI dependencies not available. Install PySide6 and matplotlib.")
        app = QApplication([])
        w = SplitterApp()
        w.show()
        raise SystemExit(app.exec())

    if not args.in_path:
        p.error("--in path is required for CLI mode (or use --gui).")

    sp = HDFSplitter(args.in_path)
    try:
        if args.list:
            for d in sp.list_3d_datasets():
                print(d)
            return

        if not args.ds_path:
            p.error("--ds is required unless using --list or --gui.")
        sp.use_dataset(args.ds_path)

        H, W, C = sp.shape
        c_lo, c_hi = args.crange

        try:
            preview = sp.preview_sum(c_lo, c_hi, 256, stride=4)
        except Exception:
            preview = None

        # default clim like GUI (Robust)
        def _robust(arr):
            a = arr; n = a.size
            if n > 200_000:
                step = max(1, n // 200_000)
                a = a.ravel()[::step]
            vmin = float(np.nanmin(a)); vmax = float(np.nanmax(a))
            try:
                qlo = float(np.percentile(a, 1.0)); qhi = float(np.percentile(a, 99.0))
                if qhi > qlo: vmin, vmax = qlo, qhi
            except Exception: pass
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin: vmin, vmax = 0.0, 1.0
            return (vmin, vmax)

        clim = _robust(preview) if preview is not None else None

        if args.roi:
            if not args.out:
                p.error("--out is required for ROI export.")
            x0, x1, y0, y1 = args.roi
            sp.export_roi(args.out, x0, x1, y0, y1, ch_lo=c_lo, ch_hi=c_hi)
            print(f"ROI written to {args.out}")
            if preview is not None:
                png = Path(args.out).with_suffix("").as_posix() + "_roi_preview.png"
                _save_overview_png(preview, png, title="ROI export preview", roi=(x0, x1, y0, y1), clim=clim, full_shape=(H, W))
                print(f"Preview PNG -> {png}")
            return

        if args.grid:
            if not args.outdir:
                p.error("--outdir is required for grid export.")
            nx, ny = args.grid
            sp.export_grid(args.outdir, nx, ny, ch_lo=c_lo, ch_hi=c_hi)
            print(f"Tiled output in {args.outdir}")
            if preview is not None:
                base = Path(sp.path).stem
                png = Path(args.outdir) / f"{base}_grid_overview.png"
                _save_overview_png(preview, png, title="Grid export overview", grid=(nx, ny), clim=clim)
                print(f"Overview PNG -> {png}")
            return

        if args.auto_grid_mb:
            if not args.outdir:
                p.error("--outdir is required for auto-grid export.")
            nx, ny = sp.suggest_grid_from_target_mb(args.auto_grid_mb, c_lo, c_hi)
            print(f"Auto grid: nx={nx}, ny={ny}")
            sp.export_grid(args.outdir, nx, ny, ch_lo=c_lo, ch_hi=c_hi)
            print(f"Tiled output in {args.outdir}")
            if preview is not None:
                base = Path(sp.path).stem
                png = Path(args.outdir) / f"{base}_grid_overview.png"
                _save_overview_png(preview, png, title="Grid export overview", grid=(nx, ny), clim=clim)
                print(f"Overview PNG -> {png}")
            return

        p.error("Nothing to do. Use --list, --roi, --grid, --auto-grid-mb or --gui.")

    finally:
        sp.close()


if __name__ == "__main__":
    _cli()
