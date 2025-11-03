# XRF Imaging GUI

## Abstract
A desktop app for **interactive clustering and exploration of X‑ray fluorescence (XRF)** maps and spectra. Load HDF5 scans with a CFG fit (via PyMca’s FastXRFLinearFit) or import a prebuilt `images.csv`, then filter, fuse features (optionally add XY), project (PCA/NMF/t‑SNE/UMAP), cluster (K‑means), and interpret results with **per‑cluster spectra**, **ROI tools**, and **cluster×element heatmaps**. The UI favors clarity (outlines, legend, stable colors), reproducibility (explicit parameters, no hidden resorting), and predictable memory (streamed spectra, cached Moran’s I).

## Check Out the Instructions Video
https://youtu.be/OEp86lLKPew

## Usage
1. **Launch**
   ```bash
   python xrf_GUI_test.py
   ```

2. **Load data**
   - **Load from HDF…** → pick HDF/H5 and CFG, select the dataset path `(rows × cols × channels)`.  
     Builds an elemental table in memory using **FastXRFLinearFit**.
   - **Load + Save raw CSV…** → same as above, plus prompt to save `images.csv`.
   - **Import images.csv…** → reuse a previously generated table (semicolon `;` delimited).

3. **(Optional) Filter**
   - **Moran’s I** to keep spatially coherent elements (cached).
   - **Foreground clustering** + **Element map** to include/exclude elements.

4. **(Optional) Feature fusion**
   - **Add XY** (then scale), **Scale** (Standard/MinMax), **Augment** (separate scalers & weights for elements vs XY).

5. **Projections**
   - PCA, NMF, t‑SNE (perplexity), UMAP (metric picker with options for Mahalanobis/wMinkowski/seuclidean).

6. **Clustering**
   - Choose *k* and **base** (dataset or any projection).  
   - Evaluate **Silhouette** and **Davies–Bouldin** sweeps to pick *k*.

7. **Analyze**
   - **Spectra**: per‑cluster mean/sum (log optional) streamed from HDF; CSV export.  
   - **ROI Imaging**: two heatmaps + draggable ROI linked to spectra.  
   - **Concentrations**: cluster×element heatmap (mean/median/sum; visible‑only/all; PNG/CSV).

8. **Export**
   - **Save canvases → PNG** (left image + right projections, side‑by‑side).  
   - CSV/PNG exports respect current aggregation and visibility.

## Prerequisites
- **Python** 3.10+ (3.11 OK)
- **pip** up to date
- **PyMca5** (installed via pip; provides FastXRFLinearFit)
- **Windows note:** Microsoft **Visual Studio C++ Build Tools** may be required for PyMca.

### Install (venv recommended)
**Windows (PowerShell)**
```powershell
install_quick.ps1
```
OR
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
pip install PySide6 numpy pandas matplotlib scikit-learn umap-learn h5py libpysal esda PyMca5
```

**macOS / Linux**
```bash
install_quick.sh
```
OR
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install PySide6 numpy pandas matplotlib scikit-learn umap-learn h5py libpysal esda PyMca5
```

## Communication
- **Issues & discussions:** use the GitHub **Issues** tab for bugs/requests, and **Discussions** (if enabled) for Q&A/ideas.
- For private datasets, please anonymize filenames/paths before sharing logs or screenshots.

## Contributing
Contributions are welcome!  
- Keep UI changes consistent with the existing layout and “explicit > implicit” approach.  
- Prefer small, focused PRs.  
- Add a short note in the log for new pipeline steps or parameters.  
- Include a brief test plan (what you ran, expected vs. observed).

## License

```
This project is licensed under the Apache License. See the LICENSE for details.
```

## Citation
If this tool supports your research, please cite the repository (XRF clustering & visualization GUI).

