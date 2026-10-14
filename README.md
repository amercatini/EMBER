# EMBER: Efficient Maximum-gap Burned-area Extraction and Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Journal](https://img.shields.io/badge/Journal-J._Imaging_2026-green.svg)](https://www.mdpi.com/journal/jimaging)

**EMBER** is an unsupervised, zero-tuning Python framework designed for rapid single-date post-fire burned area mapping using Sentinel-2 Multi-Spectral Instrument (MSI) imagery.

Tailored for emergency response scenarios where pre-fire baseline imagery is unavailable or obscured by clouds, EMBER resolves single-date spectral ambiguity by combining spatial-spectral feature representations with data-driven NIR gap decision rules.

---

## ⚙️ Methodology & Workflow

The framework processes a single post-fire Sentinel-2 image through a six-stage automated pipeline:

![EMBER Framework Workflow](docs/workflow.pdf)

1. **Data Preprocessing & Water Masking**: Ingests 10 Bottom-Of-Atmosphere (BOA) spectral bands resampled to a 10 m grid. Applies a fixed NDWI water mask expanded by a 2-pixel morphological dilation buffer ($20\text{ m}$ safety margin) to eliminate coastal and inland water false alarms.
2. **Spatial-Spectral Feature Expansion ($10\text{D} \rightarrow 20\text{D}$)**: Concatenates native 10 m surface reflectances with local $3 \times 3$ uniform filter spatial means ($30\text{ m} \times 30\text{ m}$ footprint) to evaluate contextual neighborhoods and suppress speckle noise.
3. **Dimensionality Reduction via PCA ($20\text{D} \rightarrow 8\text{D}$)**: Standardizes land reflectances and projects features onto an 8-dimensional principal component subspace retaining $>98\%$ cumulative variance.
4. **Subsampled Gaussian Mixture Model ($K=7$)**: Fits a full-covariance GMM on a random subsample of valid land pixels ($\le 200,000$ pixels) using Expectation-Maximization.
5. **Adaptive NIR Maximum-Gap Decision Rule**: Reconstructs 10D spectral centroids and evaluates mean Near-Infrared / Red-Edge reflectances (`B06`, `B07`, `B08`, `B8A`). Dynamically selects burned clusters at the largest spectral gap across candidate transitions.
6. **Spatial Vectorization & Geospatial Export**: Projects predictions into EPSG:3035 and extracts boundary contours using 4-connectivity tracing to avoid diagonal noise bridging.

---

## 📁 Repository Structure

```text
EMBER/
├── ember.py               # Main Python implementation of the framework
├── requirements.txt       # Dependencies list
├── README.md              # Framework documentation and quickstart
├── LICENSE                # MIT License
├── docs/
│   └── workflow.png       # Framework methodology diagram
└── example_data/
    └── teruel_crop/       # Small test ROI for out-of-the-box verification
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/tuo-username/EMBER.git
cd EMBER
```

### 2. Install Dependencies
It is recommended to use a virtual environment (Conda or `venv`):

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Input Data Format
EMBER expects a directory containing the 10 Sentinel-2 BOA spectral bands (`B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B11`, `B12`) in `.tif`, `.tiff`, or `.jp2` format (e.g., Copernicus L2A outputs).

### Execution

You can run EMBER directly from the command line:

```python
from ember import run_ember_standalone

run_ember_standalone(
    post_folder="example_data/teruel_crop",
    output_dir="out_test",
    n_clusters=7,
    target_crs="EPSG:3035"
)
```

Or execute `ember.py` directly:

```bash
python ember.py
```

---

## 📊 Outputs Generated

Execution produces the following artifacts in the specified output folder:

* **`ember_mask.tif`**: Single-band binary GeoTIFF raster mask (1 = Burned, 0 = Unburned, 255 = NoData/Water) projected to EPSG:3035.
* **`ember_polygon.gpkg`**: Vector polygon layer containing the unified perimeter and calculated area in hectares (`area_ha`).
* **`ember_spectral_signatures.png`**: Plot showing BOA reflectance curves for all $K=7$ cluster centroids and highlighting selected burned clusters.
* **`ember_report.txt`**: Execution log containing variance ratios, GMM convergence status, NIR gap decision metrics, and total runtime.

---

## 📖 Citation

If you use EMBER in your research or operational workflows, please cite the following paper:

```bibtex
@Article{mercatini2026ember,
  author         = {Mercatini, Alessandro and Tartaglione, Nazario},
  title          = {Can Single-Date Post-Fire Algorithms Replace Bi-Temporal ANBR? A Multi-Algorithm Assessment Across Mediterranean Fires},
  journal        = {Journal of Imaging},
  volume         = {12},
  number         = {1},
  pages          = {1--27},
  year           = {2026},
  doi            = {10.3390/jimaging12010000}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact & Institutional Affiliation

**Alessandro Mercatini**  
Italian Institute for Environmental Protection and Research (ISPRA)  
Via Vitaliano Brancati 48, 00144 Roma, Italy  
📧 Email: alessandro.mercatini@isprambiente.it
