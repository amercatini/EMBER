#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
EMBER: Efficient Maximum-gap Burned-area Extraction and Recognition
===============================================================================

DESCRIPTION & METHODOLOGY SUMMARY:
----------------------------------
EMBER is an unsupervised, zero-tuning framework designed for rapid single-date 
post-fire burned area mapping using Sentinel-2 Multi-Spectral Instrument (MSI) 
imagery. Tailored for emergency response workflows when pre-fire baseline imagery 
is unavailable, EMBER resolves single-date spectral ambiguity by combining 
spatial-spectral feature representations with adaptive spectral gap decision rules.

Core Processing Pipeline:
1. Data Preprocessing & Water Masking:
   - Ingests 10 Bottom-Of-Atmosphere (BOA) spectral bands resampled to 10 m grid.
   - Computes an adaptive NDWI water mask, expanded by a 2-pixel morphological 
     dilation buffer (20 m safety margin) to eliminate coastal and shoreline 
     mixed-pixel false alarms.
2. Spatial-Spectral Feature Expansion (10D -> 20D):
   - Concatenates native 10 m surface reflectances with local 3x3 uniform filter 
     means (30 m x 30 m spatial footprint) to evaluate pixel neighborhoods and 
     suppress single-pixel classification noise.
3. Dimensionality Reduction via PCA (20D -> 8D):
   - Standardizes land reflectances and projects features onto an 8-dimensional 
     principal component subspace retaining >98% cumulative variance, filtering 
     high-frequency noise and atmospheric quantization artifacts.
4. Subsampled Gaussian Mixture Model (GMM):
   - Fits a K=7 full-covariance GMM on a representative random sample of valid 
     land pixels (max 200,000 pixels) using Expectation-Maximization.
5. Adaptive NIR Maximum-Gap Decision Rule:
   - Reconstructs 10D spectral centroids and computes mean Near-Infrared / Red-Edge 
     reflectances (B06, B07, B08, B8A).
   - Sorts centroids in ascending order and evaluates adjacent spectral steps, 
     selecting the maximum gap within the first 3 candidate transitions to 
     dynamically isolate burned clusters regardless of scene fire extent.
6. Spatial Vectorization & Geospatial Export:
   - Projects predictions into EPSG:3035 and extracts boundary contours using 
     4-connectivity tracing to suppress diagonal noise bridging.
   - Exports binary GeoTIFF masks and unified GeoPackage (.gpkg) vector layers.

===============================================================================
METADATA & AUTHORSHIP:
----------------------
Project Name : EMBER Framework
Author(s)    : Alessandro Mercatini, Nazario Tartaglione
Affiliation  : Italian Institute for Environmental Protection and Research (ISPRA)
Address      : Via Vitaliano Brancati 48, 00144 Roma, Italy
Contact      : alessandro.mercatini@isprambiente.it nazario.tartaglione@isprambiente.it

Date         : October 2026
Version      : 1.0.0
License      : MIT License (https://opensource.org/licenses/MIT)
Repository   : https://github.com/tuo-username/EMBER
Citation     : Mercatini, A.; Tartaglione, N. Can Single-Date Post-Fire 
               Algorithms Replace Bi-Temporal ANBR? A Multi-Algorithm Assessment 
               Across Mediterranean Fires. J. Imaging 2026, 12, 1-27.
===============================================================================
"""
import os
import time

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import binary_dilation, uniform_filter
from shapely.geometry import shape
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")


# ==============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# ==============================================================================

def load_phase_space_10d_with_water_mask(
    folder_path,
    bands_needed=[
        "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B11", "B12"
    ],
    ref_band="B08",
    buffer_iterations=2,
    water_threshold=0.2,
):
    """Loads the 10 Sentinel-2 spectral bands and computes the buffered NDWI water mask using a fixed threshold."""
    files = {b: None for b in bands_needed}
    valid_extensions = (".tif", ".tiff", ".jp2")

    for f in os.listdir(folder_path):
        f_lower = f.lower()
        if f_lower.endswith((".xml", ".json")) or f.startswith("."):
            continue
        if not f_lower.endswith(valid_extensions):
            continue
        for b in bands_needed:
            if (
                f"_{b}_" in f
                or f"_{b}." in f
                or f"_{b}(" in f
                or f.endswith(f"{b}.tif")
                or f.endswith(f"{b}.tiff")
                or f.endswith(f"{b}.jp2")
            ):
                files[b] = os.path.join(folder_path, f)

    ref_path = files[ref_band]
    if ref_path is None:
        raise FileNotFoundError(
            f"Reference band '{ref_band}' not found in: {folder_path}"
        )

    with rasterio.open(ref_path) as ref_ds:
        meta = ref_ds.meta.copy()
        transform = ref_ds.transform
        crs = ref_ds.crs
        ref_shape = (ref_ds.height, ref_ds.width)

    stacked_bands = []
    combined_nodata = np.zeros(ref_shape, dtype=bool)

    for b in bands_needed:
        if files[b] is None:
            raise FileNotFoundError(
                f"Band '{b}' not found in directory: {folder_path}"
            )
        with rasterio.open(files[b]) as ds:
            raw = ds.read(1)
            native_nodata = (
                (raw == 65535)
                | (raw == 0)
                | (raw <= 0)
                | (
                    raw == ds.nodata
                    if ds.nodata is not None
                    else np.zeros_like(raw, dtype=bool)
                )
            )

            if ds.shape != ref_shape:
                data = ds.read(1, out_shape=ref_shape, resampling=Resampling.bilinear)
                resampled_nodata = (
                    rasterio.warp.reproject(
                        source=native_nodata.astype(np.uint8),
                        destination=np.zeros(ref_shape, dtype=np.uint8),
                        src_transform=ds.transform,
                        src_crs=ds.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.nearest,
                    )[0]
                    > 0
                )
            else:
                data = raw
                resampled_nodata = native_nodata

            data_float = data.astype(np.float32) / 10000.0
            data_float[resampled_nodata] = np.nan
            data_float[data_float <= 0.0] = np.nan

            combined_nodata |= resampled_nodata | np.isnan(data_float)
            stacked_bands.append(data_float)

    image_stack = np.stack(stacked_bands, axis=0).astype(np.float32)

    # Calculate NDWI for water body masking
    b03_idx = bands_needed.index("B03")
    b08_idx = bands_needed.index("B08")
    b03 = image_stack[b03_idx]
    b08 = image_stack[b08_idx]

    ndwi = (b03 - b08) / (b03 + b08 + 1e-6)

    # Water mask with fixed conservative threshold (without Otsu)
    water_mask_raw = np.nan_to_num(ndwi, nan=-1.0) > water_threshold

    if buffer_iterations > 0:
        water_mask = binary_dilation(water_mask_raw, iterations=buffer_iterations)
    else:
        water_mask = water_mask_raw

    for b in range(image_stack.shape[0]):
        image_stack[b][combined_nodata] = np.nan

    water_pixel_count = np.sum(water_mask)
    total_pixels = water_mask.size
    print(
        f"💧 Water Mask (NDWI > {water_threshold:.2f} + Buffer"
        f" {buffer_iterations}px): Removed {water_pixel_count:,} pixels"
        f" ({water_pixel_count / total_pixels * 100:.2f}% of scene)"
    )

    return image_stack, water_mask, meta, transform, crs


# ==============================================================================
# CORE EMBER ALGORITHM (PCA, GMM, NIR MAXIMUM-GAP & OUTPUTS)
# ==============================================================================

def fit_pca_gmm_fast(
    X_20d_land,
    n_clusters=7,
    pca_components=8,
    sample_fraction=0.10,
    max_samples=200000,
    min_samples=10000,
):
    """PCA dimensionality reduction (20D -> 8D) and GMM fitting on a random subsample."""
    log_lines = []
    msg_head = f"PCA Reduction (20D -> {pca_components}D) and GMM Fitting (K = {n_clusters})..."
    print(f"\n{msg_head}")
    log_lines.append(msg_head)

    X_20d_land_64 = np.asarray(X_20d_land, dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_20d_land_64)

    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled).astype(np.float64)

    exp_var = pca.explained_variance_ratio_
    total_var = np.sum(exp_var) * 100

    var_msg = f"Total variance retained by {pca_components}D: {total_var:.2f}%"
    print(var_msg)
    log_lines.append(var_msg)

    for i, ratio in enumerate(exp_var):
        comp_msg = f"   ├─ PC{i + 1}: {ratio * 100:.2f}% of variance"
        print(comp_msg)
        log_lines.append(comp_msg)

    total_land_pixels = X_pca.shape[0]
    calculated_sample = int(total_land_pixels * sample_fraction)
    sample_size = max(
        min(calculated_sample, max_samples),
        min(min_samples, total_land_pixels),
    )

    msg_pix = f"📏 Total land pixels: {total_land_pixels:,}"
    msg_samp = (
        f"Subsample drawn for GMM fit: {sample_size:,} pixels"
        f" ({sample_size / total_land_pixels * 100:.1f}% of land)"
    )

    print(msg_pix)
    print(msg_samp)
    log_lines.append(msg_pix)
    log_lines.append(msg_samp)

    np.random.seed(42)
    idx_sample = np.random.choice(total_land_pixels, size=sample_size, replace=False)
    X_sample = X_pca[idx_sample]

    # GMM fitting with adaptive reg_covar fallback
    gmm = None
    reg_covar_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    for reg_val in reg_covar_candidates:
        try:
            candidate_gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                reg_covar=reg_val,
                max_iter=150,
                n_init=10,
                random_state=42,
            )
            candidate_gmm.fit(X_sample)
            gmm = candidate_gmm
            if reg_val != 1e-6:
                warn_msg = f"GMM fit stabilized with reg_covar={reg_val} (anti-crash fallback)"
                print(warn_msg)
                log_lines.append(warn_msg)
            break
        except (ValueError, np.linalg.LinAlgError):
            continue

    if gmm is None:
        raise RuntimeError(f"Unable to fit GMM on this dataset with K={n_clusters}.")

    land_labels = gmm.predict(X_pca)

    means_10d = np.zeros((n_clusters, 10), dtype=np.float32)
    for k in range(n_clusters):
        mask = land_labels == k
        if np.any(mask):
            means_10d[k] = np.mean(X_20d_land[mask, :10], axis=0)

    return land_labels, means_10d, n_clusters, log_lines


def extract_burn_by_nir_max_gap(means_10d, bands_needed):
    """Identifies burned clusters by evaluating the maximum gap along NIR/Red-Edge channels."""
    log_lines = []

    nir_indices = [bands_needed.index(b) for b in ["B06", "B07", "B08", "B8A"]]
    nir_means = np.mean(means_10d[:, nir_indices], axis=1)

    sorted_indices = np.argsort(nir_means)
    sorted_values = nir_means[sorted_indices]

    gaps = np.diff(sorted_values)

    max_search_depth = min(3, len(gaps))
    evaluated_gaps = gaps[:max_search_depth]

    max_gap_rel_idx = int(np.argmax(evaluated_gaps)) if evaluated_gaps.size > 0 else 0

    burned_ids = sorted_indices[: max_gap_rel_idx + 1].tolist()
    reason = (
        f"Maximum relative gap identified at gap #{max_gap_rel_idx + 1}"
        f" (+{evaluated_gaps[max_gap_rel_idx]:.4f}) -> Selected {len(burned_ids)} cluster(s)"
    )

    header_1 = "[Burned Cluster Identification - Data-Driven NIR Maximum-Gap]..."
    header_2 = "=" * 65

    print(f"{header_1}")
    print(header_2)

    log_lines.extend([header_1, header_2])

    for i in range(len(sorted_indices)):
        cid = sorted_indices[i]
        val = sorted_values[i]

        gap_str = f"  -->  [Gap: +{gaps[i]:.4f}]" if i < len(gaps) else ""
        if i < max_search_depth:
            gap_str += f" (Evaluated - Gap #{i+1})"
        if cid in burned_ids:
            gap_str += " [BURNED]"

        line = f"Cluster #{cid:<2} | NIR Mean (B06-B8A): {val:.4f}{gap_str}"
        print(line)
        log_lines.append(line)

    footer_1 = "=" * 65
    footer_2 = f"Decision rule: {reason}"
    footer_3 = f"Selected clusters: {burned_ids}\n"

    print(footer_1)
    print(footer_2)
    print(footer_3)

    log_lines.extend([footer_1, footer_2, footer_3])

    return burned_ids, log_lines


def plot_spectral_signatures(means_10d, band_names, burned_ids, output_plot_path):
    """Generates and saves the spectral signature plot for cluster centroids."""
    K = means_10d.shape[0]
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(band_names))
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    for k in range(K):
        is_burn = k in burned_ids
        linestyle = "-" if is_burn else "--"
        alpha = 1.0 if is_burn else 0.4
        linewidth = 2.5 if is_burn else 1.2
        label = f"Cluster #{k} [BURNED]" if is_burn else f"Cluster #{k}"

        plt.plot(
            x_indices,
            means_10d[k],
            marker="o",
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            color=colors[k],
            label=label,
        )

    plt.xticks(x_indices, band_names, rotation=45, fontsize=10)
    plt.ylabel("Reflectance (BOA)", fontsize=12)
    plt.xlabel("Sentinel-2 Spectral Bands", fontsize=12)
    plt.title(f"Centroid Spectral Signatures (PCA + Fast GMM - K={K})", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Cluster ID", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()

    output_dir = os.path.dirname(output_plot_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def export_geospatial_outputs(
    burn_mask_native,
    meta,
    transform,
    crs,
    target_crs,
    output_dir,
    file_prefix="ember",
    area_log_prefix="EMBER",
):
    """Reprojects to EPSG:3035 and saves GeoTIFF mask and GPKG vector perimeter."""
    H, W = burn_mask_native.shape
    bounds = rasterio.transform.array_bounds(H, W, transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs, target_crs, W, H, *bounds
    )

    burn_mask_3035 = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=burn_mask_native,
        destination=burn_mask_3035,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )

    tif_path = os.path.join(output_dir, f"{file_prefix}_mask.tif")
    tif_meta = meta.copy()
    tif_meta.update(
        crs=target_crs,
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        dtype=rasterio.uint8,
        count=1,
        compress="lzw",
        nodata=255,
    )
    with rasterio.open(tif_path, "w", **tif_meta) as dst:
        dst.write(burn_mask_3035, 1)

    # Contour extraction using 4-connectivity
    shapes_gen = rasterio.features.shapes(
        burn_mask_3035,
        mask=(burn_mask_3035 == 1),
        transform=dst_transform,
        connectivity=4,
    )
    burn_polygons = [shape(geom) for geom, val in shapes_gen if val == 1]

    if burn_polygons:
        gdf_burn_3035 = gpd.GeoDataFrame({"geometry": burn_polygons}, crs=target_crs)
        unified_geometry = (
            gdf_burn_3035.union_all()
            if hasattr(gdf_burn_3035, "union_all")
            else gdf_burn_3035.unary_union
        )
        gdf_perimeter = gpd.GeoDataFrame({"geometry": [unified_geometry]}, crs=target_crs)
        gdf_perimeter["area_ha"] = gdf_perimeter.geometry.area / 10000.0
        area_ha = float(gdf_perimeter["area_ha"].iloc[0])
    else:
        gdf_perimeter = gpd.GeoDataFrame(columns=["geometry"], crs=target_crs)
        area_ha = 0.0

    gpkg_path = os.path.join(output_dir, f"{file_prefix}_polygon.gpkg")
    gdf_perimeter.to_file(gpkg_path, driver="GPKG")

    print(f"GeoTIFF mask saved: {tif_path}")
    print(f"GPKG vector saved:  {gpkg_path}")
    print(f"Estimated burned area ({target_crs}): {area_ha:.2f} ha")

    return area_ha


# ==============================================================================
# STANDALONE EXECUTION PIPELINE FOR EMBER
# ==============================================================================

def run_ember_standalone(post_folder, output_dir, n_clusters=7, target_crs="EPSG:3035"):
    """Executes the complete EMBER pipeline on a single post-fire scene."""
    bands_needed = [
        "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B11", "B12"
    ]
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    print("=" * 70)
    print("EXECUTING EMBER FRAMEWORK (SINGLE-DATE)")
    print("=" * 70)

    # Load 10D stack and water mask
    post_stack, water_mask, meta, transform, crs = load_phase_space_10d_with_water_mask(
        post_folder, bands_needed=bands_needed, buffer_iterations=2
    )
    C, H, W = post_stack.shape

    # Feature space expansion to 20D (10 spectral + 10 spatial context 3x3)
    post_stack_clean = np.nan_to_num(post_stack, nan=0.0)
    spatial_means = np.zeros_like(post_stack)
    for b in range(C):
        spatial_means[b] = uniform_filter(post_stack_clean[b], size=3)

    ext_stack = np.concatenate([post_stack, spatial_means], axis=0)
    X_20d_full = ext_stack.reshape(2 * C, -1).T

    # Filter valid land pixels
    is_nodata = np.isnan(post_stack).any(axis=0)
    valid_pixels = (~water_mask) & (~is_nodata)
    land_mask_flat = valid_pixels.reshape(-1)
    X_20d_land = X_20d_full[land_mask_flat]

    if X_20d_land.shape[0] == 0:
        raise ValueError("No valid land pixels found in the scene.")

    # PCA + GMM Fitting
    land_labels, means_10d, k_active, pca_gmm_log = fit_pca_gmm_fast(
        X_20d_land, n_clusters=n_clusters, pca_components=8
    )

    cluster_map = np.full((H * W), fill_value=-1, dtype=np.int32)
    cluster_map[land_mask_flat] = land_labels
    cluster_map = cluster_map.reshape(H, W)

    # Burned cluster selection via NIR Maximum-Gap
    burned_ids, burn_gap_log = extract_burn_by_nir_max_gap(means_10d, bands_needed)
    ember_burn_mask_native = np.isin(cluster_map, burned_ids).astype(np.uint8)

    # Export geospatial outputs and plot
    area_ha = export_geospatial_outputs(
        ember_burn_mask_native,
        meta,
        transform,
        crs,
        target_crs,
        output_dir,
        file_prefix="ember",
        area_log_prefix="EMBER",
    )

    plot_path = os.path.join(output_dir, "ember_spectral_signatures.png")
    plot_spectral_signatures(means_10d, bands_needed, burned_ids, plot_path)

    elapsed = time.time() - start_time

    # Save execution text report
    ember_report = (
        pca_gmm_log
        + [""]
        + burn_gap_log
        + [
            "",
            f"Estimated burned area: {area_ha:.2f} ha",
            f"Total execution time: {elapsed:.2f} seconds",
        ]
    )
    report_path = os.path.join(output_dir, "ember_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ember_report) + "\n")

    print(f"Execution report saved: {report_path}")
    print(f"⏱ Total execution time: {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    POST_FIRE_FOLDER = "dati/lucca_2026/post"
    OUTPUT_FOLDER = "out_test"

    run_ember_standalone(
        post_folder=POST_FIRE_FOLDER,
        output_dir=OUTPUT_FOLDER,
        n_clusters=7,
    )