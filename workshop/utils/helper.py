"""
# © Copyright IBM Corporation 2026
# SPDX-License-Identifier: Apache-2.0

Helper functions for visualization and model management in the TerraMind workshop notebooks.

This module provides utilities for:
- Loading and visualizing geospatial raster data (RGB imagery, DEM data, labels, predictions)
- Downloading models and configs from HuggingFace
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from huggingface_hub import hf_hub_download
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm


def download_model_from_hf(repo_id, config_name, checkpoint_name, project_root):
    """
    Download model config and checkpoint from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        config_name: Name of the config file in the repo (e.g., "config.yaml")
        checkpoint_name: Name of the checkpoint file in the repo (e.g., "model.ckpt")
        project_root: Path to project root directory

    Returns:
        tuple: (config_path, checkpoint_path) - Paths to downloaded files
    """

    # Download config
    config_folder = project_root / "configs" / "inference"
    config_folder.mkdir(parents=True, exist_ok=True)

    print(f"Downloading config: {config_name}")
    config_file = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=config_name,
            local_dir=config_folder,
            resume_download=True,  # Resume if interrupted
        )
    )
    print(f"✓ Config downloaded")

    # Download checkpoint (this may take a few minutes for large files)
    checkpoint_folder = project_root / "checkpoints"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading checkpoint: {checkpoint_name}")
    print("⏳ This may take a few minutes (~350MB file)...")
    checkpoint_file = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_name,
            local_dir=checkpoint_folder,
            resume_download=True,  # Resume if interrupted
        )
    )
    print(f"✓ Checkpoint downloaded")

    return config_file, checkpoint_file


def load_raster(file_path, rgb_only=False):
    """
    Load a raster file and return the data array.
    Masks nodata values (< -999) as NaN.

    Args:
        file_path: Path to the raster file
        rgb_only: If True, extract only RGB bands (1-3) from multi-band images

    Returns:
        numpy array or None if file doesn't exist
    """
    if file_path is None or not Path(file_path).exists():
        return None

    try:
        with rasterio.open(file_path) as src:
            data = src.read()
            nodata = src.nodata

            # If requesting RGB only and we have at least 3 bands
            if rgb_only and data.shape[0] >= 3:
                # Extract bands 1-3 (indices 0-2) for RGB
                rgb_data = data[0:3, :, :].astype(np.float32)

                # Mask nodata values (anything below -999)
                rgb_data = np.where(rgb_data < -999, np.nan, rgb_data)
                if nodata is not None:
                    rgb_data = np.where(rgb_data == nodata, np.nan, rgb_data)

                # Transpose to (H, W, 3) for matplotlib
                return np.transpose(rgb_data, (1, 2, 0))

            # If single band, return as 2D (e.g., labels, predictions)
            elif data.shape[0] == 1:
                single_band = data[0].astype(np.float32)
                # Mask nodata values
                single_band = np.where(single_band < -999, np.nan, single_band)
                if nodata is not None:
                    single_band = np.where(single_band == nodata, np.nan, single_band)
                return single_band

            # Default: return first band only for single-band data
            return data[0] if data.shape[0] == 1 else data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def normalize_rgb(rgb_data):
    """
    Normalize RGB data to 0-1 range for display.
    Handles NaN values (from nodata masking) properly.
    """
    if rgb_data is None:
        return None

    # Clip to 2nd and 98th percentiles for better contrast (ignoring NaN)
    p2, p98 = np.nanpercentile(rgb_data, (2, 98))

    # Normalize, keeping NaN values as NaN
    rgb_norm = np.clip((rgb_data - p2) / (p98 - p2), 0, 1)

    # Convert NaN to 0 for display (will appear as black)
    rgb_norm = np.nan_to_num(rgb_norm, nan=0.0)

    return rgb_norm


def plot_panel(
    ax,
    data,
    title,
    cmap=None,
    vmin=None,
    vmax=None,
    is_categorical=False,
    n_classes=2,
    is_dem=False,
):
    """
    Plot a single panel in the grid.

    Args:
        ax: Matplotlib axis
        data: Data to plot
        title: Panel title
        cmap: Colormap (optional)
        vmin, vmax: Value range (optional)
        is_categorical: Whether data is categorical
        n_classes: Number of classes for categorical data
        is_dem: Whether data is DEM (uses log scale and grayscale)
    """
    if data is None:
        ax.text(
            0.5,
            0.5,
            "Data not available",
            ha="center",
            va="center",
            fontsize=12,
            color="red",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        return

    if len(data.shape) == 3:  # RGB image
        ax.imshow(data)
    else:  # Single band
        if is_categorical:
            im = ax.imshow(
                data, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest"
            )
        elif is_dem:
            # DEM visualization with log scale and grayscale
            # Mask negative and zero values for log scale
            data_masked = np.ma.masked_where(data <= 0, data)
            # Use hardcoded vmin/vmax for salt marsh elevations (typically 0.1m to 10m)
            im = ax.imshow(data_masked, cmap="gray", norm=LogNorm(vmin=0.1, vmax=10))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")
        else:
            im = ax.imshow(data, cmap=cmap if cmap else "terrain", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


# Made with Bob


def get_multiclass_colormap():
    """
    Get colormap and labels for multi-class saltmarsh zone classification.

    Returns:
        tuple: (cmap, class_labels, class_colors) for 4-class zonal classification
    """
    from matplotlib.colors import ListedColormap

    # Multi-class: 0=Not Saltmarsh, 1=Pioneer, 2=Mid-Low, 3=Upper
    zone_styles = {
        0: {"label": "Not Saltmarsh", "color": "#d9d9d9"},
        1: {"label": "Pioneer", "color": "#fdae61"},
        2: {"label": "Mid-Low", "color": "#abd9e9"},
        3: {"label": "Upper", "color": "#2c7bb6"},
    }

    class_colors = [zone_styles[i]["color"] for i in range(4)]
    class_cmap = ListedColormap(class_colors)
    class_labels = [zone_styles[i]["label"] for i in range(4)]

    return class_cmap, class_labels, class_colors


def locate_checkpoints(search_dir: Path) -> list:
    """Locate checkpoints that are contained in a given directory

    Returns:
        list: checkpoints that were found
    """
    # search for checkpoints
    new_checkpoint_dir = list(search_dir.rglob("*.ckpt"))
    if len(new_checkpoint_dir) > 0:
        print(f"Found {len(new_checkpoint_dir)} files:")
        for ckpt in new_checkpoint_dir:
            print(f"  - {ckpt}")
    else:
        print("⚠ No checkpoints found in the directory.")

    return new_checkpoint_dir


def download_and_extract_zip(
    file_id, zip_filename="dataset.zip", extract_to="../../data"
):
    """
    Download a zip file from Google Drive and extract it.

    Args:
        file_id: Google Drive file ID
        zip_filename: Name for the downloaded zip file (default: "dataset.zip")
        extract_to: Directory to extract the contents to (default: "../../data")

    Returns:
        bool: True if successful, False otherwise
    """
    import os
    import subprocess

    import gdown

    # Download the zip file from Google Drive
    if not os.path.isfile(zip_filename):
        print("Downloading dataset from Google Drive...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}", zip_filename, quiet=False
            )
            print(f"✓ Downloaded to {zip_filename}")
        except Exception as e:
            print(f"⚠ Download error: {e}")
            return False
    else:
        print(f"✓ Zip file already exists: {zip_filename}")

    # Extract the zip file
    if os.path.isfile(zip_filename):
        print(f"\nExtracting to {extract_to}...")
        try:
            result = subprocess.run(
                ["unzip", "-q", "-o", zip_filename, "-d", extract_to],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("✓ Extraction complete!")
                return True
            else:
                print(f"⚠ Extraction error: {result.stderr}")
                return False
        except Exception as e:
            print(f"⚠ Extraction error: {e}")
            return False

    return False


def plot_s2_model_pred(s2_img_file, s2_lab_file, s2_pred_file):
    """
    Plot S2 model predictions with input imagery, ground truth labels, and predictions.

    Args:
        s2_img_file: Path to S2 input image file
        s2_lab_file: Path to S2 label file
        s2_pred_file: Path or list of paths to S2 prediction file(s)

    Returns:
        None (displays plot)
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    # Set up for extent model (binary classification)
    CLASS_COLORS = ["#d9d9d9", "#2c7bb6"]  # Not Saltmarsh, Saltmarsh
    CLASS_CMAP = ListedColormap(CLASS_COLORS)
    CLASS_LABELS = ["Not Saltmarsh", "Saltmarsh"]
    N_CLASSES = 2

    # Handle prediction file as list or single path
    pred_file = s2_pred_file[0] if isinstance(s2_pred_file, list) else s2_pred_file

    # Load S2 data
    img_data = normalize_rgb(load_raster(s2_img_file, rgb_only=True))
    lab_data = load_raster(s2_lab_file)
    pred_data = load_raster(pred_file)

    # Create figure with 1 row × 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("S2 Model Results (10m resolution)", fontsize=16, fontweight="bold")

    # Plot panels
    plot_panel(axes[0], img_data, "Input S2 Imagery")
    plot_panel(
        axes[1],
        lab_data,
        "Ground Truth Labels",
        cmap=CLASS_CMAP,
        is_categorical=True,
        n_classes=N_CLASSES,
    )
    plot_panel(
        axes[2],
        pred_data,
        "Model Predictions",
        cmap=CLASS_CMAP,
        is_categorical=True,
        n_classes=N_CLASSES,
    )

    # Add legend
    legend_elements = [
        Patch(facecolor=CLASS_COLORS[i], label=CLASS_LABELS[i])
        for i in range(len(CLASS_LABELS))
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Visualized S2 model results")
    print(f"  Image: {Path(s2_img_file).name}")
    print(f"  Label: {Path(s2_lab_file).name}")
    print(f"  Prediction: {Path(pred_file).name}")


def plot_s2_rgbdem_model_pred(
    s2_img_file,
    s2_lab_file,
    s2_pred_file,
    rgbdem_img_file,
    rgbdem_dem_file,
    rgbdem_lab_file,
    rgbdem_pred_file,
):
    """
    Plot comparison of S2 and RGB+DEM model predictions side by side.

    Args:
        s2_img_file: Path to S2 input image file
        s2_lab_file: Path to S2 label file
        s2_pred_file: Path or list of paths to S2 prediction file(s)
        rgbdem_img_file: Path to RGB+DEM input image file
        rgbdem_dem_file: Path to DEM file
        rgbdem_lab_file: Path to RGB+DEM label file
        rgbdem_pred_file: Path or list of paths to RGB+DEM prediction file(s)

    Returns:
        None (displays plot)
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    # Set up for extent model (binary classification)
    CLASS_COLORS = ["#d9d9d9", "#2c7bb6"]  # Not Saltmarsh, Saltmarsh
    CLASS_CMAP = ListedColormap(CLASS_COLORS)
    CLASS_LABELS = ["Not Saltmarsh", "Saltmarsh"]
    N_CLASSES = 2

    # Handle prediction files as lists or single paths
    s2_pred = s2_pred_file[0] if isinstance(s2_pred_file, list) else s2_pred_file
    rgbdem_pred = (
        rgbdem_pred_file[0] if isinstance(rgbdem_pred_file, list) else rgbdem_pred_file
    )

    # Load S2 data (10m resolution)
    s2_img_data = normalize_rgb(load_raster(s2_img_file, rgb_only=True))
    s2_lab_data = load_raster(s2_lab_file)
    s2_pred_data = load_raster(s2_pred)

    # Load RGB+DEM data (2m resolution)
    rgbdem_img_data = normalize_rgb(load_raster(rgbdem_img_file, rgb_only=True))
    rgbdem_dem_data = load_raster(rgbdem_dem_file)
    rgbdem_lab_data = load_raster(rgbdem_lab_file)
    rgbdem_pred_data = load_raster(rgbdem_pred)

    # Create figure with 2 rows × 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "Model Comparison: S2 (10m) vs RGB+DEM (2m)", fontsize=16, fontweight="bold"
    )

    # Row 1: S2 Model (10m resolution)
    plot_panel(axes[0, 0], s2_img_data, "S2 - Input Imagery (10m)")
    plot_panel(axes[0, 1], None, "S2 - DEM (N/A)")  # Blank for S2
    plot_panel(
        axes[0, 2],
        s2_lab_data,
        "S2 - Labels",
        cmap=CLASS_CMAP,
        is_categorical=True,
        n_classes=N_CLASSES,
    )
    plot_panel(
        axes[0, 3],
        s2_pred_data,
        "S2 - Predictions",
        cmap=CLASS_CMAP,
        is_categorical=True,
        n_classes=N_CLASSES,
    )

    # Row 2: RGB+DEM Model (2m resolution)
    plot_panel(axes[1, 0], rgbdem_img_data, "RGB+DEM - Input Imagery (2m)")
    plot_panel(axes[1, 1], rgbdem_dem_data, "RGB+DEM - DEM (10m)", is_dem=True)
    plot_panel(
        axes[1, 2],
        rgbdem_lab_data,
        "RGB+DEM - Labels",
        cmap=CLASS_CMAP,
        is_categorical=True,
        n_classes=N_CLASSES,
    )
    plot_panel(
        axes[1, 3],
        rgbdem_pred_data,
        "RGB+DEM - Predictions",
        cmap=CLASS_CMAP,
        is_categorical=True,
        n_classes=N_CLASSES,
    )

    # Add legend
    legend_elements = [
        Patch(facecolor=CLASS_COLORS[i], label=CLASS_LABELS[i])
        for i in range(len(CLASS_LABELS))
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Comparison visualization complete")
    print(f"\nS2 Model (10m):")
    print(f"  Image: {Path(s2_img_file).name}")
    print(f"  Label: {Path(s2_lab_file).name}")
    print(f"  Prediction: {Path(s2_pred).name}")
    print(f"\nRGB+DEM Model (2m):")
    print(f"  Image: {Path(rgbdem_img_file).name}")
    print(f"  DEM: {Path(rgbdem_dem_file).name}")
    print(f"  Label: {Path(rgbdem_lab_file).name}")
    print(f"  Prediction: {Path(rgbdem_pred).name}")


def clip_raster_to_size(
    input_path, output_path, width=400, height=400, x_offset=0, y_offset=0
):
    """
    Clip a raster file to a specified size from a given offset.

    Args:
        input_path: Path to input raster file
        output_path: Path to save clipped raster
        width: Width of output raster in pixels (default: 400)
        height: Height of output raster in pixels (default: 400)
        x_offset: X offset from top-left corner (default: 0)
        y_offset: Y offset from top-left corner (default: 0)

    Returns:
        Path: Path to the clipped output file
    """
    from pathlib import Path

    import rasterio
    from rasterio.windows import Window

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        # Define the window to read (x_offset, y_offset, width, height)
        window = Window(x_offset, y_offset, width, height)

        # Read the data for this window
        data = src.read(window=window)

        # Update metadata for the clipped raster
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": height,
                "width": width,
                "transform": rasterio.windows.transform(window, src.transform),
            }
        )

        # Write the clipped raster
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(data)

    print(f"✓ Clipped {input_path.name} to {width}x{height} pixels")
    print(f"  Saved to: {output_path}")

    return output_path


def clip_directory_rasters(
    input_dir, output_dir, file_pattern="*.tif", width=400, height=400
):
    """
    Clip all raster files matching a pattern in a directory.

    Args:
        input_dir: Directory containing input rasters
        output_dir: Directory to save clipped rasters
        file_pattern: Glob pattern for files to clip (default: "*.tif")
        width: Width of output rasters in pixels (default: 400)
        height: Height of output rasters in pixels (default: 400)

    Returns:
        list: Paths to all clipped files
    """
    from pathlib import Path

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all matching files
    input_files = list(input_dir.glob(file_pattern))

    if not input_files:
        print(f"⚠ No files found matching pattern '{file_pattern}' in {input_dir}")
        return []

    print(f"Found {len(input_files)} files to clip:")
    clipped_files = []

    for input_file in input_files:
        output_file = output_dir / input_file.name
        clipped_path = clip_raster_to_size(
            input_file, output_file, width=width, height=height
        )
        clipped_files.append(clipped_path)

    print(f"\n✓ Clipped {len(clipped_files)} files to {output_dir}")
    return clipped_files
