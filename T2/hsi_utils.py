"""Small helpers for EHU-style hyperspectral MATLAB datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


DEFAULT_CUBE_NAMES = (
    "Salinas_corrected",
    "salinas_corrected",
    "Salinas",
    "salinas",
    "indian_pines_corrected",
    "indian_pines",
    "Indian_pines_corrected",
    "Indian_pines",
    "pavia",
    "paviaU",
    "Pavia",
    "PaviaU",
    "KSC",
    "ksc",
    "Botswana",
    "botswana",
    "cuprite",
    "Cuprite",
)

DEFAULT_GT_NAMES = (
    "Salinas_gt",
    "salinas_gt",
    "indian_pines_gt",
    "Indian_pines_gt",
    "pavia_gt",
    "paviaU_gt",
    "Pavia_gt",
    "PaviaU_gt",
    "KSC_gt",
    "ksc_gt",
    "Botswana_gt",
    "botswana_gt",
)

SALINAS_REMOVED_BANDS_1BASED = (
    *range(108, 113),
    *range(154, 168),
    224,
)


CLASS_NAMES = {
    "indian": {
        1: "Alfalfa",
        2: "Corn-notill",
        3: "Corn-mintill",
        4: "Corn",
        5: "Grass-pasture",
        6: "Grass-trees",
        7: "Grass-pasture-mowed",
        8: "Hay-windrowed",
        9: "Oats",
        10: "Soybean-notill",
        11: "Soybean-mintill",
        12: "Soybean-clean",
        13: "Wheat",
        14: "Woods",
        15: "Buildings-Grass-Trees-Drives",
        16: "Stone-Steel-Towers",
    },
    "salinas_a": {
        1: "Brocoli_green_weeds_1",
        2: "Corn_senesced_green_weeds",
        3: "Lettuce_romaine_4wk",
        4: "Lettuce_romaine_5wk",
        5: "Lettuce_romaine_6wk",
        6: "Lettuce_romaine_7wk",
    },
    "salinas": {
        1: "Brocoli_green_weeds_1",
        2: "Brocoli_green_weeds_2",
        3: "Fallow",
        4: "Fallow_rough_plow",
        5: "Fallow_smooth",
        6: "Stubble",
        7: "Celery",
        8: "Grapes_untrained",
        9: "Soil_vinyard_develop",
        10: "Corn_senesced_green_weeds",
        11: "Lettuce_romaine_4wk",
        12: "Lettuce_romaine_5wk",
        13: "Lettuce_romaine_6wk",
        14: "Lettuce_romaine_7wk",
        15: "Vinyard_untrained",
        16: "Vinyard_vertical_trellis",
    },
    "pavia": {
        1: "Water",
        2: "Trees",
        3: "Asphalt",
        4: "Self-Blocking Bricks",
        5: "Bitumen",
        6: "Tiles",
        7: "Shadows",
        8: "Meadows",
        9: "Bare Soil",
    },
}


def list_mat_variables(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    return [(name, tuple(shape), dtype) for name, shape, dtype in whosmat(path)]


def choose_variable(
    path: Path,
    ndim: int,
    preferred_names: tuple[str, ...],
    requested_name: str | None = None,
    shape_prefix: tuple[int, int] | None = None,
) -> str:
    variables = list_mat_variables(path)
    names = {name for name, _, _ in variables}

    if requested_name:
        if requested_name not in names:
            raise KeyError(f"{path} does not contain variable {requested_name!r}; found {sorted(names)}")
        return requested_name

    for name in preferred_names:
        if name in names:
            return name

    candidates = []
    for name, shape, _ in variables:
        if len(shape) != ndim:
            continue
        if shape_prefix is not None and tuple(shape[:2]) != shape_prefix:
            continue
        candidates.append((name, shape))

    if not candidates:
        raise KeyError(f"{path} has no {ndim}-D variable; found {variables}")

    return max(candidates, key=lambda item: int(np.prod(item[1])))[0]


def load_cube(path: Path, variable_name: str | None = None) -> tuple[np.ndarray, str]:
    variable = choose_variable(path, 3, DEFAULT_CUBE_NAMES, variable_name)
    cube = loadmat(path, variable_names=[variable])[variable]
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D cube, got {variable} with shape {cube.shape}")
    return cube, variable


def load_ground_truth(
    path: Path,
    cube_shape: tuple[int, int],
    variable_name: str | None = None,
) -> tuple[np.ndarray, str]:
    variable = choose_variable(path, 2, DEFAULT_GT_NAMES, variable_name, shape_prefix=cube_shape)
    gt = loadmat(path, variable_names=[variable])[variable]
    if gt.shape != cube_shape:
        raise ValueError(f"Ground truth shape {gt.shape} does not match image shape {cube_shape}")
    return gt, variable


def auto_rgb_bands(band_count: int) -> tuple[int, int, int]:
    """Return 1-based RGB band numbers that are valid for any EHU cube."""
    red = max(1, min(band_count, round(band_count * 0.65)))
    green = max(1, min(band_count, round(band_count * 0.40)))
    blue = max(1, min(band_count, round(band_count * 0.20)))
    return red, green, blue


def aviris_wavelengths_nm(
    total_bands: int = 224,
    start_nm: float = 400.0,
    end_nm: float = 2500.0,
    removed_bands_1based: tuple[int, ...] = (),
) -> np.ndarray:
    """Approximate AVIRIS band-center wavelengths in nm.

    EHU .mat files do not include per-band wavelength metadata. For academic
    reporting, this approximation maps AVIRIS 224 bands linearly from
    400 nm to 2500 nm and removes the bands documented by EHU.
    """
    wavelengths = np.linspace(start_nm, end_nm, total_bands, dtype=np.float32)
    if not removed_bands_1based:
        return wavelengths

    keep = np.ones(total_bands, dtype=bool)
    for band in removed_bands_1based:
        if 1 <= band <= total_bands:
            keep[band - 1] = False
    return wavelengths[keep]


def salinas_corrected_wavelengths_nm(band_count: int) -> np.ndarray | None:
    wavelengths = aviris_wavelengths_nm(removed_bands_1based=SALINAS_REMOVED_BANDS_1BASED)
    if wavelengths.size == band_count:
        return wavelengths
    return None


def default_false_color_bands(dataset_key: str | None, band_count: int) -> tuple[int, int, int]:
    """Return 1-based RGB band numbers with a dataset-aware Salinas default."""
    if dataset_key == "salinas" and band_count >= 57:
        # NIR, red and green bands: vegetation appears bright in the red channel.
        return 57, 27, 17
    return auto_rgb_bands(band_count)


def band_wavelength_labels(
    dataset_key: str | None,
    bands_1based: tuple[int, int, int],
    band_count: int,
) -> list[str]:
    wavelengths = salinas_corrected_wavelengths_nm(band_count) if dataset_key == "salinas" else None
    labels = []
    for band in bands_1based:
        if wavelengths is not None and 1 <= band <= len(wavelengths):
            labels.append(f"{band} (~{wavelengths[band - 1]:.0f} nm)")
        else:
            labels.append(str(band))
    return labels


def spectral_axis(
    dataset_key: str | None,
    band_count: int,
    prefer_wavelength: bool = True,
) -> tuple[np.ndarray, str]:
    """Return x-axis values and label for spectral plots."""
    if prefer_wavelength and dataset_key == "salinas":
        wavelengths = salinas_corrected_wavelengths_nm(band_count)
        if wavelengths is not None:
            return wavelengths, "Approximate wavelength (nm)"
    return np.arange(1, band_count + 1), "Band number"


def infer_dataset_key(*names: str) -> str | None:
    text = " ".join(names).lower()
    if "salinas_a" in text or "salinas-a" in text:
        return "salinas_a"
    if "salinas" in text:
        return "salinas"
    if "indian" in text:
        return "indian"
    if "pavia" in text:
        return "pavia"
    return None


def label_for_class(dataset_key: str | None, class_id: int) -> str:
    if dataset_key in CLASS_NAMES and class_id in CLASS_NAMES[dataset_key]:
        return CLASS_NAMES[dataset_key][class_id]
    return f"Class {class_id}"


def parse_classes(text: str | None, gt: np.ndarray, limit: int = 5) -> list[int]:
    if text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    ids, counts = np.unique(gt, return_counts=True)
    pairs = [(int(class_id), int(count)) for class_id, count in zip(ids, counts) if class_id != 0]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [class_id for class_id, _ in pairs[:limit]]
