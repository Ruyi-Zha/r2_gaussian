""" 
Process raw CT data.
1. Normalize to [0, 1] and a cube.
2. Save as *.npy format.
"""

import os
import os.path as osp
import numpy as np
import pydicom
import argparse
import glob
import scipy.ndimage as ndimage
from tqdm import tqdm, trange
import sys
import tifffile
import importlib

sys.path.append("./")


def main(args):
    metadata_path = args.metadata
    output_path = args.output
    target_size = args.target_size
    os.makedirs(output_path, exist_ok=True)

    spec = importlib.util.spec_from_file_location("metadata", metadata_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw_info = module.raw_info

    n_case = len(raw_info)
    pbar = tqdm(np.arange(n_case))
    for i_case in pbar:
        case_info = raw_info[i_case]
        case_output_name = case_info["output_name"]
        pbar.set_description(case_output_name)
        file_type = case_info["file_type"]
        case_output_path = osp.join(output_path, f"{case_output_name}.npy")
        if not osp.exists(case_output_path):
            if file_type == "dcm":
                vol_out = process_dcm(case_info, target_size)
            elif file_type == "raw":
                vol_out = process_raw(case_info, target_size)
            elif file_type == "tif":
                vol_out = process_tif(case_info, target_size)
            else:
                raise ValueError("Unsupported file type")
            np.save(case_output_path, vol_out.astype(np.float32))


def process_dcm(case_info, target_size):
    """Process *.dcm files."""
    thickness = case_info["thickness"]
    dcm_path = sorted(glob.glob(osp.join(case_info["raw_path"], "*.dcm")))
    slices = []
    for d_path in dcm_path:
        ds = pydicom.dcmread(d_path)
        slice = np.array(ds.pixel_array).astype(float) * float(ds.RescaleSlope) + float(
            ds.RescaleIntercept
        )
        slices.append(slice)
    vol = np.stack(slices, axis=-1)
    vol = vol[:, :, ::-1]  # Upside down
    vol = vol.clip(-1000, 2000)  # From air to bone
    vol_min = vol.min()
    vol_max = vol.max()
    vol = (vol - vol_min) / (vol_max - vol_min)
    slice_thickness = ds.SliceThickness
    if thickness is None:
        thickness = ds.SliceThickness
    pixel_spacing = [float(i) for i in list(ds.PixelSpacing)]
    voxel_spacing = np.array(pixel_spacing + [slice_thickness])
    vol_new = reshape_vol(vol, voxel_spacing, target_size, None)
    vol_new = vol_new.clip(0.0, 1.0)
    if case_info["xy_invert"]:
        vol_new = vol_new[::-1, ::-1, :]
    return vol_new


def process_raw(case_info, target_size):
    """Process *.raw file."""
    data = (
        np.fromfile(case_info["raw_path"], dtype=case_info["dtype"])
        .reshape(case_info["shape"][::-1])
        .astype(float)
    )
    data = data.transpose([2, 1, 0])
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)
    data = data.clip(0.0, 1.0)
    data = reshape_vol(
        data, case_info["spacing"], target_size, mode=case_info["reshape"]
    )
    data = data.clip(0.0, 1.0)
    data = data.transpose(case_info["transpose"])
    if case_info["z_invert"]:
        data = data[:, :, ::-1]
    return data


def process_tif(case_info, target_size):
    """Process *.tif file."""
    data = tifffile.imread(case_info["raw_path"])
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)
    data = reshape_vol(
        data, case_info["spacing"], target_size, mode=case_info["reshape"]
    )
    data = data.clip(0.0, 1.0)
    data = data.transpose(case_info["transpose"])
    if case_info["z_invert"]:
        data = data[:, :, ::-1]
    return data


def reshape_vol(image, spacing, target_size, mode=None):
    """Reshape a CT volume."""

    if mode is not None:
        image, _ = resample(image, spacing, [1, 1, 1])
        if mode == "crop":
            image = crop_to_cube(image)
        elif mode == "expand":
            image = expand_to_cube(image)
        else:
            raise ValueError("Unsupported reshape mode!")

    image_new = resize(image, target_size)
    return image_new


def expand_to_cube(array):
    # Step 1: Find the maximum dimension
    max_dim = max(array.shape)

    # Step 2: Calculate the padding for each dimension
    padding = [(max_dim - s) // 2 for s in array.shape]
    # For odd differences, add an extra padding at the end
    padding = [(pad, max_dim - s - pad) for pad, s in zip(padding, array.shape)]

    # Step 3: Pad the array to get the cubic shape
    cubic_array = np.pad(
        array, padding, mode="constant", constant_values=0
    )  # Using zero padding

    return cubic_array


def crop_to_cube(array):
    # Step 1: Find the minimum dimension
    min_dim = min(array.shape)

    # Step 2: Define the start and end indices for cropping
    start_indices = [(dim_size - min_dim) // 2 for dim_size in array.shape]
    end_indices = [start + min_dim for start in start_indices]

    # Step 3: Crop the array to get the cubic region
    cubic_region = array[
        start_indices[0] : end_indices[0],
        start_indices[1] : end_indices[1],
        start_indices[2] : end_indices[2],
    ]

    return cubic_region


def resample(image, spacing, new_spacing=[1, 1, 1]):
    """Resample to stantard spacing (keep physical scale stable, change pixel numbers)"""
    # .mhd image order : z, y, x
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)

    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = ndimage.zoom(image, real_resize_factor, mode="nearest")
    return image, new_spacing


def resize(scan, target_size):
    """Resize the scan based on given voxel dimension."""
    scan_x, scan_y, scan_z = scan.shape
    zoom_x = target_size / scan_x
    zoom_y = target_size / scan_y
    zoom_z = target_size / scan_z

    if zoom_x != 1.0 or zoom_y != 1.0 or zoom_z != 1.0:
        scan = ndimage.zoom(
            scan,
            (zoom_x, zoom_y, zoom_z),
            mode="nearest",
        )
    return scan


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data_generator/raw_metadata.py", type=str, help="Path to metadata.")
    parser.add_argument("--output", default="data_generator/volume_gt", type=str, help="Path to output folder.")
    parser.add_argument("--target_size", default=256, type=int, help="Target volume size (a cube)")

    # fmt: on

    args = parser.parse_args()

    main(args)
