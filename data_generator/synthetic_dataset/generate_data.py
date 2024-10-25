import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
import plotly.graph_objects as go
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise
import json
import matplotlib.pyplot as plt
import tigre.algorithms as algs
import argparse
import open3d as o3d
import cv2
import pickle
import copy

import sys

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry, recon_volume


def main(args):
    """Assume CT is in a unit cube. We synthesize X-ray projections."""
    vol_path = args.vol
    scanner_cfg_path = args.scanner
    n_train = args.n_train
    n_test = args.n_test
    vol_name = osp.basename(vol_path)[:-4]
    output_path = args.output

    # Load configuration
    with open(scanner_cfg_path, "r") as handle:
        scanner_cfg = yaml.safe_load(handle)

    case_name = f"{vol_name}_{scanner_cfg['mode']}"
    print(f"Generate data for case {case_name}")
    geo = get_geometry(scanner_cfg)

    # Load volume
    vol = np.load(vol_path).astype(np.float32)

    # Generate training projections
    projs_train_angles = (
        np.linspace(0, scanner_cfg["totalAngle"] / 180 * np.pi, n_train + 1)[:-1]
        + scanner_cfg["startAngle"] / 180 * np.pi
    )
    projs_train = tigre.Ax(
        np.transpose(vol, (2, 1, 0)).copy(), geo, projs_train_angles
    )[:, ::-1, :]
    if scanner_cfg["noise"]:
        projs_train = CTnoise.add(
            projs_train,
            Poisson=scanner_cfg["possion_noise"],
            Gaussian=np.array(scanner_cfg["gaussian_noise"]),
        )  #
        projs_train[projs_train < 0.0] = 0.0

    # Generate testing projections (we don't use them in our work)
    projs_test_angles = (
        np.sort(np.random.rand(n_test) * 360 / 180 * np.pi)  # Evaluate full circle
        + scanner_cfg["startAngle"] / 180 * np.pi
    )
    projs_test = tigre.Ax(np.transpose(vol, (2, 1, 0)).copy(), geo, projs_test_angles)[
        :, ::-1, :
    ]

    # Save
    case_save_path = osp.join(output_path, case_name)
    os.makedirs(case_save_path, exist_ok=True)
    np.save(osp.join(case_save_path, "vol_gt.npy"), vol)
    file_path_dict = {}
    for split, projs, angles in zip(
        ["proj_train", "proj_test"],
        [projs_train, projs_test],
        [projs_train_angles, projs_test_angles],
    ):
        os.makedirs(osp.join(case_save_path, split), exist_ok=True)
        file_path_dict[split] = []
        for i_proj in range(projs.shape[0]):
            proj = projs[i_proj]
            frame_save_name = osp.join(split, f"{split}_{i_proj:04d}.npy")
            np.save(osp.join(case_save_path, frame_save_name), proj)
            file_path_dict[split].append(
                {
                    "file_path": frame_save_name,
                    "angle": angles[i_proj],
                }
            )
    meta = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "bbox": [[-1, -1, -1], [1, 1, 1]],
        "proj_train": file_path_dict["proj_train"],
        "proj_test": file_path_dict["proj_test"],
    }
    with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"Generate data for case {case_name} complete!")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Data generator parameters")
    
    parser.add_argument("--vol", default="data_generator/volume_gt/0_chest.npy", type=str, help="Path to volume.")
    parser.add_argument("--scanner", default="data_generator/scanner/cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/cone_ntrain_50_angle_360", type=str, help="Path to output.")
    parser.add_argument("--n_train", default=50, type=int, help="Number of projections for training.")
    parser.add_argument("--n_test", default=100, type=int, help="Number of projections for evaluation.")
    # fmt: on

    args = parser.parse_args()
    main(args)
