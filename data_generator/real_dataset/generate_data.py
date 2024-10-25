import os
import os.path as osp
import sys
import argparse
import glob
import numpy as np
from tqdm import trange
import tigre.algorithms as algs
import scipy
import cv2
import random
import json

random.seed(0)

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry


def main(args):
    input_data_path = args.data
    proj_subsample = args.proj_subsample
    proj_rescale = args.proj_rescale
    object_scale = args.object_scale

    # Read configuration
    config_file_path = osp.join(input_data_path, "config.txt")
    with open(config_file_path, "r") as f:
        for config_line in f.readlines():
            if "NumberImages" in config_line:
                n_proj = int(config_line.split("=")[-1])
            elif "AngleInterval" in config_line:
                angle_interval = float(config_line.split("=")[-1])
            elif "AngleFirst" in config_line:
                angle_start = float(config_line.split("=")[-1])
            elif "AngleLast" in config_line:
                angle_last = float(config_line.split("=")[-1])
            elif "DistanceSourceDetector" in config_line:
                DSD = float(config_line.split("=")[-1]) / 1000 * object_scale
            elif "DistanceSourceOrigin" in config_line:
                DSO = float(config_line.split("=")[-1]) / 1000 * object_scale
            elif "PixelSize" in config_line and "PixelSizeUnit" not in config_line:
                dDetector = (
                    float(config_line.split("=")[-1])
                    * proj_subsample
                    / 1000
                    * object_scale
                )
    angles = np.concatenate(
        [np.arange(angle_start, angle_last, angle_interval), [angle_last]]
    )
    angles = angles / 180.0 * np.pi

    # Read and save projections
    output_path = args.output
    all_save_path = osp.join(output_path, "proj_all")
    train_save_path = osp.join(output_path, "proj_train")
    test_save_path = osp.join(output_path, "proj_test")
    os.makedirs(all_save_path, exist_ok=True)
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    proj_mat_paths = sorted(glob.glob(osp.join(input_data_path, "*.mat")))
    projection_train_list = []
    projection_test_list = []
    train_ids = np.linspace(0, n_proj - 1, args.n_train).astype(int)
    test_ids = sorted(
        random.sample(np.setdiff1d(np.arange(n_proj), train_ids).tolist(), args.n_test)
    )
    for i_proj in trange(len(proj_mat_paths), desc=osp.basename(output_path)):
        proj_mat_path = proj_mat_paths[i_proj]
        proj_save_name = osp.basename(proj_mat_path).split(".")[0]
        if i_proj in train_ids:
            projection_train_list.append(
                {
                    "file_path": osp.join(
                        osp.basename(train_save_path), proj_save_name + ".npy"
                    ),
                    "angle": angles[i_proj],
                }
            )
        elif i_proj in test_ids:
            projection_test_list.append(
                {
                    "file_path": osp.join(
                        osp.basename(test_save_path), proj_save_name + ".npy"
                    ),
                    "angle": angles[i_proj],
                }
            )

        proj = scipy.io.loadmat(proj_mat_path)["img"] / proj_rescale * object_scale
        proj = proj.astype(np.float32)
        proj[proj < 0] = 0
        # Shift left for 5 pixels according to dataset description
        proj_new = np.zeros_like(proj)
        proj_new[:-5] = proj[5:]
        proj = proj_new
        if proj_subsample != 1.0:
            h_ori, w_ori = proj.shape
            h_new, w_new = int(h_ori / proj_subsample), int(w_ori / proj_subsample)
            proj = cv2.resize(proj, [w_new, h_new])
            # crop to rectangle
            dim_x, dim_y = proj.shape
            if dim_x > dim_y:
                dim_offset = int((dim_x - dim_y) / 2)
                proj = proj[dim_offset:-dim_offset, :]
            elif dim_x < dim_y:
                dim_offset = int((dim_y - dim_x) / 2)
                proj = proj[:, dim_offset:-dim_offset]

        np.save(osp.join(all_save_path, proj_save_name + ".npy"), proj)
        if i_proj in train_ids:
            np.save(osp.join(train_save_path, proj_save_name + ".npy"), proj)
        elif i_proj in test_ids:
            np.save(osp.join(test_save_path, proj_save_name + ".npy"), proj)

    # Scanner config
    proj = np.load(osp.join(output_path, projection_train_list[0]["file_path"]))
    nDetector = [proj.shape[0], proj.shape[1]]
    sDetector = np.array(nDetector) * np.array(dDetector)
    nVoxel = args.nVoxel
    sVoxel = args.sVoxel
    offOrigin = args.offOrigin
    bbox = np.array(
        [
            np.array(offOrigin) - np.array(sVoxel) / 2,
            np.array(offOrigin) + np.array(sVoxel) / 2,
        ]
    ).tolist()
    scanner_cfg = {
        "mode": "cone",
        "DSD": DSD,
        "DSO": DSO,
        "nDetector": nDetector,
        "sDetector": sDetector.tolist(),
        "nVoxel": nVoxel,
        "sVoxel": sVoxel,
        "offOrigin": offOrigin,
        "offDetector": args.offDetector,
        "accuracy": args.accuracy,
        "totalAngle": angle_last - angle_start,
        "startAngle": angle_start,
        "noise": True,
        "filter": None,
    }

    # Reconstruct with FDK as gt
    ct_gt_save_path = osp.join(output_path, "vol_gt.npy")
    if not osp.exists(ct_gt_save_path):
        projs = []
        skip = 1
        proj_paths = sorted(glob.glob(osp.join(all_save_path, "*.npy")))
        for proj_path in proj_paths[::skip]:
            proj = np.load(proj_path)
            nDetector = proj.shape
            projs.append(proj)
        projs = np.stack(projs, axis=0)
        print("reconstruct with FDK")
        geo = get_geometry(scanner_cfg)
        ct_gt = algs.fdk(projs[:, ::-1, :], geo, angles[::skip])
        ct_gt = ct_gt.transpose((2, 1, 0))
        ct_gt[ct_gt < 0] = 0
        np.save(ct_gt_save_path, ct_gt)

    # Save
    meta_data = {
        "scanner": scanner_cfg,
        "ct": "vol_gt.npy",
        "radius": 1.0,
        "bbox": bbox,
        "proj_train": projection_train_list,
        "proj_test": projection_test_list,
    }
    with open(osp.join(output_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    print(f"Data saved in {output_path}")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to FIPS processed data.")
    parser.add_argument("--output", type=str, help="Path to output.")
    parser.add_argument("--proj_subsample", default=4, type=int, help="subsample projections pixels")
    parser.add_argument("--proj_rescale", default=400.0, type=float, help="rescale projection values to fit density to around [0,1]")
    parser.add_argument("--object_scale", default=50, type=int, help="subsample number of views as sparse-view")
    parser.add_argument("--n_test", default=100, type=int, help="number of test")
    parser.add_argument("--n_train", default=75, type=int, help="number of train")
    
    parser.add_argument("--nVoxel", nargs="+", default=[256, 256, 256], type=int, help="voxel dimension")
    parser.add_argument("--sVoxel", nargs="+", default=[2.0, 2.0, 2.0], type=float, help="volume size")
    parser.add_argument("--offOrigin", nargs="+", default=[0.0, 0.0, 0.0], type=float, help="offOrigin")
    parser.add_argument("--offDetector", nargs="+", default=[0.0, 0.0], type=float, help="offDetector")
    parser.add_argument("--accuracy", default=0.5, type=float, help="accuracy")
    
    
    args = parser.parse_args()
    main(args)
    # fmt: on
