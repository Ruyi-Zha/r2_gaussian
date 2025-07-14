import os
import os.path as osp
import argparse
import json
import random
import numpy as np
import h5py
import yaml
import scipy.io


random.seed(0)


def load_scanner_from_args(args, proj_shape, angles, cam_z=None, pixel_size=None):
    if args.scanner is not None:
        with open(args.scanner, "r") as f:
            scanner_cfg = yaml.safe_load(f)
    else:
        # compute detector size
        nDetector = [proj_shape[1], proj_shape[2]]
        px = args.pixel_size if pixel_size is None else float(pixel_size)
        sDetector = (np.array(nDetector) * px).tolist()
        cam_dist = args.DSO if cam_z is None else float(cam_z)
        scanner_cfg = {
            "mode": "parallel",
            "DSD": cam_dist,
            "DSO": cam_dist,
            "nDetector": nDetector,
            "sDetector": sDetector,
            "nVoxel": args.nVoxel,
            "sVoxel": args.sVoxel,
            "offOrigin": args.offOrigin,
            "offDetector": args.offDetector,
            "accuracy": args.accuracy,
            "totalAngle": float(angles.max() - angles.min()),
            "startAngle": float(angles.min()),
            "filter": None,
            "noise": False,
        }
    return scanner_cfg


def main(args):
    with h5py.File(args.h5_file, "r") as f:
        data = f["/entry/data/data"][:]
        angles = f["/entry/instrument/NDAttributes/SAMPLE_MICOS_W2"][:]
        cam_z = f["/entry/instrument/NDAttributes/CT_Camera_Z"][0]
        pixel_size = f["/entry/instrument/NDAttributes/CT_Pixelsize"][0]

    angles = angles.astype(float)

    flat = data[: args.numFF].astype(float)
    FF = flat.mean(axis=0)
    DF = args.darkfield
    proj_raw = data[args.numFF :]
    projections = -np.log(np.clip((proj_raw - DF) / (FF - DF), 1e-6, None))

    if args.shift_px != 0:
        shift = int(args.shift_px)
        if shift > 0:
            pad = np.zeros_like(projections[:, :, :shift])
            projections = np.concatenate([projections[:, :, shift:], pad], axis=2)
        elif shift < 0:
            pad = np.zeros_like(projections[:, :, : -shift])
            projections = np.concatenate([pad, projections[:, :, :shift]], axis=2)

    n_proj = projections.shape[0]
    train_ids = np.linspace(0, n_proj - 1, args.n_train).astype(int)
    remain = list(set(range(n_proj)) - set(train_ids))
    test_ids = sorted(random.sample(remain, args.n_test))

    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    train_path = osp.join(output_path, "proj_train")
    test_path = osp.join(output_path, "proj_test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    if args.save_mat:
        mat_path = osp.join(output_path, "mat")
        os.makedirs(mat_path, exist_ok=True)

    proj_train_list = []
    proj_test_list = []

    for i in range(n_proj):
        proj = projections[i]
        file_name = f"proj_{i:04d}.npy"
        mat_name = f"{i+1:04d}.mat"
        if i in train_ids:
            save_rel = osp.join("proj_train", file_name)
            np.save(osp.join(output_path, save_rel), proj)
            proj_train_list.append({"file_path": save_rel, "angle": float(angles[i])})
        elif i in test_ids:
            save_rel = osp.join("proj_test", file_name)
            np.save(osp.join(output_path, save_rel), proj)
            proj_test_list.append({"file_path": save_rel, "angle": float(angles[i])})
        if args.save_mat:
            scipy.io.savemat(osp.join(mat_path, mat_name), {"img": proj})

    scanner_cfg = load_scanner_from_args(
        args, projections.shape, angles, cam_z=cam_z, pixel_size=pixel_size
    )
    bbox = [
        (np.array(scanner_cfg["offOrigin"]) - np.array(scanner_cfg["sVoxel"]) / 2).tolist(),
        (np.array(scanner_cfg["offOrigin"]) + np.array(scanner_cfg["sVoxel"]) / 2).tolist(),
    ]
    meta = {
        "scanner": scanner_cfg,
        "bbox": bbox,
        "proj_train": proj_train_list,
        "proj_test": proj_test_list,
    }
    with open(osp.join(output_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True, help="Input h5 file")
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument("--scanner", type=str, default=None, help="YAML file of scanner geometry")
    parser.add_argument("--DSD", type=float, default=7.0, help="Distance Source Detector")
    parser.add_argument("--DSO", type=float, default=5.0, help="Distance Source Origin")
    parser.add_argument("--pixel_size", type=float, default=0.008, help="Detector pixel size")
    parser.add_argument("--numFF", type=int, default=20, help="Number of flat field frames")
    parser.add_argument("--darkfield", type=float, default=100, help="Dark field value")
    parser.add_argument("--nVoxel", nargs="+", type=int, default=[256, 256, 256], help="Voxel dimension")
    parser.add_argument("--sVoxel", nargs="+", type=float, default=[2.0, 2.0, 2.0], help="Volume size")
    parser.add_argument("--offOrigin", nargs="+", type=float, default=[0.0, 0.0, 0.0], help="Origin offset")
    parser.add_argument("--offDetector", nargs="+", type=float, default=[0.0, 0.0], help="Detector offset")
    parser.add_argument("--accuracy", type=float, default=0.5, help="Projection accuracy")
    parser.add_argument("--n_train", type=int, default=50, help="Number of training projections")
    parser.add_argument("--n_test", type=int, default=100, help="Number of testing projections")
    parser.add_argument("--save_mat", action="store_true", help="Save processed projections as .mat files")
    parser.add_argument("--shift_px", type=int, default=0, help="Pixel shift for center-of-rotation correction")
    args = parser.parse_args()
    main(args)
