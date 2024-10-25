# Script to convert our format to NAF format (*.pickle)

import os
import sys
import glob
import os.path as osp
import json
import numpy as np
import copy
import pickle
from tqdm import tqdm

# data_path = "data/synthetic_dataset/cone_ntrain_75_angle_360"
# output_path = "data/synthetic_dataset_naf_format/cone_ntrain_75_angle_360"

data_path = "data/real_dataset/cone_ntrain_75_angle_360"
output_path = "data/real_dataset_naf_format/cone_ntrain_75_angle_360"

case_path_list = sorted(glob.glob(osp.join(data_path, "*/")))
os.makedirs(output_path, exist_ok=True)

for case_path in tqdm(case_path_list):
    case_path = case_path[:-1]
    case_name = osp.basename(case_path)
    meta_data_path = osp.join(case_path, "meta_data.json")
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)
    projs_train = np.stack(
        [np.load(osp.join(case_path, m["file_path"])) for m in meta_data["proj_train"]],
        axis=0,
    )
    angles_train = np.stack([m["angle"] for m in meta_data["proj_train"]], axis=0)
    n_train = angles_train.shape[0]
    projs_test = np.stack(
        [np.load(osp.join(case_path, m["file_path"])) for m in meta_data["proj_test"]],
        axis=0,
    )
    angles_test = np.stack([m["angle"] for m in meta_data["proj_test"]], axis=0)
    n_test = angles_test.shape[0]

    img = np.load(osp.join(case_path, meta_data["vol"]))

    scanner_cfg = meta_data["scanner"]
    pkl_dict = copy.deepcopy(scanner_cfg)
    # fmt: off
    pkl_dict.update(
        {
            "numTrain": n_train,
            "numVal": n_test,
            "dDetector": (np.array(scanner_cfg["sDetector"]) / np.array(scanner_cfg["nDetector"]) * 1000).tolist(), # in mm
            "dVoxel": (np.array(scanner_cfg["sVoxel"]) / np.array(scanner_cfg["nVoxel"])* 1000).tolist(), # in mm
            "train": {
                "projections": projs_train,
                "angles": angles_train,
            },
            "val": {
                "projections": projs_test,
                "angles": angles_test,
            },
            "image": img,
        }
    )
    pkl_dict["DSD"] = (np.array(pkl_dict["DSD"]) * 1000).tolist()  # in mm
    pkl_dict["DSO"] = (np.array(pkl_dict["DSO"]) * 1000).tolist()  # in mm
    pkl_dict["sDetector"] =  (np.array(pkl_dict["sDetector"]) * 1000).tolist()  # in mm
    pkl_dict["sVoxel"] =  (np.array(pkl_dict["sVoxel"]) * 1000).tolist()  # in mm
    pkl_dict["offOrigin"] = (np.array(pkl_dict["offOrigin"]) * 1000).tolist()  # in mm
    pkl_dict["offDetector"] = (np.array(pkl_dict["offDetector"]) * 1000).tolist()  # in mm

    # fmt: on
    with open(osp.join(output_path, f"{case_name}.pickle"), "wb") as f:
        pickle.dump(pkl_dict, f, pickle.HIGHEST_PROTOCOL)
