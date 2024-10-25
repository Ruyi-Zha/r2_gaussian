import os
import sys

import os.path as osp
import glob
import subprocess
import argparse


sys.path.append("./")
from initialize_pcd import InitParams


def main(args, init_parser: InitParams):
    dataset_path = args.data
    device = args.device

    case_paths = sorted(glob.glob(osp.join(dataset_path, "*/")))

    if len(case_paths) == 0:
        raise ValueError("{} find no case!".format(case_paths))

    for case_path in case_paths:
        case_path = case_path[:-1]
        case_name = osp.basename(case_path)
        init_output_path = osp.join(
            case_path,
            f"init_{case_name}.npy",
        )
        os.makedirs(osp.dirname(init_output_path), exist_ok=True)
        print(f"Initialization for {osp.basename(case_path)} start.")
        cmd = f"CUDA_VISIBLE_DEVICES={device} python initialize_pcd.py --data {case_path} --output {init_output_path}"
        init_args = vars(init_parser)
        for var in init_args:
            cmd += f" --{var} {init_args[var]}"
        os.system(cmd)
        print(f"Initialization for {osp.basename(case_path)} complete.")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    init_parser = InitParams(parser)
    parser.add_argument("--data", type=str, help="Path to dataset.")
    parser.add_argument("--device", default=0, type=int, help="GPU device.")
    # fmt: on

    args = parser.parse_args()
    main(args, init_parser.extract(args))
