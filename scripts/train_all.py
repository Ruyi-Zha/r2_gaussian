# Script to train all cases.

import os
import os.path as osp
import glob
import subprocess
import argparse


def main(args):
    source_path = args.source
    output_path = args.output
    device = args.device
    config_path = args.config

    case_paths = sorted(glob.glob(osp.join(source_path, "*")))

    if len(case_paths) == 0:
        raise ValueError("{} find no folder!".format(case_paths))

    for case_path in case_paths:
        case_name = osp.basename(case_path)
        case_output_path = f"{output_path}/{case_name}"
        if not osp.exists(case_output_path):
            cmd = f"CUDA_VISIBLE_DEVICES={device} python train.py -s {case_path} -m {case_output_path}"
            if config_path:
                cmd += f" --config {config_path}"
            os.system(cmd)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/synthetic_dataset/cone_ntrain_50_angle_360", type=str, help="Path to ct dataset.")
    parser.add_argument("--output", default="output/synthetic_dataset/cone_ntrain_50_angle_360", type=str, help="Path to output.")
    parser.add_argument("--config", default=None, type=str, help="Path to config.")
    parser.add_argument("--device", default=0, type=int, help="GPU device.")
    # fmt: on

    args = parser.parse_args()
    main(args)
