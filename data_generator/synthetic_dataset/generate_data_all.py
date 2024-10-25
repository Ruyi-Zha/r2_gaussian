import os
import os.path as osp
import glob
import argparse


def main(args):
    vol_dataset_path = args.vol
    output_path = args.output
    scanner_path = args.scanner
    n_train = args.n_train
    n_test = args.n_test
    device = args.device

    vol_file_paths = sorted(glob.glob(osp.join(vol_dataset_path, "*.npy")))

    if len(vol_file_paths) == 0:
        raise ValueError("{} find no *.npy file!".format(vol_file_paths))

    for vol_file_path in vol_file_paths:
        cmd = f"CUDA_VISIBLE_DEVICES={device} python data_generator/synthetic_dataset/generate_data.py --vol {vol_file_path} --scanner {scanner_path} --output {output_path}  --n_train {n_train} --n_test {n_test}"
        os.system(cmd)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol", default="data_generator/volume_gt", type=str, help="Path to vol dataset.")
    parser.add_argument("--scanner", default="data_generator/scanner/cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/cone_ntrain_50_angle_360", type=str, help="Path to output.")
    parser.add_argument("--n_train", default=50, type=int, help="Number of projections for training.")
    parser.add_argument("--n_test", default=100, type=int, help="Number of projections for test.")
    parser.add_argument("--device", default=0, type=int, help="GPU device.")
    # fmt: on

    args = parser.parse_args()
    main(args)
