import os
import os.path as osp
import glob
import argparse


def main(args):
    input_path = args.data
    output_path = args.output
    device = args.device

    input_case_paths = sorted(glob.glob(input_path + "/*/"))

    if len(input_case_paths) == 0:
        raise ValueError("{} find no case!".format(input_case_paths))

    for input_case_path in input_case_paths:
        output_case_path = osp.join(
            output_path,
            f"cone_ntrain_{args.n_train}_angle_360",
            osp.basename(input_case_path[:-1]),
        )
        os.makedirs(output_case_path, exist_ok=True)
        cmd = (
            f"CUDA_VISIBLE_DEVICES={device} python data_generator/real_dataset/generate_data.py "
            + f"--data {input_case_path} "
            + f"--output {output_case_path} "
            + f"--proj_subsample {args.proj_subsample} "
            + f"--proj_rescale {args.proj_rescale} "
            + f"--object_scale {args.object_scale} "
            + f"--n_test {args.n_test} "
            + f"--n_train {args.n_train} "
            + f"--nVoxel {args.nVoxel[0]} {args.nVoxel[1]} {args.nVoxel[2]} "
            + f"--sVoxel {args.sVoxel[0]} {args.sVoxel[1]} {args.sVoxel[2]} "
            + f"--offOrigin {args.offOrigin[0]} {args.offOrigin[1]} {args.offOrigin[2]} "
            + f"--offDetector {args.offDetector[0]} {args.offDetector[1]} "
            + f"--accuracy {args.accuracy}"
        )
        os.system(cmd)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_generator/real_dataset/FIPS_processed", type=str, help="Path to FIPS processed data.")
    parser.add_argument("--output", default="data/real_dataset", type=str, help="Path to output.")
    
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
    
    parser.add_argument("--device", default=0, type=int, help="GPU device.")
    # fmt: on

    args = parser.parse_args()
    main(args)
