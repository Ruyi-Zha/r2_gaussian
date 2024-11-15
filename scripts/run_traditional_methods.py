# There is a np.int bug in TIGREv2.3. You need to change np.int to int manually.
# Usage example: python scripts/run_traditional_methods.py -s 0_chest_cone -m output/0_chest_cone_trad

import os
import os.path as osp
import numpy as np
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import tigre
import yaml
import copy

sys.path.append("./")
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.general_utils import t2a, safe_state
from r2_gaussian.utils.ct_utils import get_geometry_tigre, run_ct_recon_algs
from r2_gaussian.arguments import ModelParams


def main(dataset: ModelParams):

    # Set up dataset
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg
    geo = get_geometry_tigre(scanner_cfg)

    projs_train = np.concatenate(
        [t2a(c.original_image) for c in scene.getTrainCameras()],
        axis=0,
    )
    projs_test = np.concatenate(
        [t2a(c.original_image) for c in scene.getTestCameras()],
        axis=0,
    )
    train_angles = np.stack([c.angle for c in scene.getTrainCameras()], axis=0)
    test_angles = np.stack([c.angle for c in scene.getTestCameras()], axis=0)

    vol_gt = t2a(scene.vol_gt)
    save_path = dataset.model_path

    out_dict = {}

    data_name = osp.basename(dataset.source_path)

    print("Run traditional algorithms on {}".format(data_name))
    methods = ["fdk", "sart", "asd_pocs"]
    for method in methods:
        out_dict[method], ct_pred, _ = run_ct_recon_algs(
            projs_train, train_angles, copy.deepcopy(geo), vol_gt, save_path, method
        )
        # Render projections in test
        projs_test_pred = tigre.Ax(
            np.transpose(ct_pred, (2, 1, 0)).copy(), copy.deepcopy(geo), test_angles
        )[:, ::-1, :]
        proj_save_path = osp.join(save_path, method, "projs")
        os.makedirs(proj_save_path, exist_ok=True)
        for i_proj in range(projs_test_pred.shape[0]):
            np.save(
                osp.join(proj_save_path, "{0:05d}_render.npy".format(i_proj)),
                projs_test_pred[i_proj],
            )
            np.save(
                osp.join(proj_save_path, "{0:05d}_gt.npy".format(i_proj)),
                projs_test[i_proj],
            )
            plt.imsave(
                osp.join(proj_save_path, "{0:05d}_render.png".format(i_proj)),
                projs_test_pred[i_proj],
                cmap="gray",
            )
            plt.imsave(
                osp.join(proj_save_path, "{0:05d}_gt.png".format(i_proj)),
                projs_test[i_proj],
                cmap="gray",
            )

    with open(osp.join(save_path, "eval_3d.yml"), "w") as f:
        yaml.dump(out_dict, f, default_flow_style=False, sort_keys=False)

    print("Run traditional algorithms on {} complete".format(data_name))


if __name__ == "__main__":
    # fmt: off
    parser = ArgumentParser(description="Traditional method script parameters")
    model = ModelParams(parser)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    # fmt: on
    safe_state(args.quiet)

    main(model.extract(args))
