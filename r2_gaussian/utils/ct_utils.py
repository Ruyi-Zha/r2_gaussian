import os
import sys
import numpy as np
import tigre
import os.path as osp
import yaml
import time
import tigre.algorithms as algs
from tqdm import trange
from tigre.utilities.im3Dnorm import im3DNORM
import matplotlib.pyplot as plt

sys.path.append("./")
from r2_gaussian.utils.image_utils import metric_vol


def recon_volume(projs, angles, geo, recon_method):
    """Reconstruct ct with traditional methods."""
    if recon_method == "fdk":
        vol = algs.fdk(projs[:, ::-1, :], geo, angles)
    elif recon_method == "cgls":
        vol, _ = algs.cgls(projs[:, ::-1, :], geo, angles, 60, computel2=True)
    else:
        raise ValueError("Unsupported reconstruction method")
    vol = np.transpose(vol, (2, 1, 0))
    return vol


def get_geometry_tigre(cfg):
    """For TIGRE only."""
    if cfg["mode"] == "parallel":
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(cfg["nVoxel"][::-1]))
    elif cfg["mode"] == "cone":
        geo = tigre.geometry(mode="cone")
    else:
        raise NotImplementedError("Unsupported scanner mode!")

    geo.DSD = cfg["DSD"]  # Distance Source Detector
    geo.DSO = cfg["DSO"]  # Distance Source Origin
    # Detector parameters
    geo.nDetector = np.array(cfg["nDetector"])  # number of pixels
    geo.sDetector = np.array(cfg["sDetector"])  # size of each pixel
    geo.dDetector = geo.sDetector / geo.nDetector  # total size of the detector
    # Image parameters
    geo.nVoxel = np.array(cfg["nVoxel"][::-1])  # number of voxels
    geo.sVoxel = np.array(cfg["sVoxel"][::-1])  # size of each voxel
    geo.dVoxel = geo.sVoxel / geo.nVoxel  # total size of the image
    # Offsets
    geo.offOrigin = np.array(cfg["offOrigin"][::-1])  # Offset of image from origin
    geo.offDetector = np.array(
        [cfg["offDetector"][1], cfg["offDetector"][0], 0]
    )  # Offset of Detector
    # Auxiliary
    geo.accuracy = cfg["accuracy"]  # Accuracy of FWD proj
    # Mode
    geo.filter = cfg["filter"]
    return geo


def run_ct_recon_algs(projs, angles, geo, ct_gt, save_path, method):
    print("Run {}...".format(method))
    save_path = osp.join(save_path, method)
    slice_save_path = osp.join(save_path, "slice_{}".format(method))
    os.makedirs(slice_save_path, exist_ok=True)
    start_time = time.time()

    if method == "fdk":
        ct_pred = algs.fdk(projs[:, ::-1, :], geo, angles)
    elif method == "sart":
        lmbda = 1
        lambdared = 0.999
        initmode = None
        verbose = True
        qualmeas = ["RMSE"]
        blcks = 10
        order = "ordered"
        ct_pred, _ = algs.sart(
            projs[:, ::-1, :],
            geo,
            angles,
            20,
            lmbda=lmbda,
            lmbda_red=lambdared,
            verbose=verbose,
            Quameasopts=qualmeas,
            computel2=True,
        )
    elif method == "ossart":
        lmbda = 1
        lambdared = 0.999
        initmode = None
        verbose = True
        qualmeas = ["RMSE"]
        blcks = 10
        order = "ordered"
        ct_pred, qualityOSSART = algs.ossart(
            projs[:, ::-1, :],
            geo,
            angles,
            20,
            lmbda=lmbda,
            lmbda_red=lambdared,
            verbose=verbose,
            Quameasopts=qualmeas,
            computel2=False,
            blocksize=blcks,
            OrderStrategy=order,
        )
    elif method == "asd_pocs":
        epsilon = (
            im3DNORM(
                tigre.Ax(algs.fdk(projs[:, ::-1, :], geo, angles), geo, angles)
                - projs[:, ::-1, :],
                2,
            )
            * 0.15
        )
        alpha = 0.002
        ng = 20
        lmbda = 1
        lambdared = 0.9999
        alpha_red = 0.95
        ratio = 0.94
        verb = True
        order = "ordered"
        ct_pred = algs.asd_pocs(
            projs[:, ::-1, :],
            geo,
            angles,
            10,  # these are very important
            tviter=ng,
            maxl2err=epsilon,
            alpha=alpha,  # less important.
            lmbda=lmbda,
            lmbda_red=lambdared,
            rmax=ratio,
            verbose=verb,
        )
    elif method == "os_asd_pocs":
        epsilon = (
            im3DNORM(
                tigre.Ax(algs.fdk(projs[:, ::-1, :], geo, angles), geo, angles)
                - projs[:, ::-1, :],
                2,
            )
            * 0.15
        )
        alpha = 0.002
        ng = 20
        lmbda = 1
        lambdared = 0.9999
        alpha_red = 0.95
        ratio = 0.94
        verb = True
        order = "ordered"
        blcks = 10
        ct_pred = algs.os_asd_pocs(
            projs[:, ::-1, :],
            geo,
            angles,
            10,  # these are very important
            tviter=ng,
            maxl2err=epsilon,
            alpha=alpha,  # less important.
            lmbda=lmbda,
            lmbda_red=lambdared,
            rmax=ratio,
            verbose=verb,
            OrderStrategy=order,
            blocksize=blcks,
        )
    elif method == "cgls":
        ct_pred, _ = algs.cgls(projs[:, ::-1, :], geo, angles, 60, computel2=True)
    else:
        raise NotImplementedError("Unsupported reconstruction method!")
    ct_pred = ct_pred.transpose((2, 1, 0))

    duration = time.time() - start_time
    psnr_3d, _ = metric_vol(ct_gt, ct_pred, "psnr")
    ssim_3d, ssim_3d_axis = metric_vol(ct_gt, ct_pred, "ssim")

    np.save(osp.join(save_path, "ct_gt.npy"), ct_gt)
    np.save(osp.join(save_path, "ct_pred.npy"), ct_pred)

    n_slice = ct_gt.shape[2]
    for i_slice in trange(n_slice, desc="[{}] Save slice".format(method), leave=False):
        plt.imsave(
            osp.join(slice_save_path, "{0:05d}_gt.png".format(i_slice)),
            ct_gt[:, :, i_slice],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.imsave(
            osp.join(slice_save_path, "{0:05d}_pred.png".format(i_slice)),
            ct_pred[:, :, i_slice],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    report_dict = {
        "method": method,
        "psnr_3d": psnr_3d,
        "ssim_3d": float(ssim_3d),
        "ssim_3d_x": ssim_3d_axis[0],
        "ssim_3d_y": ssim_3d_axis[1],
        "ssim_3d_z": ssim_3d_axis[2],
        "duration (sec)": duration,
        "duration (min)": duration / 60,
    }
    with open(osp.join(save_path, "eval_3d.yml"), "w") as f:
        yaml.dump(report_dict, f, default_flow_style=False, sort_keys=False)

    print("[{}] psnr_3d: {}, ssim_3d: {}".format(method, psnr_3d, ssim_3d))
    return report_dict, ct_pred, ct_gt
