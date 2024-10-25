#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
import torch
import numpy as np

sys.path.append("./")
from r2_gaussian.dataset.cameras import Camera


def loadCam(args, id, cam_info):
    gt_image = torch.from_numpy(cam_info.image)[None]

    return Camera(
        colmap_id=cam_info.uid,
        scanner_cfg=cam_info.scanner_cfg,
        R=cam_info.R,
        T=cam_info.T,
        angle=cam_info.angle,
        mode=cam_info.mode,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
    )


def cameraList_from_camInfos(cam_infos, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.eye(4)
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T

    W2C = Rt
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "mode": camera.mode,
        "position_w2c": pos.tolist(),
        "rotation_w2c": serializable_array_2d,
        "FovY": camera.FovY,
        "FovX": camera.FovX,
    }
    return camera_entry
