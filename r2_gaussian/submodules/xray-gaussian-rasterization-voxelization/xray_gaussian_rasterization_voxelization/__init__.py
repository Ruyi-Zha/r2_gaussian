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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.mode,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = (
                    _C.rasterize_gaussians(*args)
                )
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = (
                _C.rasterize_gaussians(*args)
            )

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.mode = raster_settings.mode
        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        mode = ctx.mode
        (
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            means3D,
            radii,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            mode,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_opacities,
                    _,  # grad_mu
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_opacities,
                _,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)
        grads = (
            grad_means3D,
            grad_means2D,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


# Change according to line 45 gaussian_renderer/__init__.py
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    prefiltered: bool
    mode: int
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):

        raster_settings = self.raster_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )


class GaussianVoxelizationSettings(NamedTuple):
    scale_modifier: float
    nVoxel_x: int
    nVoxel_y: int
    nVoxel_z: int
    sVoxel_x: float
    sVoxel_y: float
    sVoxel_z: float
    center_x: float
    center_y: float
    center_z: float
    prefiltered: bool
    debug: bool


def voxelize_gaussians(
    means3D, opacities, scales, rotations, cov3Ds_precomp, voxel_settings
):
    return _VoxelizeGaussians.apply(
        means3D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        voxel_settings,
    )


class _VoxelizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        voxel_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            opacities,
            scales,
            rotations,
            voxel_settings.scale_modifier,
            cov3Ds_precomp,
            voxel_settings.nVoxel_x,
            voxel_settings.nVoxel_y,
            voxel_settings.nVoxel_z,
            voxel_settings.sVoxel_x,
            voxel_settings.sVoxel_y,
            voxel_settings.sVoxel_z,
            voxel_settings.center_x,
            voxel_settings.center_y,
            voxel_settings.center_z,
            voxel_settings.prefiltered,
            voxel_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if voxel_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                num_rendered, fields, radii, geomBuffer, binningBuffer, imgBuffer = (
                    _C.voxelize_gaussians(*args)
                )
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            num_rendered, fields, radii, geomBuffer, binningBuffer, imgBuffer = (
                _C.voxelize_gaussians(*args)
            )

        # Keep relevant tensors for backward
        ctx.voxel_settings = voxel_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )

        return fields, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        voxel_settings = ctx.voxel_settings
        (
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            means3D,
            radii,
            scales,
            rotations,
            voxel_settings.scale_modifier,
            cov3Ds_precomp,
            grad_out_color,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            voxel_settings.nVoxel_x,
            voxel_settings.nVoxel_y,
            voxel_settings.nVoxel_z,
            voxel_settings.sVoxel_x,
            voxel_settings.sVoxel_y,
            voxel_settings.sVoxel_z,
            voxel_settings.center_x,
            voxel_settings.center_y,
            voxel_settings.center_z,
            voxel_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if voxel_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_scales,
                    grad_rotations,
                ) = _C.voxelize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_scales,
                grad_rotations,
            ) = _C.voxelize_gaussians_backward(*args)
        grads = (
            grad_means3D,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianVoxelizer(nn.Module):
    def __init__(self, voxel_settings):
        super().__init__()
        self.voxel_settings = voxel_settings

    def forward(
        self,
        means3D,
        opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):

        voxel_settings = self.voxel_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return voxelize_gaussians(
            means3D,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            voxel_settings,
        )
