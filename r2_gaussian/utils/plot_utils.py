import numpy as np
import matplotlib
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from scipy.spatial.transform import Rotation
from tqdm import trange
import open3d as o3d
import sys

sys.path.append("./")
from r2_gaussian.utils.general_utils import t2a


def show_gaussians(
    gaussians,
    n_gaussian=None,
    bbox=[[-1, -1, -1], [1, 1, 1]],
    sort_gaussians="no",  # ["density", "scale", None]
    mesh_resoltuion=3,
):
    """Visualize gaussians for debugging."""
    g_density = t2a(gaussians.get_density).numpy()[:, 0]
    indices = g_density != 0
    g_density = g_density[indices]
    g_center = t2a(gaussians.get_xyz)[indices]
    g_scale = t2a(gaussians.get_scaling)[indices]
    g_quad = t2a(gaussians.get_rotation)[indices]

    if sort_gaussians != "no":
        if sort_gaussians == "density":
            indices = np.argsort(g_density)[::-1]
        elif sort_gaussians == "scale":
            mean_g_scale = g_scale.mean(axis=-1)
            indices = np.argsort(mean_g_scale)[::-1]
        else:
            raise NotImplementedError("Unknown sort gaussian mode!")
        g_density = g_density[indices]
        g_center = g_center[indices]
        g_scale = g_scale[indices]
        g_quad = g_quad[indices]

    density_scale = 0.95 / g_density.max()

    total_gaussian = g_density.shape[0]
    if n_gaussian is None:
        n_gaussian = total_gaussian
    elif n_gaussian > total_gaussian:
        n_gaussian = total_gaussian

    ellipses = []
    for i_ellipse in trange(n_gaussian, desc="visualize gaussians"):
        if i_ellipse == 8498:
            a = 1
        ellipse = create_o3d_ellipse(
            g_center[i_ellipse],
            g_scale[i_ellipse],
            g_quad[i_ellipse],
            g_density[i_ellipse] * density_scale,
            resolution=mesh_resoltuion,
        )
        ellipses.append(ellipse)

    bbox_o3d = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[0], max_bound=bbox[1])
    )
    bbox_o3d_color = np.zeros((np.asarray(bbox_o3d.lines).shape[0], 3))
    bbox_o3d_color[:, 0] = 1
    bbox_o3d.colors = o3d.utility.Vector3dVector(bbox_o3d_color)

    show_set = ellipses + [bbox_o3d]

    o3d.visualization.draw_geometries(show_set)
    # o3d.visualization.webrtc_server.enable_webrtc()
    # o3d.visualization.draw(show_set)


def create_o3d_ellipse(center, scale, quad, density, resolution=5):
    axes_lengths = 1 * scale
    r = Rotation.from_quat(quad)
    rotation_matrix = r.as_matrix()

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)

    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = center
    transform = np.matmul(T, np.diag(np.append(axes_lengths, [1])))

    vertex_colors = np.ones_like(np.asarray(mesh.vertices)) * density
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    mesh = mesh.transform(transform)

    return mesh


def show_one_slice(
    slice,
    title,
    cmap="viridis",
    vmax=None,
    vmin=None,
    save=False,
):
    if save:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    if torch.is_tensor(slice):
        slice = t2a(slice)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    img = ax.imshow(
        slice,
        cmap=cmap,
        vmin=slice.min() if vmin is None else vmin,
        vmax=slice.max() if vmax is None else vmax,
    )
    ax.title.set_text(title)

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(img, cax=cax)

    plt.tight_layout()

    if save:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = np.array(data.reshape(fig.canvas.get_width_height()[::-1] + (4,)))[
            ..., :3
        ]
        plt.close()
        return data
    else:
        plt.show()
        plt.close()
        return


def show_three_volume(
    volume1,
    volume2,
    volume3,
    cmap="viridis",
    title1="1",
    title2="2",
    title3="3",
    vmax=None,
):
    diff2 = np.abs(volume1 - volume2)
    diff3 = np.abs(volume1 - volume3)

    # Create a figure and axis
    fig, ax = plt.subplots(2, 3, figsize=(20, 8))
    plt.subplots_adjust(bottom=0.25)

    # Initial slice index
    initial_slice = int(volume1.shape[2] / 2)

    error_max = np.max([diff2.max(), diff3.max()])
    # Display the initial slices
    img1 = ax[0, 0].imshow(
        volume1[:, :, initial_slice],
        cmap=cmap,
        vmin=0.0,
        vmax=volume1.max() if vmax is None else vmax,
    )
    ax[0, 0].title.set_text(title1)
    img2 = ax[0, 1].imshow(
        volume2[:, :, initial_slice],
        cmap=cmap,
        vmin=0.0,
        vmax=volume2.max() if vmax is None else vmax,
    )
    ax[0, 1].title.set_text(title2)
    img3 = ax[0, 2].imshow(
        diff2[:, :, initial_slice], cmap=cmap, vmin=0.0, vmax=error_max
    )
    ax[0, 2].title.set_text("error")
    img4 = ax[1, 0].imshow(
        volume1[:, :, initial_slice],
        cmap=cmap,
        vmin=0.0,
        vmax=volume1.max() if vmax is None else vmax,
    )
    ax[1, 0].title.set_text(title1)
    img5 = ax[1, 1].imshow(
        volume3[:, :, initial_slice],
        cmap=cmap,
        vmin=0.0,
        vmax=volume3.max() if vmax is None else vmax,
    )
    ax[1, 1].title.set_text(title3)
    img6 = ax[1, 2].imshow(
        diff3[:, :, initial_slice], cmap=cmap, vmin=0.0, vmax=error_max
    )
    ax[1, 2].title.set_text("error")

    # Add colorbars
    cax1 = make_axes_locatable(ax[0, 0]).append_axes("right", size="5%", pad=0.1)
    cax2 = make_axes_locatable(ax[0, 1]).append_axes("right", size="5%", pad=0.1)
    cax3 = make_axes_locatable(ax[0, 2]).append_axes("right", size="5%", pad=0.1)
    cax4 = make_axes_locatable(ax[1, 0]).append_axes("right", size="5%", pad=0.1)
    cax5 = make_axes_locatable(ax[1, 1]).append_axes("right", size="5%", pad=0.1)
    cax6 = make_axes_locatable(ax[1, 2]).append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(img1, cax=cax1)
    cbar2 = fig.colorbar(img2, cax=cax2)
    cbar3 = fig.colorbar(img3, cax=cax3)
    cbar4 = fig.colorbar(img4, cax=cax4)
    cbar5 = fig.colorbar(img5, cax=cax5)
    cbar6 = fig.colorbar(img6, cax=cax6)

    plt.tight_layout()
    # Add a single slider for both volumes
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slice_slider = Slider(
        ax_slider,
        "Slice",
        0,
        max(volume1.shape[2], volume2.shape[2], volume3.shape[2]) - 1,
        valinit=initial_slice,
        valstep=1,
    )

    # Update function for the slider
    def update(val):
        slice_index = int(slice_slider.val)
        img1.set_array(volume1[:, :, slice_index])
        img2.set_array(volume2[:, :, slice_index])
        img3.set_array(diff2[:, :, slice_index])
        img4.set_array(volume1[:, :, slice_index])
        img5.set_array(volume3[:, :, slice_index])
        img6.set_array(diff3[:, :, slice_index])
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slice_slider.on_changed(update)

    # Show the plot
    plt.show()


def show_two_slice(
    slice1,
    slice2,
    title1,
    title2,
    cmap="viridis",
    vmax=None,
    vmin=None,
    gamma=1.0,
    save=False,
    no_diff=False,
):
    if save:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    if torch.is_tensor(slice1):
        slice1 = t2a(slice1)
    if torch.is_tensor(slice2):
        slice2 = t2a(slice2)
    if not no_diff:
        diff = np.abs(slice1 - slice2) ** gamma
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    img1 = ax[0].imshow(
        slice1,
        cmap=cmap,
        vmin=slice1.min() if vmin is None else vmin,
        vmax=slice1.max() if vmax is None else vmax,
    )
    ax[0].title.set_text(title1)
    cax0 = make_axes_locatable(ax[0]).append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(img1, cax=cax0)

    img2 = ax[1].imshow(
        slice2,
        cmap=cmap,
        vmin=slice2.min() if vmin is None else vmin,
        vmax=slice2.max() if vmax is None else vmax,
    )
    ax[1].title.set_text(title2)
    cax1 = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(img2, cax=cax1)

    if not no_diff:
        img3 = ax[2].imshow(diff, cmap=cmap, vmin=0.0)
        ax[2].title.set_text("error")
        cax2 = make_axes_locatable(ax[2]).append_axes("right", size="5%", pad=0.1)
        cbar3 = fig.colorbar(img3, cax=cax2)

    plt.tight_layout()

    if save:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = np.array(data.reshape(fig.canvas.get_width_height()[::-1] + (4,)))[
            ..., :3
        ]
        plt.close()
        return data
    else:
        plt.show()
        plt.close()
        return


def show_one_volume(volume1, cmap="viridis"):
    if torch.is_tensor(volume1):
        volume1 = t2a(volume1)
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=0.25)

    # Initial slice index
    initial_slice = int(volume1.shape[2] / 2)

    # Display the initial slices
    img1 = ax.imshow(
        volume1[:, :, initial_slice], cmap=cmap, vmin=0.0, vmax=volume1.max()
    )

    # Add colorbars
    cbar1 = fig.colorbar(img1, ax=ax, orientation="vertical")

    # Add a single slider for both volumes
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slice_slider = Slider(
        ax_slider,
        "Slice",
        0,
        volume1.shape[2] - 1,
        valinit=initial_slice,
        valstep=1,
    )

    # Update function for the slider
    def update(val):
        slice_index = int(slice_slider.val)
        img1.set_array(volume1[:, :, slice_index])
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slice_slider.on_changed(update)

    # Show the plot
    plt.show()


def show_two_volume(
    volume1,
    volume2,
    cmap="viridis",
    gamma=1.0,
    vmax=None,
    title1=None,
    title2=None,
    no_diff=False,
    axis=2,
):
    """Show two volumes. gamma is used to visualize error."""
    if torch.is_tensor(volume1):
        volume1 = t2a(volume1)
    if torch.is_tensor(volume2):
        volume2 = t2a(volume2)
    if axis == 0:
        volume1 = volume1.transpose([1, 2, 0])
        volume2 = volume2.transpose([1, 2, 0])
    elif axis == 1:
        volume1 = volume1.transpose([0, 2, 1])
        volume2 = volume2.transpose([0, 2, 1])

    if not no_diff:
        diff = np.abs(volume1 - volume2) ** gamma

    # Create a figure and axis
    if no_diff:
        fig, ax = plt.subplots(1, 2)
    else:
        fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(bottom=0.25)

    # Initial slice index
    initial_slice = int(volume1.shape[2] / 2)

    # Display the initial slices
    img1 = ax[0].imshow(
        volume1[:, :, initial_slice],
        cmap=cmap,
        vmin=0.0,
        vmax=volume1.max() if vmax is None else vmax,
    )
    if title1 is not None:
        ax[0].title.set_text(title1)
    img2 = ax[1].imshow(
        volume2[:, :, initial_slice],
        cmap=cmap,
        vmin=0.0,
        vmax=volume2.max() if vmax is None else vmax,
    )
    if title2 is not None:
        ax[1].title.set_text(title2)
    if not no_diff:
        img3 = ax[2].imshow(
            diff[:, :, initial_slice], cmap=cmap, vmin=0.0, vmax=diff.max()
        )
        ax[2].title.set_text("error")

    # Add colorbars
    cbar1 = fig.colorbar(img1, ax=ax[0], orientation="horizontal")
    cbar2 = fig.colorbar(img2, ax=ax[1], orientation="horizontal")
    if not no_diff:
        cbar3 = fig.colorbar(img3, ax=ax[2], orientation="horizontal")

    # Add a single slider for both volumes
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slice_slider = Slider(
        ax_slider,
        "Slice",
        0,
        max(volume1.shape[2], volume2.shape[2]) - 1,
        valinit=initial_slice,
        valstep=1,
    )

    # Update function for the slider
    def update(val):
        slice_index = int(slice_slider.val)
        img1.set_array(volume1[:, :, slice_index])
        img2.set_array(volume2[:, :, slice_index])
        if not no_diff:
            img3.set_array(diff[:, :, slice_index])
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slice_slider.on_changed(update)

    # Show the plot
    plt.show()
