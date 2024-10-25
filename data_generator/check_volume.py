import numpy as np
import pyvista as pv

vol_path = "data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/vol_gt.npy"


vol = np.load(vol_path)

plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
plotter.add_volume(vol, cmap="viridis", opacity="linear")
plotter.show()
