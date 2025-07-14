# H5 dataset conversion

This tool converts CT projection data stored in HDF5 files to the NeRF style dataset
format used in this repository.

The expected HDF5 structure contains the projection array at `/entry/data/data`,
the projection angles at `/entry/instrument/NDAttributes/SAMPLE_MICOS_W2`, the
camera--sample distance at `/entry/instrument/NDAttributes/CT_Camera_Z`, and the
pixel size at `/entry/instrument/NDAttributes/CT_Pixelsize`.

```
└── <case>.h5
    ├── entry
    │   ├── data
    │   │   └── data  # (N, H, W)
    │   └── instrument
    │       └── NDAttributes
    │           ├── SAMPLE_MICOS_W2  # (N,)
    │           ├── CT_Camera_Z     # (N,)
    │           └── CT_Pixelsize    # (N,)
```

Run the converter by specifying the HDF5 file and output directory.
Scanner geometry can be provided via a YAML file or individual command line
arguments.

```sh
python data_generator/h5_dataset/convert_h5.py \
  --h5_file <raw.h5> \
  --output data/h5_case \
  --scanner data_generator/synthetic_dataset/scanner/parallel_beam.yml \
  --n_train 50 \
  --n_test 100 \
  --numFF 20 \
  --darkfield 100 \
  --save_mat \
  --shift_px 5
```

The first `numFF` frames are averaged as the flat field. Each projection is
normalized as `-log((proj - darkfield) / (flat - darkfield))` before being saved.

If `--save_mat` is provided, the processed projections are also written to a
`mat` subfolder as `0001.mat`, `0002.mat`, ... with variable name `img`.  The
`--shift_px` option shifts the projections horizontally prior to saving, which
can be used for simple center-of-rotation correction.

If `--scanner` is omitted, geometry parameters are inferred from the HDF5 file
(`CT_Camera_Z` and `CT_Pixelsize`) and the following arguments must be supplied:
`--nVoxel`, `--sVoxel`, `--offOrigin`, `--offDetector`, and `--accuracy`.

The script saves the projections to `proj_train` and `proj_test` directories as
`*.npy` files and generates `meta_data.json` describing the geometry and
projection parameters.
