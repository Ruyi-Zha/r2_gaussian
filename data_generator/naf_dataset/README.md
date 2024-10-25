# NAF format dataset

You can download NAF format dataset (`*.pickle` or `*.pkl`) from [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF).

## Initialize Gaussians
You need to first generate point clouds for Gaussian initialization. Assume `*.pickle` files are in `data/sax-nerf`.

```sh
└── data   
│   └── sax-nerf
│   │   ├── abdomen_50.pickle
│   │   └── ...
```

Run `data_generator/naf_dataset/initialize_pcd_all.py` to generate initialization files. You can also generate one case with `initialize_pcd.py`.

```sh
# Generate initialization files for all cases in a folder.
python data_generator/naf_dataset/initialize_pcd_all.py --data data/sax-nerf

# Generate initialization file for one case.
python initialize_pcd.py --data *.pickle
```
