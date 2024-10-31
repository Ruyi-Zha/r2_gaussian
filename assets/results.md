# Experiment results

## [Synthetic dataset](https://drive.google.com/drive/folders/1K3-8RECm-MwUpphIXEeqWznHpTdv4sHz?usp=sharing)

We use the default settings for R2-Gaussian and SAX-NeRF.

<div style="overflow-x:auto;">

| (PSNR3D) | SAX-NeRF (75-view) | R2-Gaussian (75-view) | SAX-NeRF (50-view) | R2-Gaussian (50-view) | SAX-NeRF (25-view) | R2-Gaussian (25-view) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Chest | 35.92 | 35.85 | 34.45 | 35.09 | 31.54 | 31.93 |
| Foot | 32.80 | 32.51 | 31.90 | 31.83 | 30.27 | 30.25 |
| Head | 41.01 | 40.95 | 39.51 | 40.12 | 36.72 | 36.36 |
| Jaw | 36.81 | 37.31 | 35.37 | 36.45 | 31.68 | 33.32 |
| Pancreas | 37.52 | 38.05 | 35.71 | 36.72 | 32.33 | 33.28 |
| Beetle | 42.21 | 43.33 | 41.39 | 42.46 | 38.81 | 40.20 |
| Bonsai | 36.58 | 35.56 | 35.13 | 34.86 | 33.12 | 32.90 |
| Broccoli | 35.89 | 36.52 | 34.11 | 34.47 | 29.65 | 29.05 |
| Kingsnake | 40.20 | 39.92 | 39.56 | 39.71 | 38.58 | 38.90 |
| Pepper | 36.87 | 39.44 | 35.54 | 38.30 | 33.10 | 34.33 |
| Backpack | 36.65 | 38.92 | 35.28 | 38.16 | 32.74 | 35.43 |
| Engine | 38.23 | 40.23 | 36.90 | 39.00 | 34.69 | 34.64 |
| Mount | 38.49 | 38.69 | 37.87 | 38.17 | 36.03 | 36.59 |
| Present | 36.34 | 38.25 | 35.58 | 37.84 | 32.50 | 35.19 |
| Teapot | 45.50 | 48.12 | 44.68 | 47.57 | 43.18 | 45.96 |
| Average PSNR | 38.07 | 38.91 | 36.86 | 38.05 | 34.33 | 35.22 |
| Time | ~13h | 5-15min | ~13h | 5-15min | ~13h | 5-15min |

</div>

## [Real-world dataset (FIPS)](https://drive.google.com/drive/folders/1A1wriUWJcg8vnoMf71iDxGVriXrBq5kF?usp=sharing)

We use the default settings for R2-Gaussian and SAX-NeRF.

<div style="overflow-x:auto;">

| (PSNR3D) | SAX-NeRF (75-view) | R2-Gaussian (75-view) | SAX-NeRF (50-view) | R2-Gaussian (50-view) | SAX-NeRF (25-view) | R2-Gaussian (25-view) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pine | 37.46 | 41.09 | 36.96 | 40.41 | 35.74 | 37.35 |
| Seashell | 36.41 | 43.70 | 36.71 | 42.85 | 35.55 | 39.09 |
| Walnut | 30.91 | 33.43 | 30.99 | 31.83 | 29.19 | 28.42 |
| Average PSNR | 34.93 | 39.41 | 34.89 | 38.36 | 33.49 | 34.95 |
| Time | ~13h | 10-17min | ~13h | 10-17min | ~13h | 10-17min |

</div>


## [SAX-NeRF dataset](https://drive.google.com/drive/folders/1SlneuSGkhk0nvwPjxxnpBCO59XhjGGJX?usp=sharing)

We use the default settings for R2-Gaussian, except for setting `--densify_grad_threshold` to `3.0e-5`. SAX-NeRF is run with the default settings.

<div style="overflow-x:auto;">

| (PSNR3D) | SAX-NeRF | R2-Gaussian (default) | R2-Gaussian (densify_grad_threshold=3e-5) |
|:---:|:---:|:---:|:---:|
| Abdomen | 35.00 | 33.27 | 33.55 |
| Aneurism | 41.43 | 41.81 | 42.20 |
| Backpack | 35.97 | 39.73 | 40.29 |
| Bonsai | 36.52 | 35.22 | 35.33 |
| Box | 35.33 | 37.05 | 37.23 |
| Carp | 42.72 | 41.16 | 41.21 |
| Chest | 34.36 | 33.58 | 33.49 |
| Engine | 38.77 | 39.63 | 39.60 |
| Foot | 32.25 | 31.96 | 32.03 |
| Head | 39.70 | 39.70 | 39.95 |
| Jaw | 35.47 | 36.41 | 36.57 |
| Leg | 43.47 | 43.88 | 43.91 |
| Pancreas | 22.95 | 21.75 | 22.06 |
| Pelvis | 40.38 | 39.25 | 39.52 |
| Teapot | 44.33 | 47.16 | 47.33 |
| Average PSNR | 37.24 | 37.44 | 37.62 |
| Time | ~13h | 5-33min | 6-54min |

</div>