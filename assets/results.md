# Experiment results

## [Synthetic dataset](https://drive.google.com/drive/folders/1K3-8RECm-MwUpphIXEeqWznHpTdv4sHz?usp=sharing)

We use the default settings for R2-Gaussian and SAX-NeRF.

<div style="overflow-x:auto;">

| (PSNR3D) | SAX-NeRF (75-view) | R2-Gaussian (75-view) | SAX-NeRF (50-view) | R2-Gaussian (50-view) | SAX-NeRF (25-view) | R2-Gaussian (25-view) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Chest | 35.92 | 35.88 | 34.45 | 35.10 | 31.54 | 31.89 |
| Foot | 32.80 | 32.50 | 31.90 | 31.84 | 30.27 | 30.25 |
| Head | 41.01 | 40.95 | 39.51 | 40.13 | 36.72 | 36.35 |
| Jaw | 36.81 | 37.31 | 35.37 | 36.45 | 31.68 | 33.31 |
| Pancreas | 37.52 | 38.10 | 35.71 | 36.56 | 32.33 | 33.24 |
| Beetle | 42.21 | 43.33 | 41.39 | 42.46 | 38.81 | 40.22 |
| Bonsai | 36.58 | 35.53 | 35.13 | 34.84 | 33.12 | 32.90 |
| Broccoli | 35.89 | 36.52 | 34.11 | 34.46 | 29.65 | 29.07 |
| Kingsnake | 40.20 | 39.91 | 39.56 | 39.71 | 38.58 | 38.90 |
| Pepper | 36.87 | 39.45 | 35.54 | 38.30 | 33.10 | 34.35 |
| Backpack | 36.65 | 38.90 | 35.28 | 47.36 | 32.74 | 35.39 |
| Engine | 38.23 | 40.24 | 36.90 | 39.00 | 34.69 | 34.65 |
| Mount | 38.49 | 38.66 | 37.87 | 38.14 | 36.03 | 36.56 |
| Present | 36.34 | 38.25 | 35.58 | 37.86 | 32.50 | 35.18 |
| Teapot | 45.50 | 48.09 | 44.68 | 47.59 | 43.18 | 46.02 |
| Average | 38.07 | 38.91 | 36.86 | 38.65 | 34.33 | 35.22 |
| Time | ~13h | 5-15min | ~13h | 5-15min | ~13h | 5-15min |

</div>

## [Real-world dataset (FIPS)](https://drive.google.com/drive/folders/1A1wriUWJcg8vnoMf71iDxGVriXrBq5kF?usp=sharing)

We use the default settings for R2-Gaussian and SAX-NeRF.

<div style="overflow-x:auto;">

| (PSNR3D) | SAX-NeRF (75-view) | R2-Gaussian (75-view) | SAX-NeRF (50-view) | R2-Gaussian (50-view) | SAX-NeRF (25-view) | R2-Gaussian (25-view) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Pine | 37.46 | 40.95 | 36.96 | 40.38 | 35.74 | 37.41 |
| Seashell | 36.41 | 43.74 | 36.71 | 42.85 | 35.55 | 39.10 |
| Walnut | 30.91 | 33.42 | 30.99 | 31.83 | 29.19 | 28.44 |
| Average | 34.93 | 39.37 | 34.89 | 38.35 | 33.49 | 34.98 |
| Time | ~13h | 9-16min | ~13h | 9-16min | ~13h | 9-16min |


</div>


## [SAX-NeRF dataset](https://drive.google.com/drive/folders/1SlneuSGkhk0nvwPjxxnpBCO59XhjGGJX?usp=sharing)

We use the default settings for R2-Gaussian, except for setting `--densify_grad_threshold` to `3.0e-5`. SAX-NeRF is run with the default settings.

| (PSNR3D) | SAX-NeRF | R2-Gaussian (default) | R2-Gaussian (densify_grad_threshold=3e-5) |
|---|:---:|:---:|:---:|
| Abdomen | 35.00 | 33.28 | 33.56 |
| Aneurism | 41.43 | 41.78 | 42.22 |
| Backpack | 35.97 | 39.74 | 40.33 |
| Bonsai | 36.52 | 35.21 | 35.34 |
| Box | 35.33 | 37.06 | 37.23 |
| Carp | 42.72 | 41.18 | 41.21 |
| Chest | 34.36 | 33.56 | 33.52 |
| Engine | 38.77 | 39.55 | 39.57 |
| Foot | 32.25 | 31.96 | 32.02 |
| Head | 39.70 | 39.62 | 39.98 |
| Jaw | 35.47 | 36.41 | 36.57 |
| Leg | 43.47 | 43.86 | 43.96 |
| Pancreas | 22.95 | 21.74 | 22.06 |
| Pelvis | 40.38 | 39.25 | 39.52 |
| Teapot | 44.33 | 47.17 | 47.32 |
| Average | 37.24 | 37.42 | 37.63 |
| Time | ~13h | 5-35min | 6-53min |