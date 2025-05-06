# Data

Data is not stored in this repository to keep the size of the repository small.
You can download the 3d meshes from [darioizzo/geodesyNets/3dmeshes](https://github.com/darioizzo/geodesyNets/tree/1edbb64d1e8e355e124a41eac27a14d7c5c5d881/3dmeshes). They should be copied inside the `3dmeshes` folder.
3D meshes with `_raw` in the name are the original meshes, and they must be processed with the `mascon_cube.data.mesh.convert_mesh` function.

To generate the ground-truth mascon models, you can use the `mascon_cube.data.mesh.mesh_to_gt` function.
You can run `scripts/generate_ground_truth.py` to generate all the ground-truth mascon models used in the paper; remember to copy the 3D meshes in the `3dmeshes` folder before running the script.

The structure of the generated ground-truth mascon models is as follows:

```
groun_truths/
├── gt_name_1/
│   ├── mascon_model_uniform.pk     # uniform mascon model
│   ├── mascon_model.pk             # mascon model
│   ├── mesh.pk                     # 3D mesh
│   ├── plot_xy.png                 # 2D plot of the xy plane   
│   ├── plot_xz.png                 # 2D plot of the xz plane    
│   ├── plot_yz.png                 # 2D plot of the yz plane
│   └── plot.png                    # 3D plot of the mascon model
...
```