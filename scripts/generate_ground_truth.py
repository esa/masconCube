from mascon_cube.data.mesh import mesh_to_gt

mask_scalar = 2850 / 1750


def mask_generator(mascon_points):
    return mascon_points[:, 0] - 0.5 * mascon_points[:, 2] > 0.4


mesh_to_gt("itokawa_lp", mask_generator, mask_scalar, save_image=True)
