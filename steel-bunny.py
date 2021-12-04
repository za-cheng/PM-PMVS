import trimesh
import numpy as np
from render import render_mesh
from utils import load_brdf
import cv2

# camera intrinsic matrix
K = np.array([[ 0.0000000e+00, -1.1149023e+03, -2.0000000e+02,  0.0000000e+00],
       [ 1.1149023e+03,  0.0000000e+00, -3.2000000e+02,  0.0000000e+00],
       [ 0.0000000e+00,  0.0000000e+00, -1.0000000e+00,  0.0000000e+00]])


HEIGHT, WIDTH = 400, 640 # image dimensions

brdf_str = 'steel'

# load mesh and normalize
mesh = trimesh.load_mesh('data/bunny.obj')
scale = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max()
mesh.vertices = mesh.vertices / scale * 10

Ps = np.load('data/bunny_Poses.npy') # camera extrinsic matrices

# render images, as well as ground truth normal maps and 3D locations for each pixel
org_imgs, r_normals, r_locations = zip(*[render_mesh(mesh, K, P, [HEIGHT,WIDTH], load_brdf(brdf_str, log=True, epsilon=1e-10), c=1, brdf_is_log=True, smooth=True, return_normals=True, return_locations=True, no_bins=None, l=None) for P in Ps])

org_imgs = np.array(org_imgs)

# pack the input file, all input files need to follow this format
np.save('data/steel-bunny', {
    'imgs': org_imgs, # input images
    'K': K, # intrinsic images
    'P': Ps, # camera poses
    'height': HEIGHT,
    'width': WIDTH,
})

# visualize images and save to disk
org_imgs[~np.isfinite(org_imgs)] = 0
median = np.median(org_imgs[org_imgs > 0])
for i, img in enumerate(org_imgs):
    img = img * 0.5 / median
    img = np.clip(img, 0, 1)

    cv2.imwrite(f'results/input-{i:02}.png', (img*255).astype(np.uint8)[...,::-1].reshape([HEIGHT, WIDTH, 3]))

