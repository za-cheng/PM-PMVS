import numpy as np
import trimesh

from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
TOL=1e-3

def save_hist(x, bins, fname):
    plt.figure()
    plt.hist(x, bins=bins)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

I = 0

def render_mesh_from_views(mesh, brdf, o, view_angles, c=1, return_normals=False, return_locations=False, brdf_is_log=False, smooth=True, no_bins=None, case='1d', return_observation_map=False, render_cast_shadow=True):
    #defined in world frame, i.e. coords system of mesh object
    #brdf is defined with respect to cos value of half angle, input [n] returns [n,3]
    #view_angles [height, width, 3] pointing outwards from camera centre
    global I
    ray_tracer = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, True)
    face_normals = mesh.face_normals
    im_shape = view_angles.shape
    view_angles = np.reshape(view_angles, (-1, 3)) # [no_pixels, 3]
    view_angles = view_angles / np.linalg.norm(view_angles, axis=-1, keepdims=True)
    
    o, l = o

    os = o * np.ones_like(view_angles)
    
    lit_map = np.zeros(len(mesh.faces))
    view_map = np.zeros(len(mesh.faces))


    face_id, ray_id, loc = ray_tracer.intersects_id(os, view_angles, multiple_hits=False, return_locations=True)
    view_angles = -view_angles
    face_indices = -np.ones(shape=[len(view_angles)], dtype=np.int32)
    face_indices[ray_id] = face_id
    
    for i, intsct_loc in zip(face_id, loc):
        view_map[i] = view_map[i] + 1
        if ((l - intsct_loc) * mesh.face_normals[i]).sum(-1) > 0:
            lit_map[i] = lit_map[i] + 1
    
    

    intersect_distance = -np.ones(shape=[len(view_angles)], dtype=np.float32) # default -1 if no intersection
    intersect_l_distance = -np.ones(shape=[len(view_angles)], dtype=np.float32) # default -1 if no intersection
    intersect_normals = np.zeros_like(view_angles) + np.nan # defult nan vectors if no intersection
    intersect_locations = np.zeros_like(view_angles) + np.nan # defult nans if no intersection
    
    intersect_distance[ray_id] = np.linalg.norm(loc - o, axis=-1)
    intersect_l_distance[ray_id] = np.linalg.norm(loc - l, axis=-1)
    intersect_locations[ray_id] = loc
    light_angles = normalise(l - intersect_locations)


    if smooth:
        vertices_ids = mesh.faces[face_id]
        vertices_pos = mesh.vertices[vertices_ids.flatten()].reshape(vertices_ids.shape+(3,))
        vertices_pos[...,-1] = 1
        loc_ = loc.copy()
        loc_[...,-1] = 1
        
        barycentric_w = np.array([l_.dot(p) for p, l_ in zip(np.linalg.inv(vertices_pos), loc_)])
        barycentric_w = barycentric_w / np.sum(barycentric_w, axis=-1, keepdims=True)
        vertices_normals = mesh.vertex_normals[vertices_ids.flatten()].reshape(vertices_ids.shape+(3,))
        smoothed_normals = np.sum(barycentric_w[...,None] * vertices_normals, axis=1)
        intersect_normals[ray_id] = smoothed_normals / np.linalg.norm(smoothed_normals, axis=-1, keepdims=True)
    else:
        intersect_normals[ray_id] = face_normals[face_id]
    
    im_normals = intersect_normals
    intersect_distance = intersect_distance[...,None]
    intersect_l_distance = intersect_l_distance[...,None]

 
    half_angles = normalise(light_angles + view_angles)
    cos_half_angles = (half_angles * intersect_normals).sum(-1)
    cos_diff_angles = (half_angles * light_angles).sum(-1)

    # save_hist(np.arccos(cos_diff_angles[np.isnan(cos_diff_angles) == False])*180/np.pi, 100, '{:02}-theta_d.png'.format(I))
    I = I+1


    ###################
    
    if brdf_is_log:
        if case == '1d':
            im_albedos = np.exp(brdf(np.sum(im_normals*view_angles, axis=-1))) * c / (intersect_distance**2) # [n,3]
        else:
            im_albedos = np.zeros_like(intersect_normals) + np.nan
            mask = np.isnan(cos_half_angles) == False
            im_albedos[mask] = brdf(cos_half_angles[mask], cos_diff_angles[mask])
            im_albedos = np.exp(im_albedos) * c / (intersect_l_distance**2) # [n,3]
            im_albedos[((light_angles*im_normals).sum(-1) <= 0)] = 0
            im_albedos = im_albedos * (light_angles*im_normals).sum(-1)[...,None]
        
    else:
        if case == '1d':
            im_albedos = brdf(np.sum(im_normals*view_angles, axis=-1)) * c / (intersect_distance**2) # [n,3]
        else:
            im_albedos = np.zeros_like(intersect_normals) + np.nan
            mask = np.isnan(cos_half_angles) == False
            im_albedos[mask] = brdf(cos_half_angles[mask], cos_diff_angles[mask])
            im_albedos = im_albedos * c / (intersect_l_distance**2) # [n,3]
            im_albedos[((light_angles*im_normals).sum(-1) <= 0)] = 0
            im_albedos = im_albedos * (light_angles*im_normals).sum(-1)[...,None]

    if render_cast_shadow:
        valid_mask = np.isnan(im_albedos.sum(-1)) == False
        valid_albedo = im_albedos[valid_mask]
        valid_loc = intersect_locations[valid_mask]
        lit_mask = np.zeros(valid_albedo.shape[:1], dtype=bool)

        valid_rays = -(l - valid_loc) * 0.1
        new_face_id, new_ray_id, new_loc = ray_tracer.intersects_id(l * np.ones_like(valid_loc), valid_rays, multiple_hits=False, return_locations=True)

        mask_ = ((valid_loc[new_ray_id] - new_loc) ** 2).sum(-1) <= TOL ** 2

        lit_map[:] = 0

        for i in new_face_id[mask_]:
            lit_map[i] = lit_map[i] + 1

        lit_mask[new_ray_id[mask_]] = True
        valid_albedo[lit_mask == False] = 0
        im_albedos[valid_mask] = valid_albedo

    # im_albedos = np.log(np.maximum(im_albedos, 1e-15))
    # im_albedos = (np.floor(np.clip(im_albedos + 15, 0, 30)) / 30 * 65535) * 30.0 / 65535.0 - 15
    # im_albedos = np.exp(im_albedos)


    if no_bins:
        im_albedos = dicretize_array(im_albedos, no_bins)
    
    ret = [im_albedos.reshape(im_shape)]
    
    if return_normals: ret.append(im_normals.reshape(im_shape))
    if return_locations: ret.append(intersect_locations.reshape(im_shape))

    if return_observation_map:
        ret = tuple(list(ret) + list((view_map, lit_map)))

    ret = ret[0] if len(ret) == 1 else tuple(ret)
    return ret
    
def create_view_angles_from_vcam(P, V, height, width):
    w, h = np.meshgrid(range(width), range(height))
    img_coords = np.stack([h,w,np.ones_like(w)], axis=-1).reshape((-1,3))
    camera_coords = img_coords.dot(np.linalg.inv(P[:3,:3]).T) # [height*width, 3]
    ray_camera_coords_xyz = (-camera_coords / camera_coords[:,2:]) # [height*width, 3]
    ray_world_coords_xyz = ray_camera_coords_xyz.dot(V[:3,:3]) # [height*width, 3]
    o = np.linalg.inv(V).dot([0,0,0,1])[:3] # view origin/camera centre
    ds = ray_world_coords_xyz.reshape([height, width, 3]) # view angles for each pixel
    return o, ds


def render_mesh(mesh, P, V, img_size, brdf_func, c=1, return_normals=False, return_locations=False, brdf_is_log=False, smooth=True, no_bins=None, l=None, return_observation_map=False):
    height, width = img_size
    o, ds = create_view_angles_from_vcam(P, V, height, width)
    if l is None:
        o = (o,o)
        case = '1d'
    else:
        o = (o,l+o)
        case = '2d'
    return render_mesh_from_views(mesh, brdf_func, o, ds, c, return_normals, return_locations, brdf_is_log, smooth=smooth, no_bins=no_bins, case=case, return_observation_map=return_observation_map)

