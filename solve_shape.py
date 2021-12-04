import numpy as np
from utils import *
import torch
import scipy as sp
from scipy.sparse.linalg import lsqr
from params import *

def create_view(v):
    '''returns a V matrix facing origin point'''
    v = np.array(v)
    v_unit = v / np.linalg.norm(v)
    half_vector = (v_unit.reshape([3]) + [0,0,1]  + np.random.normal(0, 0.01, 3)).reshape([3,1])
    half_vector = half_vector / np.linalg.norm(half_vector)
    
    R = 2 * (half_vector.dot(half_vector.T)) - np.eye(3)
    T = -R.dot(v).reshape([3,1])
    return np.concatenate([np.concatenate([R,T],axis=-1), [[0,0,0,1]]], axis=0)

def generate_spiral_grid_vectors(alpha, revs, minimum_diff_degrees=0.5):
    i = 0.0
    vectors = []
    z = 1
    while z > 0:
        a = np.power(i, 1.1) * alpha
        b = np.power(a/revs, 5) - 0.5 * np.pi
        x = np.cos(a) * np.cos(b)
        y = np.sin(a) * np.cos(b)
        z = -np.sin(b)
        vectors.append([x,y,z])
        i = i + 1
    vectors = np.array(vectors)
    if minimum_diff_degrees is not None and minimum_diff_degrees > 0:
        vectors_refined = []
        for v in vectors:
            keep_it = True
            for vi in vectors_refined:
                if angular_error_in_degrees(vi, v, True) < minimum_diff_degrees:
                    keep_it = False
                    break
            if keep_it:
                vectors_refined.append(v)
        vectors = vectors_refined
    return np.array(vectors)


def render_point(xyzs, normals, optical_centres, log_brdf_func, epsilon):
    '''
    CPU only
    xyzs: [no_points, 3]
    normals: [no_points, 3], unit vectors
    optical_centres: [no_views, 3]
    returns: [no_points, no_views, 3]
    '''
    no_points, no_views = xyzs.shape[0], optical_centres.shape[0]

    visual_rays = optical_centres.reshape([1, no_views, 3]) - xyzs.reshape([no_points, 1, 3]) # [no_points, no_views, 3]
    theta_d = (normalise(visual_rays, epsilon=epsilon) * normals.reshape([no_points, 1, 3])).sum(-1) # [no_points, no_views]
    theta_d[theta_d < np.cos(89.0 * np.pi / 180)] = np.cos(89.0 * np.pi / 180)
    log_intensities = log_brdf_func(theta_d.reshape([-1])).reshape([no_points, no_views, 3]) \
                     - np.log(np.maximum(np.sum(visual_rays**2, -1).reshape([no_points, no_views, 1]), epsilon))
    return log_intensities

def render_point_gpu(xyzs, normals, optical_centres, log_brdf_func_gpu, epsilon):
    '''
    GPU only
    xyzs: [no_points, 3]
    normals: [no_points, 3], unit vectors
    optical_centres: [no_views, 3], or [no_views, 3, 2] last dim [camera, light]
    returns: [no_points, no_views, 3]
    '''
    no_points, no_views = xyzs.shape[0], optical_centres.shape[0]

    if len(optical_centres.shape) == 2:
        # univariate brdf
        visual_rays = optical_centres.reshape([1, no_views, 3]) - xyzs.reshape([no_points, 1, 3]) # [no_points, no_views, 3]
        theta_d = (torch.nn.functional.normalize(visual_rays, dim=-1, p=2) * normals.reshape([no_points, 1, 3])).sum(-1) # [no_points, no_views]
        zero_mask = theta_d < np.cos(89.0 * np.pi / 180)
        theta_d[theta_d < np.cos(89.0 * np.pi / 180)] = np.cos(89.0 * np.pi / 180)

        log_intensities = log_brdf_func_gpu(theta_d.reshape([-1])).reshape([no_points, no_views, 3]) \
                        - torch.log(torch.max(torch.sum(visual_rays**2, -1).reshape([no_points, no_views,1]), immutable_tensor([epsilon], dtype=visual_rays.dtype).cuda())) \
                        + 0 #torch.log(theta_d)[...,None]
        log_intensities[zero_mask] = np.inf

    elif len(optical_centres.shape) == 3 and optical_centres.shape[2]==2:
        # 2d brdf
        os = optical_centres[...,0]
        ls = optical_centres[...,1]

        visual_rays_o = os.reshape([1, no_views, 3]) - xyzs.reshape([no_points, 1, 3]) # [no_points, no_views, 3]
        visual_rays_l = ls.reshape([1, no_views, 3]) - xyzs.reshape([no_points, 1, 3]) # [no_points, no_views, 3]
        rays_h = torch.nn.functional.normalize(visual_rays_o, dim=-1, p=2) + torch.nn.functional.normalize(visual_rays_l, dim=-1, p=2) # [no_points, no_views, 3]
        rays_h = torch.nn.functional.normalize(rays_h, dim=-1, p=2) # [no_points, no_views, 3]
        cos_theta_h = (rays_h * normals.reshape([no_points, 1, 3])).sum(-1) # [no_points, no_views]
        
        cos_theta_d = (torch.nn.functional.normalize(visual_rays_o, dim=-1, p=2) * rays_h).sum(-1)


        log_intensities = log_brdf_func_gpu(cos_theta_h.reshape([-1]), cos_theta_d.reshape([-1])).reshape([no_points, no_views, 3]) \
                        - torch.log(torch.max(torch.sum(visual_rays_l**2, -1).reshape([no_points, no_views,1]), immutable_tensor([epsilon], dtype=visual_rays_l.dtype).cuda())) + \
                            torch.log(torch.clamp((torch.nn.functional.normalize(visual_rays_l, dim=-1, p=2) * normals.reshape([no_points, 1, 3])).sum(-1), 1e-6, 1))[...,None]
        log_intensities[(visual_rays_l * normals.reshape([no_points, 1, 3])).sum(-1) < 0] = np.log(epsilon)
    else:
        raise ValueError 
    return log_intensities

def query_intensity_profile_LHS(xyzs, img_funcs_batch_gpu, Ps, epsilon):
    '''
    CPU interface, GPU backend
    x [no_points, 3] world coords
    img_funcs [no_views] image functions
    Ps [no_views, 3, 4] projection matrices
    returns [no_points, no_views, 3] intensity values
    '''
    xyzs = immutable_tensor(xyzs).cuda()
    Ps_gpu = immutable_tensor(Ps).cuda()
    ret = query_intensity_profile_LHS_batch_gpu(xyzs, img_funcs_batch_gpu, Ps_gpu, epsilon=epsilon)
    ret = ret.cpu().numpy()
    return ret

def query_intensity_profile_LHS_batch_gpu(xyzs, img_funcs_batch_gpu, Ps_gpu, epsilon=None):
    '''xyzs [no_points, 3] world coords
       img_funcs [no_views] image functions
       Ps [no_views, 3, 4] projection matrices
       returns [no_points, no_views, 3] intensity values
       '''    
    xyzs = torch.cat([xyzs, torch.ones_like(xyzs[...,-1:])], dim=-1) # [no_points, 4]
    img_coords = torch.matmul(Ps_gpu, xyzs.t()).transpose(1,2) # [no_views, no_points, 3]
    img_coords = img_coords[...,:-1] / img_coords[...,-1:]
    intensities = img_funcs_batch_gpu(img_coords) # [no_views, no_points, 2]
    intensities[torch.isnan(intensities)] = np.inf
    return intensities.transpose(0,1) # [no_points, no_views, 3]

def photometric_loss(observed_intensities, query_intensities, delta=DELTA):
    # Huber loss
    diff = np.abs(observed_intensities - query_intensities)
    
    d = np.minimum(diff, delta)
    loss = d*(diff - 0.5*d)
    return loss * 1

def photometric_loss_gpu(observed_intensities, query_intensities, delta=DELTA):

    # Huber loss
    diff = torch.abs(observed_intensities - query_intensities)
    d = torch.min(diff, immutable_tensor([DELTA], dtype=diff.dtype).cuda())
    d2 = torch.min(diff, delta)
    loss = d*(diff - 0.5*d)
    loss = d2*(diff - 0.5*d2)
    loss[torch.isnan(loss)] = np.inf
    return loss * 1

    # # Huber loss
    # diff = torch.abs(observed_intensities - query_intensities)
    # delta = delta

    # d = torch.min(diff, delta)
    # loss = d*(diff - 0.5*d)
    # return loss

def solve_depth_range_gpu(visual_ray_dirs, optical_centre, img_funcs_batch_gpu, Ps, min_len, max_len, init_bin_size, final_bin_size, near_inclusive=True, far_inclusive=True):
    '''
    GPU only, implements a ray_based visual hull
    visual_ray_dirs [no_points, 3] visual rays unit vectors, defined in world coords
    optical_centre [3] world coords, 
    min_len, scalar or [no_points], minimal length
    max_len, scalar or [no_points], maximal length
    init_bin_size, float, fractional initial bin size
    final_bin_size, float, fractional final bin size
    near_inclusive, whether the returned min length should be strictly inside object
    far_inclusive, whether the returned max length should be strictly inside object

    returns min_length, max_length, [no_points, 2], lengths in ray directions
    '''

    min_ray = immutable_tensor([min_len], dtype=visual_ray_dirs.dtype).reshape([-1, 1]).cuda() * visual_ray_dirs # [no_points, 3]
    max_ray = immutable_tensor([max_len], dtype=visual_ray_dirs.dtype).reshape([-1, 1]).cuda() * visual_ray_dirs # [no_points, 3]

    rays = min_ray + (max_ray - min_ray) * immutable_tensor(np.arange(-init_bin_size, 1+init_bin_size, init_bin_size)).reshape([-1,1,1]).cuda() # [no_bins, no_points, 3]

    xyzs = rays + optical_centre # [no_bins, no_points, 3]

    no_bins, no_points = xyzs.shape[:2]

    in_mask = query_intensity_profile_LHS_batch_gpu(xyzs.reshape([-1,3]), img_funcs_batch_gpu, Ps)\
        .reshape([no_bins, no_points, -1]).sum(-1) < float('inf') # [no_bins, no_points]

    if in_mask[0].sum() > 0:
        raise ValueError('Part of visual hull is outside search range. Consider reducing min_len.')
    if in_mask[-1].sum() > 0:
        raise ValueError('Part of visual hull is outside search range. Consider increasing max_len.')


    # solving min
    min_step_no = immutable_tensor(np.zeros([no_points]) + no_bins + 1).cuda() # [no_points]
    max_step_no = immutable_tensor(np.zeros([no_points]) - 1).cuda() # [no_points]
    for i in range(no_bins-1):
        # scan through i's
        mask = (~in_mask[i]) * in_mask[i+1]
        min_step_no[mask] = torch.min(min_step_no[mask], immutable_tensor([i], dtype=min_step_no.dtype).cuda())

        mask = in_mask[i] * (~in_mask[i+1])
        max_step_no[mask] = torch.max(min_step_no[mask], immutable_tensor([i+1], dtype=min_step_no.dtype).cuda())
    
    
    _mask = min_step_no <= no_bins # [no_points]
    no_in_range_points = _mask.sum()

    min_min_ray = min_ray[_mask] + min_step_no[_mask].reshape([-1,1]) * init_bin_size * (max_ray[_mask] - min_ray[_mask]) # [no_in_range_points, 3]
    min_max_ray = min_ray[_mask] + (min_step_no[_mask]+1).reshape([-1,1]) * init_bin_size * (max_ray[_mask] - min_ray[_mask]) # [no_in_range_points, 3]
    max_min_ray = min_ray[_mask] + (max_step_no[_mask]-1).reshape([-1,1]) * init_bin_size * (max_ray[_mask] - min_ray[_mask]) # [no_in_range_points, 3]
    max_max_ray = min_ray[_mask] + max_step_no[_mask].reshape([-1,1]) * init_bin_size * (max_ray[_mask] - min_ray[_mask]) # [no_in_range_points, 3]

    def find_min(min_ray, max_ray):
        mid_ray = (min_ray+max_ray)*0.5 # [no_in_range_points, 3]
        xyzs = optical_centre + mid_ray # [no_in_range_points, 3]
        mask = query_intensity_profile_LHS_batch_gpu(xyzs.reshape([-1,3]), img_funcs_batch_gpu, Ps)\
            .reshape([no_in_range_points, -1]).sum(-1) < float('inf') # [no_in_range_points]

        max_ray[mask] = mid_ray[mask]
        min_ray[~mask] = mid_ray[~mask]
    
    def find_max(min_ray, max_ray):
        mid_ray = (min_ray+max_ray)*0.5 # [no_in_range_points, 3]
        xyzs = optical_centre + mid_ray # [no_in_range_points, 3]
        mask = query_intensity_profile_LHS_batch_gpu(xyzs.reshape([-1,3]), img_funcs_batch_gpu, Ps)\
            .reshape([no_in_range_points, -1]).sum(-1) < float('inf') # [no_in_range_points]

        max_ray[~mask] = mid_ray[~mask]
        min_ray[mask] = mid_ray[mask]

    bin_size = init_bin_size
    
    while bin_size  > final_bin_size:
        find_min(min_min_ray, min_max_ray)
        find_max(max_min_ray, max_max_ray)
        bin_size = bin_size * 0.5
    
    if near_inclusive:
        min_ray[_mask] = min_max_ray
    else:
        min_ray[_mask] = min_min_ray
    
    if far_inclusive:
        max_ray[_mask] = max_min_ray
    else:
        max_ray[_mask] = max_max_ray
    
    min_length = torch.norm(min_ray, p=2, dim=-1).reshape([-1,1]) # [no_points, 1]
    max_length = torch.norm(max_ray, p=2, dim=-1).reshape([-1,1]) # [no_points, 1]

    # assert(all(torch.norm(min_length * visual_ray_dirs - min_ray, p=2, dim=-1) < 1e-5))
    # assert(all(torch.norm(max_length * visual_ray_dirs - max_ray, p=2, dim=-1) < 1e-5))

    return torch.cat([min_length, max_length], dim=-1), _mask



def query_intensities_gpu(xyzs, normals, img_funcs_batch_gpu, optical_centres, Ps, log_brdf_func_gpu, epsilon):
    '''
    GPU only
    xyzs [no_points, 3]
    normals [no_points, 3]
    optical_centres [no_views, 3] or [no_views, 3, 2]
    returns a [no_points, no_views, 3] matrix of lhs-rhs errors
    '''
    lhs_intensities = query_intensity_profile_LHS_batch_gpu(xyzs, img_funcs_batch_gpu, Ps, epsilon)
    rhs_intensities = render_point_gpu(xyzs, normals, optical_centres, log_brdf_func_gpu, epsilon)

    return lhs_intensities, rhs_intensities

    
def query_photometric_loss(xyzs, normals, img_funcs_batch_gpu, optical_centres, Ps, log_brdf_func_gpu, N, epsilon, return_mask=False):
    '''
    GPU only
    xyzs [no_points, 3]
    normals [no_points, 3]
    returns a [no_points] photomteric loss, [no_points, no_views] selection mask bool
    '''
    lhs_intensities, rhs_intensities = query_intensities_gpu(xyzs, normals, img_funcs_batch_gpu, optical_centres, Ps, log_brdf_func_gpu, epsilon) # [no_points, no_views, 3]
    raw_err = lhs_intensities - rhs_intensities
    delta = raw_err * 0 + DELTA
    delta[:,0] = float('inf')
    p_loss = photometric_loss_gpu(raw_err, 0, delta) # [no_points, no_views, 3]

    ######################
    #p_loss[torch.abs(p_loss[:,0]).sum(-1) > DELTA * 6] = np.inf
    ######################

    p_loss = p_loss.sum(-1) # [no_points, no_views]
    
    if N <=0:
        N = N + Ps.shape[0] - 1

    # p_loss = torch.topk(p_loss, k=N, dim=1, largest=False)[0] # [no_points, N]
    # p_loss = p_loss.sum(1) # [no_points]

    p_loss_1 = torch.topk(p_loss[:,1:], k=N, dim=1, largest=False)[0] # [no_points, N]
    select_mask = (p_loss <= p_loss_1[:, -1:])
    select_mask[:,0] = True
    p_loss = (p_loss[:,0] + p_loss_1.sum(1)) # [no_points]
    if return_mask:
        return p_loss, select_mask
    return p_loss

def query_surface_shape_loss(xyzs, xyzs_n, normals):
    '''
    GPU only
    xyzs [no_points, 3]
    xyzs_n [no_points, 4 (no_neighbors), 3]
    normals [no_candidates, no_points, 3]
    '''
    vs = (xyzs_n - xyzs.unsqueeze(1))
    vs[torch.isinf(vs)] = 0
    vs[torch.isnan(vs)] = 0
    return ((vs * normals.unsqueeze(dim=2)).sum(-1) ** 2).sum(2) # [no_candidates, no_points]

def query_lagragian_loss(depth, depth_hat):
    '''
    GPU only
    depth [no_candidates, no_points]
    depth_hat [no_points]
    '''
    loss = (depth - depth_hat)**2
    loss[torch.isinf(loss)] = 0
    loss[torch.isnan(loss)] = 0
    return loss # [no_candidates, no_points]




def patchmatch(red_dist, red_normals, red_rays, red_nnf, black_dist, black_normals, black_rays, black_nnf, no_candidates, theta_d_dev, len_dev, no_iterations, atteuation_factor, img_funcs_batch_gpu, optical_centres, the_optical_centre, the_optical_axis, Ps, log_brdf_func_gpu, N, epsilon, _rd=None, _rn=None, _bd=None, _bn=None, z_hat=None, nnf=None, lambda_n=0, lambda_z=0, min_depth=0):

    def __rectify_normals(view_rays, normals):
        '''
        view_rays [no_points, 3]
        normals [no_points, 3]
        force view_rays and normals to have negative product, if not flip normals
        change normals in place
        '''
        return
        pass
        flip_mask = (view_rays*normals).sum(-1) > 0
        normals[flip_mask] = -normals[flip_mask]
        
    def __pm_red_pass(red_depths, red_normals, red_rays, red_urays, red_nnf, black_depths, black_normals, optical_centre, red_hat_depth, red_hat_xyz, red_hat_xyz_n):
        '''
        GPU only
        red_depths [no_reds], xyzs+normals
        red_normals [no_reds, 3], xyzs+normals
        red_rays [no_reds, 3], vectors of search direction, can be non-uniform
        red_urays [no_reds, 3], scaled red_rays so that last dimension is 1
        red_nnf [no_reds, no_neighbors], int values in [0, no_blacks)
        '''

        no_reds = red_depths.shape[0]
        no_reds_idx = immutable_tensor(range(no_reds)).cuda()

        red_depths_gathered = black_depths[red_nnf.reshape([-1])].reshape(red_nnf.shape) # [no_reds, no_neighbors]
        red_depths_gathered = torch.cat([red_depths.unsqueeze(1), red_depths.unsqueeze(1), red_depths_gathered], dim=1)  # [no_reds, no_neighbors + 2]

        red_normals_gathered = black_normals[red_nnf.reshape([-1])].reshape(red_nnf.shape + (3,)) # [no_reds, no_neighbors, 3]
        normalised_red_rays= torch.nn.functional.normalize(red_rays).reshape([-1,3,1]) # [no_reds, 3, 1]
        red_normals_revert_R = 2 * torch.matmul(normalised_red_rays, normalised_red_rays.transpose(1,2)) - torch.eye(3).cuda() # [no_reds, 3, 3]
        red_normals_revert = torch.matmul(red_normals.unsqueeze(1), red_normals_revert_R) # [no_reds, 1, 3]
        red_normals_gathered = torch.cat([red_normals.unsqueeze(1), red_normals_revert, red_normals_gathered], dim=1)  # [no_reds, no_neighbors + 2, 3]
        red_normals_gathered_ = red_normals_gathered.reshape([-1,3])
        __rectify_normals(red_rays.unsqueeze(1).expand_as(red_normals_gathered).reshape([-1,3]), red_normals_gathered_)
        red_normals_gathered = red_normals_gathered_.reshape(red_normals_gathered.shape)


        red_xyzs_gathered = red_depths_gathered.unsqueeze(2) * red_urays.unsqueeze(1) + optical_centre # [no_reds, no_neighbors, 3]

        loss = query_photometric_loss(red_xyzs_gathered.reshape([-1,3]), # [no_reds*(no_neighbors+1), 3]
                                      red_normals_gathered.reshape([-1,3]),  # [no_reds*(no_neighbors+1), 3]
                                      img_funcs_batch_gpu, 
                                      optical_centres, Ps, log_brdf_func_gpu, N, epsilon) # [no_reds*(no_neighbors+1)]
        
        
        
        loss = loss.reshape([no_reds, -1]) # [no_reds, no_neighbors+1]
        if has_hat:

            loss_shape_penalty = query_surface_shape_loss(red_hat_xyz, red_hat_xyz_n, red_normals_gathered.transpose(0,1)).transpose(0,1) * lambda_n # [no_reds, no_neighbors+1]
            loss_shape_lagrang = query_lagragian_loss(red_depths_gathered.transpose(0,1), red_hat_depth).transpose(0,1) * lambda_z # [no_reds, no_neighbors+1]
            loss = loss + loss_shape_penalty + loss_shape_lagrang

        idx = loss.argmin(dim=-1) # [no_reds], in [0, no_neighbors+1)

        new_red_depths = red_depths_gathered[no_reds_idx, idx] # [no_reds]
        new_red_normals = red_normals_gathered[no_reds_idx, idx] # [no_reds, 3]

        return new_red_depths, new_red_normals

    def __pm_black_pass(black_depths, black_normals, black_rays, black_urays, black_nnf, red_depths, red_normals, optical_centre, black_hat_depth, black_hat_xyz, black_hat_xyz_n):
        return __pm_red_pass(black_depths, black_normals, black_rays, black_urays, black_nnf, red_depths, red_normals, optical_centre, black_hat_depth, black_hat_xyz, black_hat_xyz_n)
    
    def __pm_randomise(depths, normals, rays, urays, theta_d_dev, len_dev, no_candidates, optical_centre, _d, _n, hat_depth, hat_xyz, hat_xyz_n):
        # generate random shifts
        no_points = depths.shape[0]
        no_points_idx = immutable_tensor(range(no_points)).cuda()

        angular_shift = sample_unit_vectors(no_points*no_candidates, theta_d_dev, dist_d='normal') # [no_candidates * no_points, 3]
        angular_shift = immutable_tensor(angular_shift.reshape([no_candidates, no_points, 3])).cuda() # [no_candidates, no_points, 3]
        
        half_vec = normals.clone()
        half_vec[:,-1] = half_vec[:,-1] + 1 # [no_points, 3]
        half_vec = torch.nn.functional.normalize(half_vec, dim=-1, p=2) # [no_points, 3]
        Rs = 2 * torch.bmm(half_vec.reshape([no_points, 3, 1]), half_vec.reshape([no_points, 1, 3])) - torch.eye(3, dtype=normals.dtype).cuda() #[no_points, 3, 3]
        new_normals = torch.bmm(angular_shift.transpose(0, 1), Rs).transpose(0, 1) # [no_candidates, no_points, 3]


        # new_depths = depths + immutable_tensor((np.random.rand(no_candidates, no_points)-0.5)).cuda() * len_dev # [no_candidates, no_points]
        new_depths = depths + immutable_tensor((np.random.normal(0, 1, [no_candidates,no_points]))).cuda() * len_dev # [no_candidates, no_points]
        new_depths = torch.max(new_depths, immutable_tensor([min_depth], dtype=new_depths.dtype).cuda())

        new_normals = torch.cat([normals.unsqueeze(0), new_normals], dim=0) # [no_candidates + 1, no_points, 3]
        new_depths = torch.cat([depths.unsqueeze(0), new_depths], dim=0) # [no_candidates + 1, no_points]

        if _d is not None and _n is not None:
            new_normals = torch.cat([_n.unsqueeze(0), new_normals], dim=0) # [no_candidates + 1, no_points, 3]
            new_depths = torch.cat([_d.unsqueeze(0), new_depths], dim=0) # [no_candidates + 1, no_points]
        
        new_normals_ = new_normals.reshape([-1,3])
        __rectify_normals(rays.unsqueeze(0).expand_as(new_normals).reshape([-1,3]), new_normals_)
        new_normals = new_normals_.reshape(new_normals.shape)

        new_xyzs = new_depths.unsqueeze(2) * urays.unsqueeze(0) + optical_centre # [no_candidates + 1, no_points, 3]

        loss, select_mask = query_photometric_loss(new_xyzs.reshape([-1,3]), # [(no_candidates + 1) * no_points, 3]
                                      new_normals.reshape([-1,3]),  # [(no_candidates + 1) * no_points, 3]
                                      img_funcs_batch_gpu, 
                                      optical_centres, Ps, log_brdf_func_gpu, N, epsilon, True) # [(no_candidates + 1) * no_points]

        loss = loss.reshape([-1, no_points]) # [no_candidates + 1, no_points]
        select_mask = select_mask.reshape(loss.shape + (-1,)) # [no_candidates + 1, no_points, no_views]
        if has_hat:

            loss_shape_penalty = query_surface_shape_loss(hat_xyz, hat_xyz_n, new_normals) * lambda_n # [no_reds, no_neighbors+1]
            loss_shape_lagrang = query_lagragian_loss(new_depths, hat_depth) * lambda_z# [no_reds, no_neighbors+1]
            loss = loss + loss_shape_penalty + loss_shape_lagrang

        idx = loss.argmin(dim=0)

        ret_depths = new_depths[idx, no_points_idx] # [no_points]
        ret_normals = new_normals[idx, no_points_idx] # [no_points, 3]
        select_mask = select_mask[idx, no_points_idx] # [no_points, no_views]
        mask = loss.min(dim=0)[0] < np.inf

        return ret_depths, ret_normals, mask, loss.min(dim=0)[0][mask].mean(), select_mask

    # the_optical_centre, the_optical_axis
    the_optical_axis = torch.nn.functional.normalize(the_optical_axis, dim=-1, p=2)
    red_urays = red_rays / (red_rays * the_optical_axis).sum(-1, keepdim=True)
    black_urays = black_rays / (black_rays * the_optical_axis).sum(-1, keepdim=True)

    red_depths2dist_ratio = (torch.nn.functional.normalize(red_rays, dim=-1, p=2) * the_optical_axis).sum(-1)
    black_depths2dist_ratio = (torch.nn.functional.normalize(black_rays, dim=-1, p=2) * the_optical_axis).sum(-1)

    assert(all(red_depths2dist_ratio > 0))
    assert(all(red_depths2dist_ratio <= 1+1e-5))
    assert(all(black_depths2dist_ratio > 0))
    assert(all(black_depths2dist_ratio <= 1+1e-5))

    red_depths = red_dist * red_depths2dist_ratio
    black_depths = black_dist * black_depths2dist_ratio

    if z_hat is not None and nnf is not None:
        assert(z_hat.shape[0] == red_dist.shape[0] + black_dist.shape[0])
        red_depth_hat = z_hat[:red_dist.shape[0]]
        black_depth_hat = z_hat[red_dist.shape[0]:]

        red_xyz_hat = red_depth_hat.unsqueeze(1) * red_urays
        black_xyz_hat = black_depth_hat.unsqueeze(1) * black_urays
        xyz_hat = torch.cat([red_xyz_hat, black_xyz_hat], dim=0)
        red_xyz_hat_n = xyz_hat[nnf[:red_dist.shape[0]].reshape([-1])].reshape(nnf[:red_dist.shape[0]].shape + (3,))
        black_xyz_hat_n = xyz_hat[nnf[red_dist.shape[0]:].reshape([-1])].reshape(nnf[red_dist.shape[0]:].shape + (3,))
        has_hat = True
    else:
        red_depth_hat = None
        black_depth_hat = None
        red_xyz_hat = None
        black_xyz_hat = None
        red_xyz_hat_n = None
        black_xyz_hat_n = None
        has_hat = False


    for _ in range(no_iterations):

        # red pass
        red_depths, red_normals = __pm_red_pass(red_depths, red_normals, red_rays, red_urays, red_nnf, black_depths, black_normals, the_optical_centre, red_depth_hat, red_xyz_hat, red_xyz_hat_n)
        red_depths, red_normals, mask_r, loss_r, select_mask_r = __pm_randomise(red_depths, red_normals, red_rays, red_urays, theta_d_dev, len_dev, no_candidates, the_optical_centre, _rd, _rn, red_depth_hat, red_xyz_hat, red_xyz_hat_n)

        # black pass
        black_depths, black_normals = __pm_black_pass(black_depths, black_normals, black_rays, black_urays, black_nnf, red_depths, red_normals, the_optical_centre, black_depth_hat, black_xyz_hat, black_xyz_hat_n)
        black_depths, black_normals, mask_b, loss_b, select_mask_b = __pm_randomise(black_depths, black_normals, black_rays, black_urays, theta_d_dev, len_dev, no_candidates, the_optical_centre, _bd, _bn, black_depth_hat, black_xyz_hat, black_xyz_hat_n)

        theta_d_dev = theta_d_dev * atteuation_factor
        len_dev = len_dev * atteuation_factor

    return red_depths / red_depths2dist_ratio, red_normals, black_depths / black_depths2dist_ratio, black_normals, mask_r, mask_b, loss_r * mask_r.sum() / (mask_r.sum()+mask_b.sum()) + loss_b * mask_b.sum() / (mask_r.sum()+mask_b.sum()), select_mask_r,select_mask_b



def solve_d_hat(normals, depths, urays, nnf, mask, lambda_n, lambda_z, lambda_l=0):
    '''
    urays has length 1 when projected onto optical axis
    nnf [no_points, no_neighbors]
    normals and depths are only considered when mask is true
    returned values are only valid when mask is true
    '''
    no_points = normals.shape[0]
    i = immutable_tensor(np.arange(no_points), dtype=torch.long).unsqueeze(1).cuda().expand_as(nnf).reshape([-1]) # [no_points*no_neighbors]

    nnf[nnf == -1] = i.reshape(nnf.shape)[nnf == -1]
    j = nnf.reshape([-1]) # [no_points*no_neighbors]
    

    n_i = normals[i] # [no_points*no_neighbors, 3]

    a_i = urays[i]
    a_j = urays[j]

    k_ii = (n_i*a_i).sum(-1).cpu().numpy() # [no_points*no_neighbors]
    k_ij = (n_i*a_j).sum(-1).cpu().numpy() # [no_points*no_neighbors]

    mask_ij = (mask[i] * mask[j]).cpu().numpy().astype(bool) # [no_points*no_neighbors] boolean
    i = i.cpu().numpy().astype(np.int32)
    j = j.cpu().numpy().astype(np.int32)
    
    col_idx = np.stack([i, j], axis=-1) # [no_points*no_neighbors, 2]
    coefs = np.stack([k_ii, -k_ij], axis=-1) # [no_points*no_neighbors, 2]

    no_eq = mask_ij.sum()
    row_idx = np.stack([np.arange(no_eq), np.arange(no_eq)], axis=-1) # [no_eq, 2]
    col_idx = col_idx[mask_ij] # [no_eq, 2]
    coefs = coefs[mask_ij] * lambda_n # [no_eq, 2]
    b = np.zeros([no_eq]) # [no_eq]

    col_idx_ = np.arange(no_points) # [no_points]
    coefs_ = np.ones(no_points) # [no_points]
    b_ = depths.cpu().numpy() # [no_points]

    mask = mask.cpu().numpy().astype(bool)

    row_idx_ = np.arange(mask.sum()) + no_eq # [no_points]
    col_idx_ = col_idx_[mask]
    coefs_ = coefs_[mask] * lambda_z
    b_ = b_[mask] * lambda_z

    no_eq = no_eq + mask.sum()

    row_idx__ = np.stack([[np.arange(no_points) + no_eq]] * 5, -1) # [no_points, 5]
    col_idx__ = np.concatenate([np.arange(no_points).reshape([-1,1]), nnf.cpu().numpy().astype(np.int32)], axis=-1) # [no_points, 1+4]
    coefs__ = np.array([[-4, 1, 1, 1, 1]]*no_points) * lambda_l
    b__ = np.zeros([no_points])

    # A = sp.sparse.coo_matrix((np.concatenate([coefs.flatten(), coefs_.flatten()]), 
    #                          (np.concatenate([row_idx.flatten(), row_idx_.flatten()]), np.concatenate([col_idx.flatten(), col_idx_.flatten()]))), 
    #                          shape=(mask_ij.sum()+mask.sum(), no_points))
    # B = np.concatenate([b, b_])

    A = sp.sparse.coo_matrix((np.concatenate([coefs.flatten(), coefs_.flatten(), coefs__.flatten()]), 
                             (np.concatenate([row_idx.flatten(), row_idx_.flatten(), row_idx__.flatten()]), np.concatenate([col_idx.flatten(), col_idx_.flatten(), col_idx__.flatten()]))), 
                             shape=(mask_ij.sum()+mask.sum()+no_points, no_points))

    B = np.concatenate([b, b_, b__])

    z_hat = lsqr(A, B)[0]

    return immutable_tensor(z_hat, dtype=depths.dtype).cuda()


