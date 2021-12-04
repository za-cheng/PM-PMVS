
from render import *
from solve_shape import *
from solve_brdf import *
from utils import *
from params import *
import numpy as np
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore") # get rid of pytorch warnings


def save_hist(x, bins, label='error', fname=None):
    plt.figure()
    plt.hist(x, bins=bins)
    plt.xlabel(label, fontweight='bold')
    plt.ylabel('no. surface points', fontweight='bold')
    plt.savefig(fname)
    plt.close()

def save_plot(x, label='error', fname=None):
    plt.figure()
    plt.plot(np.arange(len(x)), x, '-', lw=2)
    plt.xlabel('iterations', fontweight='bold')
    plt.ylabel(label, fontweight='bold')
    plt.savefig(fname)
    plt.close()


def generate_extrinsics(mesh, no_views, model_str=None):
    xys = mesh.vertices[:, :2]
    x_max, x_min = np.max(mesh.vertices[:, 0]), np.min(mesh.vertices[:, 0])
    y_max, y_min = np.max(mesh.vertices[:, 1]), np.min(mesh.vertices[:, 1])
    z_max, z_min = np.max(mesh.vertices[:, 2]), np.min(mesh.vertices[:, 2])

    np.random.seed(0)

    vrange = np.array([(x_max-x_min), (y_max-y_min), (z_max-z_min)*0.5])

    v_mean = np.array([x_max+x_min, y_max+y_min, 0*z_max+(z_max-z_min)*3]) * 0.5

    views_world_coords = np.random.rand(no_views,3) * np.asarray(vrange)*5 - vrange + v_mean
    views_world_coords[0] = [1.0/40, 1.0/40, (z_max-z_min)*3 + z_min]

    # make the process reproducable...

    # force the first view to be fronto facing
    # TODO: test why x=0,y=0 doesn't work
    
    extrinsics_mats = [create_view(v) for v in views_world_coords]

    return np.array(extrinsics_mats)

def get_results(r_mask, r_points, r_normals, vmask_r, b_mask, b_points, b_normals, vmask_b, brdf_func, optical_centre, optical_axis, epsilon):
    '''
    ray_map normalised in depth direction
    '''
    r_mask = immutable_tensor(r_mask.astype(np.uint8)).cuda()
    b_mask = immutable_tensor(b_mask.astype(np.uint8)).cuda()

    optical_axis = torch.nn.functional.normalize(optical_axis, p=2, dim=-1)
    # rednder first view
    l_img = torch.zeros(r_mask.shape + (3,), dtype=r_points.dtype).cuda() + np.inf
    n_img = torch.zeros(r_mask.shape + (3,), dtype=r_points.dtype).cuda() + np.inf
    d_img = torch.zeros(r_mask.shape, dtype=r_points.dtype).cuda() + np.inf

    r_intensities = render_point_gpu(r_points, r_normals, optical_centre.unsqueeze(0), brdf_func, epsilon).squeeze(1)
    b_intensities = render_point_gpu(b_points, b_normals, optical_centre.unsqueeze(0), brdf_func, epsilon).squeeze(1)


    r_depths = (optical_axis * (r_points - optical_centre)).sum(-1)
    b_depths = (optical_axis * (b_points - optical_centre)).sum(-1)

    def assign(a, b, mask1, mask2):
        aa = a[mask1]
        aa[mask2] = b[mask2]
        a[mask1] = aa

    assign(l_img, r_intensities, r_mask, vmask_r)
    assign(l_img, b_intensities, b_mask, vmask_b)

    assign(n_img, r_normals, r_mask, vmask_r)
    assign(n_img, b_normals, b_mask, vmask_b)

    assign(d_img, r_depths, r_mask, vmask_r)
    assign(d_img, b_depths, b_mask, vmask_b)

    return l_img, n_img, d_img

    



def main(exp_name, input_file, max_dist, min_dist, kernel_connectivity, lambda_c, no_outter_iterations, no_inner_iterations, no_qpm_iterations, no_patchmatch_iterations, theta_d_dev, len_dev, outter_attenuation, inner_attenuation, no_candidates, lambda_n, lambda_z_init, lambda_z_increase_rate_outter, lambda_z_increase_rate_inner, lambda_l, epsilon=1e-10):


    np.random.seed(0)

    ####################### PREPARATIONS ########################

    inputs = np.load(input_file, allow_pickle=True).item()
    imgs = inputs['imgs']
    K = inputs['K']
    Ps = inputs['P']
    height, width = inputs['height'], inputs['width']

    log_imgs = [np.log(np.maximum(img, epsilon)) for img in imgs]
    log_img_funcs = convert_img_interp_cpu_to_gpu([bilinear_image_interp(img, fill_val=np.nan) for img in log_imgs])
    
    log_D = np.log(np.maximum(np.load('D.npy'), epsilon)) + np.log(np.cos(np.arange(90) * np.pi / 180)).reshape([-1,1,1])

    optical_centre, visual_ray_dirs = create_view_angles_from_vcam(K, Ps[0], height, width)

    optical_centres = [create_view_angles_from_vcam(K, P, height, width)[0] for P in Ps]
    optical_centres = immutable_tensor(optical_centres).cuda()


    ##### partition pixels into two groups : red and black #####
    ##### this partition is later used in parallel patchmatch #####
    maskr = np.zeros([height,width], dtype=bool)
    maskb = np.zeros([height,width], dtype=bool)
    
    for i in range(height):
        for j in range(width):
            if (i+j) % 2 == 0:
                maskr[i,j] = True
            else:
                maskb[i,j] = True
                    

    optical_axis = Ps[0][:3,:3].T.dot([0,0,-1])
    optical_axis = immutable_tensor(optical_axis).cuda()

    KPs = immutable_tensor([K.dot(P) for P in Ps]).cuda()

    mask = np.isnan(log_imgs[0][...,0].reshape([-1]))==False
    mask = mask.reshape([height, width])
    for i in range(height):
        for j in range(width):
            if i < 0 or i >= height or j < 0 or j >= width or np.isnan(log_imgs[0][i,j].sum()):
                mask[i,j] = False
    mask = mask.reshape([-1])

    maskr = mask * maskr.flatten()
    maskb = mask * maskb.flatten()

    r_id = np.zeros((height, width), dtype=np.int32) - 1
    b_id = np.zeros((height, width), dtype=np.int32) - 1

    r_id[maskr.reshape([height, width])] = np.arange(maskr.sum())
    b_id[maskb.reshape([height, width])] = np.arange(maskb.sum())

    r_nnf = []
    b_nnf = []
    img_nnf_r = []
    img_nnf_b = []

    def find_list(idmap, i, j, kernel_connectivity=kernel_connectivity):
        if kernel_connectivity == 20:
            id_shift = [
                (-3, 0),
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 0),
                (-1, 2),
                (0, -3),
                (0, -1),
                (0, 1),
                (0, 3),
                (1, -2),
                (1, 0),
                (1, 2),
                (2, -1),
                (2, 1),
                (3, 0),
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5),
            ]
        elif kernel_connectivity == 32:
            id_shift = [
                (-3, 0),
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 0),
                (-1, 2),
                (0, -3),
                (0, -1),
                (0, 1),
                (0, 3),
                (1, -2),
                (1, 0),
                (1, 2),
                (2, -1),
                (2, 1),
                (3, 0),
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5),
                (9, 0),
                (-9, 0),
                (0, 9),
                (0, -9),
                (4, 5),
                (5, -4),
                (-4, -5),
                (-5, 4),
                (15, 0),
                (-15, 0),
                (0, 15),
                (0, -15),
            ]
        elif kernel_connectivity == 8:
            id_shift = [
                (-1, 0),
                (0, -1),
                (0, 1),
                (1, 0),
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5),
            ]
        
        elif kernel_connectivity == 4:
            id_shift = [
                (-1, 0),
                (0, -1),
                (0, 1),
                (1, 0),
            ]
        rt_l = []
        for _i, _j in id_shift:
            _i = _i + i
            _j = _j + j
            
            if _i >=0 and _i < idmap.shape[0] and _j>=0 and _j < idmap.shape[1]:
                rt_l.append(idmap[_i, _j])
            else:
                rt_l.append(-1)
        return rt_l

    _r, _b = 0, 0

    for i in range(height):
        for j in range(width):
            if r_id[i,j] > -1:
                assert(r_id[i,j] == _r)
                _r = _r + 1
                r_nnf.append(find_list(b_id, i, j))
                img_nnf_r.append(find_list(b_id, i, j, 4))
            if b_id[i,j] > -1:
                assert(b_id[i,j] == _b)
                _b = _b + 1
                b_nnf.append(find_list(r_id, i, j))
                img_nnf_b.append(find_list(r_id, i, j, 4))

    r_nnf = immutable_tensor(r_nnf, dtype=torch.long).cuda()           
    b_nnf = immutable_tensor(b_nnf, dtype=torch.long).cuda() 

    img_nnf = torch.cat([immutable_tensor(img_nnf_r, dtype=torch.long) + len(img_nnf_r), immutable_tensor(img_nnf_b, dtype=torch.long)], dim=0).cuda()
            
    optical_centre = immutable_tensor(optical_centre).cuda()
    visual_ray_dirs_r = visual_ray_dirs.reshape([-1, 3])[maskr]
    visual_ray_dirs_r = immutable_tensor(visual_ray_dirs_r).cuda()

    visual_ray_dirs_b = visual_ray_dirs.reshape([-1, 3])[maskb]
    visual_ray_dirs_b = immutable_tensor(visual_ray_dirs_b).cuda()

    visual_ray_dirs_r = visual_ray_dirs_r / torch.norm(visual_ray_dirs_r, p=2, dim=-1).unsqueeze(-1)
    visual_ray_dirs_b = visual_ray_dirs_b / torch.norm(visual_ray_dirs_b, p=2, dim=-1).unsqueeze(-1)

    # compute normal from depth, used only when initialize shape for solving camera response gamma in the first iteration
    def cal_normal(id_r, id_b, xyzr, xyzb):
        conn_list = [
            (1,0),
            (1,-1),
            (0,-1),
            (-1,-1),
            (-1,0),
            (-1,1),
            (0,1),
            (1,0),
        ]
        normals = []
        for i in range(height):
            for j in range(width):
                if id_r[i,j] < 0:
                    continue
                n_xyz = []
                for _i,_j in conn_list:
                    if id_r[i+_i, j+_j] >= 0:
                        n_xyz.append(xyzr[id_r[i+_i, j+_j]])
                    elif id_b[i+_i, j+_j] >= 0:
                        n_xyz.append(xyzb[id_b[i+_i, j+_j]])
                    else:
                        n_xyz.append(xyzr[id_r[i,j]])
                xyz = xyzr[id_r[i,j]]
                normal = xyz*0
                for xyz1, xyz2 in zip(n_xyz[:-1], n_xyz[1:]):
                    normal = normal - np.cross(xyz1, xyz2)
                if np.linalg.norm(normal, ord=2) > 0:
                    normal = normal / np.linalg.norm(normal, ord=2)
                else:
                    normal = [0,0,1] # if no neighbor
                normals.append(normal)
        return np.asarray(normals)
        
    with torch.no_grad():
        lr, v_mask_r = solve_depth_range_gpu(visual_ray_dirs_r, optical_centre, log_img_funcs, KPs, min_len=min_dist, max_len=max_dist, init_bin_size=1e-2, final_bin_size=1e-10, near_inclusive=True, far_inclusive=True)
        lb, v_mask_b = solve_depth_range_gpu(visual_ray_dirs_b, optical_centre, log_img_funcs, KPs, min_len=min_dist, max_len=max_dist, init_bin_size=1e-2, final_bin_size=1e-10, near_inclusive=True, far_inclusive=True)

        points_r = lr[:,0:1] * visual_ray_dirs_r + optical_centre
        points_b = lb[:,0:1] * visual_ray_dirs_b + optical_centre
        
        red_normals = cal_normal(r_id, b_id, points_r.cpu().numpy(), points_b.cpu().numpy())
        black_normals = cal_normal(b_id, r_id, points_b.cpu().numpy(), points_r.cpu().numpy())
        
        red_normals = immutable_tensor(red_normals).cuda()
        black_normals = immutable_tensor(black_normals).cuda()

        red_dist = lr[...,0]
        black_dist = lb[...,0]

        red_dist_ = red_dist
        black_dist_ = black_dist

        red_normals_ = red_normals
        black_normals_ = black_normals

    red_depth2dist = (visual_ray_dirs_r * optical_axis).sum(-1)
    black_depth2dist = (visual_ray_dirs_b * optical_axis).sum(-1)

    red_urays = visual_ray_dirs_r / (visual_ray_dirs_r * optical_axis).sum(-1, keepdim=True)
    black_urays = visual_ray_dirs_b / (visual_ray_dirs_b * optical_axis).sum(-1, keepdim=True)
    urays = torch.cat([red_urays, black_urays], dim=0)
    
    select_mask_r, select_mask_b = None, None
    print('finished prep...')
    

    with tqdm(total=no_outter_iterations*no_inner_iterations) as pbar:
        for _i_ in range(no_outter_iterations):
            
            pbar.set_description(f'iteration #{_i_}/{no_outter_iterations}')

            points_r = red_dist.unsqueeze(1) * visual_ray_dirs_r + optical_centre
            points_b = black_dist.unsqueeze(1) * visual_ray_dirs_b + optical_centre
            
            points_r = points_r[v_mask_r]
            points_b = points_b[v_mask_b]

            normals_r = red_normals[v_mask_r]
            normals_b = black_normals[v_mask_b]

            points = torch.cat([points_r, points_b], dim=0)
            normals = torch.cat([normals_r, normals_b], dim=0)

            brdf_mask = immutable_tensor(np.random.choice(len(points), 500), dtype=torch.long).cuda()
            points = points[brdf_mask]
            normals = normals[brdf_mask]

            if select_mask_r is not None and select_mask_b is not None:
                select_mask_r = select_mask_r[v_mask_r]
                select_mask_b = select_mask_b[v_mask_b]
                select_mask = torch.cat([select_mask_r, select_mask_b], dim=0)[brdf_mask].cpu().numpy().T # [no_views, no_points]
            else:
                select_mask = None

            lhs_intensities = query_intensity_profile_LHS_batch_gpu(points, log_img_funcs, KPs, epsilon).transpose(0,1) #[no_points, no_views, 3]

            ########### solve brdf and/or camera response when fix shape
            k = 15 # k principal components
            D = log_D.reshape([90, -1])
            mu = np.mean(D.reshape([90, -1]), axis=-1, keepdims=True)
            U,_,V = np.linalg.svd(D-mu, full_matrices=False)
            U = U.dot(np.diag(_))[:, :k]
            log_brdf_func, xs, ys, loss_brdf = solve_brdf_fix_shape(U, mu, points.cpu().numpy(), 
                                                        normals.cpu().numpy(), 
                                                        optical_centres.cpu().numpy(), 
                                                        lhs_intensities.cpu().numpy(), 
                                                        lambda_c,
                                                        'huber-2', 
                                                        _i_==0, #solve only gamma if first iteration
                                                        epsilon=epsilon,
                                                        select_mask=(None if _i_ < 40 else select_mask))

            log_brdf_func_gpu = convert_brdf_interp_cpu_to_gpu(log_brdf_func)
            
            plot_brdf(log_brdf_func, xs, ys, f'results/{exp_name}-{_i_:02}-f.png')
            ######################################



            ######################### write results
            if _i_ == 0:
                points_r = red_dist.unsqueeze(1) * visual_ray_dirs_r + optical_centre
                points_b = black_dist.unsqueeze(1) * visual_ray_dirs_b + optical_centre
                coss = np.cos(np.arange(90)*np.pi/180)
                __brdf = load_brdf(None, (coss, np.stack([coss]*3, axis=-1)), log=True, epsilon=epsilon)
                l_img, n_img, d_img = get_results(maskr, points_r, red_normals, v_mask_r, maskb, points_b, black_normals, v_mask_b, convert_brdf_interp_cpu_to_gpu(__brdf), optical_centre, optical_axis, epsilon=epsilon)

                l_img = np.exp(l_img.cpu().numpy())
                l_img[l_img == np.inf] = 0
                l_img = l_img / np.median(l_img) * 0.5

                n_img = (n_img.cpu().numpy() + 1) / 2
                n_img[n_img == np.inf] = 0

                d_img = d_img.cpu().numpy()
                max_d = min(max_dist, d_img[(d_img > 0) & np.isfinite(d_img)].max())
                min_d = max(min_dist, d_img[(d_img > 0) & np.isfinite(d_img)].min())
                d_img = (max_d - d_img) / (max_d - min_d)
                d_img[d_img == np.inf] = 0
                d_img = np.clip(d_img, 0, 1)

                cv2.imwrite(f'results/{exp_name}-{_i_:02}-shape.png', (l_img*255).astype(np.uint8)[...,::-1].reshape([height, width, 3]))
                cv2.imwrite(f'results/{exp_name}-{_i_:02}-normal.png', (n_img*255).astype(np.uint8)[...,::-1].reshape([height, width, 3]))
                cv2.imwrite(f'results/{exp_name}-{_i_:02}-depth.png', (d_img*255).astype(np.uint8).reshape([height, width]))
            ##########################################################################3


            ###################### solve shape when fix brdf
            with torch.no_grad():
                lambda_z = lambda_z_init
                depth_hat = torch.cat([red_dist*red_depth2dist, black_dist*black_depth2dist], dim=0)
                mask = torch.cat([v_mask_r, v_mask_b], dim=0)
                depth_hat[~mask] = np.nan

                theta_d_dev_init = theta_d_dev
                len_dev_init = len_dev

                for __i__ in range(no_inner_iterations):
                    pbar.update(1)
                    for ___i___ in range(no_qpm_iterations):
                        
                        red_dist, red_normals, black_dist, black_normals, v_mask_r, v_mask_b, loss_shape, select_mask_r, select_mask_b = patchmatch(red_dist, red_normals, visual_ray_dirs_r, r_nnf, black_dist, black_normals, visual_ray_dirs_b, b_nnf, no_candidates=no_candidates, theta_d_dev=theta_d_dev, len_dev=len_dev, no_iterations=no_patchmatch_iterations, atteuation_factor=inner_attenuation, img_funcs_batch_gpu=log_img_funcs, optical_centres=optical_centres, the_optical_centre=optical_centre, the_optical_axis=optical_axis, Ps=KPs, log_brdf_func_gpu=log_brdf_func_gpu, N=N, epsilon=epsilon, _rd=red_dist_, _rn=red_normals_, _bd=black_dist_, _bn=black_normals_,  z_hat=depth_hat, nnf=img_nnf, lambda_n=lambda_n, lambda_z=lambda_z, min_depth=min_dist)
                        depths = torch.cat([red_dist*red_depth2dist, black_dist*black_depth2dist], dim=0)
                        normals = torch.cat([red_normals, black_normals], dim=0)
                        mask = torch.cat([v_mask_r, v_mask_b], dim=0)
                        depth_hat = solve_d_hat(normals, depths, urays, img_nnf, mask, lambda_n, lambda_z, lambda_l)
                        depth_hat[~mask] = np.nan

                    lambda_z = lambda_z * lambda_z_increase_rate_inner
                
                theta_d_dev = theta_d_dev_init
                len_dev = len_dev_init


            lambda_z_init = lambda_z_init * lambda_z_increase_rate_outter
            
            
            ####################################################
            points_r = red_dist.unsqueeze(1) * visual_ray_dirs_r + optical_centre
            points_b = black_dist.unsqueeze(1) * visual_ray_dirs_b + optical_centre


            ##########################
            coss = np.cos(np.arange(90)*np.pi/180)
            __brdf = load_brdf(None, (coss, np.stack([coss]*3, axis=-1)), log=True, epsilon=epsilon) # lambertian BRDF for visualization
            l_img, n_img, d_img = get_results(maskr, points_r, red_normals, v_mask_r, maskb, points_b, black_normals, v_mask_b, convert_brdf_interp_cpu_to_gpu(__brdf), optical_centre, optical_axis, epsilon=epsilon)
            ##########################333

            
            l_img = np.exp(l_img.cpu().numpy())
            l_img[l_img == np.inf] = 0
            l_img = l_img / np.median(l_img[l_img > epsilon]) * 0.5
            l_img = np.clip(l_img, 0, 1)

            n_img = (n_img.cpu().numpy() + 1) / 2
            n_img[n_img == np.inf] = 0

            d_img = d_img.cpu().numpy()
            max_d = min(max_dist, d_img[(d_img > 0) & np.isfinite(d_img)].max())
            min_d = max(min_dist, d_img[(d_img > 0) & np.isfinite(d_img)].min())
            d_img = (max_d - d_img) / (max_d - min_d)
            d_img[d_img == np.inf] = 0
            d_img = np.clip(d_img, 0, 1)

            cv2.imwrite(f'results/{exp_name}-{_i_:02}-shape.png', (l_img*255).astype(np.uint8)[...,::-1].reshape([height, width, 3]))
            cv2.imwrite(f'results/{exp_name}-{_i_:02}-normal.png', (n_img*255).astype(np.uint8)[...,::-1].reshape([height, width, 3]))
            cv2.imwrite(f'results/{exp_name}-{_i_:02}-depth.png', (d_img*255).astype(np.uint8).reshape([height, width]))

    return red_dist.cpu().numpy(), red_normals.cpu().numpy(), black_dist.cpu().numpy(), black_normals.cpu().numpy(), v_mask_r.cpu().numpy().astype(bool), v_mask_b.cpu().numpy().astype(bool), visual_ray_dirs_r.cpu().numpy(), visual_ray_dirs_b.cpu().numpy(), optical_centres


main(
    exp_name=exp_name,
    input_file=input_file,
    max_dist=max_dist,
    min_dist=min_dist,
    kernel_connectivity=kernel_connectivity, 
    lambda_c=lambda_c , 
    no_outter_iterations=no_outter_iterations, 
    no_inner_iterations=no_inner_iterations,
    no_qpm_iterations=no_qpm_iterations,
    no_patchmatch_iterations=no_patchmatch_iterations,
    theta_d_dev = theta_d_dev, 
    len_dev = len_dev, 
    outter_attenuation=1, 
    inner_attenuation=inner_attenuation, 
    no_candidates=no_candidates,
    lambda_n = lambda_n, 
    lambda_z_init = lambda_z_init,
    lambda_l = lambda_l,
    lambda_z_increase_rate_outter = 1, 
    lambda_z_increase_rate_inner = lambda_z_increase_rate_inner,
    epsilon=1e-10
    )
