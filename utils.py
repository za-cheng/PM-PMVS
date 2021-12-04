import numpy as np
import brdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
import torch
import trimesh
import h5py
from glob import glob

def normalise(x, axis=-1, epsilon=1e-10):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), epsilon)

def vector_rotation(src, dst, normalise_vectors=True):
    '''returns a rotation matrix that roates src to dst or the other way around'''
    if normalise_vectors:
        src = normalise(src)
        dst = normalise(dst)
    
    half_vector = normalise(src + dst).reshape([3,1])
    R = 2 * (half_vector.dot(half_vector.T)) - np.eye(3)
    return R

def vector_rotation_batch_gpu(src, dst, normalise_vectors=True):
    
    '''
    src [n_batches, 3]
    dst [n_batches, 3]
    returns a rotation matrices that roate src to dst or the other way around, [n_batches,3,3]'''
    if normalise_vectors:
        src = torch.nn.functional.normalize(src, dim=-1, p=2)
        dst = torch.nn.functional.normalize(dst, dim=-1, p=2)
    
    half_vector = torch.nn.functional.normalize(src + dst, dim=-1, p=2).reshape([-1,3,1]) # [n_batches,3,1]
    Rs = 2 * torch.matmul(half_vector, half_vector.transpose(1,2)) - torch.eye(3) # [n_batches, 3, 3]
    return Rs

def angular_error_in_degrees(v1, v2, normalise_vectors=True):
    ''' v1 [3]
        v2 [3]'''
    if normalise_vectors:
        v1 = normalise(v1)
        v2 = normalise(v2)
    return np.arccos(np.clip((v1*v2).sum(), -1, 1)) * 180 / np.pi

def convert_to_homogeneous_coords(x, axis=-1):
    shape_ones = list(x.shape)
    shape_ones[axis] = 1
    return np.concatenate([x, np.ones(shape_ones, dtype=x.dtype)], axis=axis)

def revert_homogeneous_coords(x, axis=-1):
    if axis != -1: raise NotImplementedError
    return np.true_divide(x[...,:-1], x.take(indices=[-1], axis=axis))
    

def load_brdf(material_str, brdf_param=None, log=False, epsilon=1e-10):
    # either brdf_param ([cos_thetas], [rgbs]), or load material_str
    if brdf_param is None:        
        brdf_param = brdf.create_constant_knots_from_merl('BRDF/{}.h5'.format(material_str), dtype=np.float32, no_knots=None)
        brdf_param = brdf_param[0], brdf_param[1] * np.cos(np.arange(90) * np.pi / 180)[...,None]
    brdf_param = np.concatenate([brdf_param[0],[0],[-1]], axis=0), np.concatenate([brdf_param[1],[brdf_param[1][-1]],[brdf_param[1][-1]]], axis=0)
    ret = brdf_interp(brdf_param[0], brdf_param[1], log_scale=log, epsilon=epsilon)
    return ret

def world_coords_2_image_coords(x, P):
    ''' x [n 3], P [3,4]'''
    x = convert_to_homogeneous_coords(x)
    img_coords = revert_homogeneous_coords((x).dot(P.T))
    return img_coords

def interp_brdf_call(x, gran, rhos):
    x_mask = (np.isnan(x) == False)
    x0 = np.floor(x).astype(np.int32)[x_mask]
    x1 = np.ceil(x).astype(np.int32)[x_mask]
    
    ret = np.zeros(x.shape+(3,)) + np.nan
    
    alpha = x[x_mask]-x0
    ret[x_mask] = (1-alpha).reshape((-1,1))*rhos[x0] + alpha.reshape((-1,1))*rhos[x1]
    return ret

class brdf_interp(object):
    def __init__(self, coss, ys, granularity_in_degree=.1, log_scale=False, epsilon=1e-10):
        # coss [n]
        # ys [n, 3]
        # handles nan values
        thetas = np.arange(0, 90+granularity_in_degree, granularity_in_degree)
        thetas_ref = np.arccos(np.clip(coss, -1, 1)) * 180 / np.pi
        #print interpolate.interp1d(thetas_ref, ys, axis=0)(thetas).shape. interpolate.interp1d(thetas_ref, ys, axis=0)(thetas).dtype
        if log_scale:
            self.rhos = interpolate.interp1d(thetas_ref, np.log(np.maximum(ys, epsilon)), axis=0)(thetas)
        else:
            self.rhos = interpolate.interp1d(thetas_ref, ys, axis=0)(thetas)
        self.gran = granularity_in_degree
    def __call__(self, x):  
        x = np.arccos(np.clip(x, -1, 1)) * 180 / np.pi / self.gran
        x = np.minimum(np.maximum(x, 0.0), (len(self.rhos)-1)*1.0)
        return interp_brdf_call(x, self.gran, self.rhos)
    def save(self, file_name):
        np.savez(file_name, self.gran, self.rhos)
    @staticmethod
    def load(file_name):
        ret = brdf_interp(np.array([1,0]), np.ones([2,3]))
        data = np.load(file_name)
        ret.gran, ret.rhos = data['arr_0'], data['arr_1']
        data.close()
        return ret

        

def interp_image_call(x, y, im, fill_val):
    if x < 0 or x > im.shape[1] - 1: return np.array([fill_val]*3).astype(im.dtype)
    if y < 0 or y > im.shape[0] - 1: return np.array([fill_val]*3).astype(im.dtype)
    
    x0 = np.floor(x)
    x1 = x0 + 1
    y0 = np.floor(y)
    y1 = y0 + 1

    x0 = np.minimum(np.maximum(x0, 0), im.shape[1]-1)
    x1 = np.minimum(np.maximum(x1, 0), im.shape[1]-1)
    y0 = np.minimum(np.maximum(y0, 0), im.shape[0]-1)
    y1 = np.minimum(np.maximum(y1, 0), im.shape[0]-1)
    Ia = im[ int(y0), int(x0) ]
    Ib = im[ int(y1), int(x0) ]
    Ic = im[ int(y0), int(x1) ]
    Id = im[ int(y1), int(x1) ]

    wa = 1.0*(x1-x) * (y1-y)
    wb = 1.0*(x1-x) * (y-y0)
    wc = 1.0*(x-x0) * (y1-y)
    wd = 1.0*(x-x0) * (y-y0)

    return (wa*Ia + wb*Ib + wc*Ic + wd*Id)
        
class bilinear_image_interp(object):
    def __init__(self, img, fill_val=-10):
        self.img = img.astype(np.float64) #np.log(np.maximum(img, epsilon))
        self.fill_val = fill_val
    def __call__(self, coords):
        y,x = np.asarray(coords)
        return interp_image_call(x, y, self.img, self.fill_val)

        
def plot_brdf(log_brdf_func, xs, ys, fname=None):
    x = np.arange(90).astype(np.float64)
    y = log_brdf_func(np.cos(x/180*np.pi))
    #ys = np.clip(ys, min(-3, y.min()), max(-2, y.max()))
    xs = np.arccos(np.clip(xs, -1, 1)) * 180 / np.pi
    plt.figure()
    plt.plot(x, y[:,1], 'b', # x, y[:,0], 'r', x, y[:,2], 'b',
        linewidth=2)
    plt.plot(xs, ys[:,1], 'bo', #xs, ys[:,0], 'ro', xs, ys[:,1], 'bo',
        markersize=0.1)
    plt.xlabel('$\\theta_d$', fontsize=12, fontweight='bold')
    plt.ylabel('$log\\rho$', fontsize=12, fontweight='bold')
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()


def dicretize_array(x, no_b, epsilon=1e-10):
    # discretize values linearly, starting from 0, handels nan
    ret = x.copy()
    vals = np.log(np.maximum(ret[np.isnan(x) == False], epsilon))
    
    vals_ref = np.log(np.arange(no_b)[1:])
    
    return ret

##########################################GPU VERSION######################
    

def interp_brdf_call_gpu(x, gran, rhos):
    x0 = torch.floor(x).type(torch.cuda.LongTensor)
    x1 = torch.ceil(x).type(torch.cuda.LongTensor)
    
    alpha = x-x0.type(x.dtype)
    ret = (1-alpha).reshape((-1,1))*rhos[x0] + alpha.reshape((-1,1))*rhos[x1]
    return ret

class brdf_interp_gpu(object):
    def __init__(self, coss=None, ys=None, granularity_in_degree=.1, log_scale=False, epsilon=1e-10):
        # coss [n]
        # ys [n, 3]
        # handles nan values
        
        if coss is not None and ys is not None:
            thetas = np.arange(0, 90+granularity_in_degree, granularity_in_degree)
            thetas_ref = np.arccos(np.clip(coss, -1, 1)) * 180 / np.pi
            #print interpolate.interp1d(thetas_ref, ys, axis=0)(thetas).shape. interpolate.interp1d(thetas_ref, ys, axis=0)(thetas).dtype
            if log_scale:
                self.rhos = interpolate.interp1d(thetas_ref, np.log(np.maximum(ys, epsilon)), axis=0)(thetas)
            else:
                self.rhos = interpolate.interp1d(thetas_ref, ys, axis=0)(thetas)
            self.gran = granularity_in_degree
            
            self.rhos = immutable_tensor(self.rhos).cuda()
            self.gran = immutable_tensor(self.gran).cuda()
        
    def __call__(self, x):  
        x = torch.acos(torch.clamp(x, -1, 1)) * 180 / np.pi / self.gran
        x = torch.clamp(x, 0.0, (len(self.rhos)-2 / self.gran)*1.0)
        # x = torch.clamp(x, 0.0, (len(self.rhos)-2 / self.gran)*1.0)
        return interp_brdf_call_gpu(x, self.gran, self.rhos)
    
        
class bilinear_image_interp_batch_gpu(object):
    def __init__(self, img, fill_val=np.inf):
        ## hotfix to prevent https://github.com/pytorch/pytorch/issues/24823
        ## this is non-ideal but works fine...
        self.img = immutable_tensor(img).permute(0,3,1,2).cuda() #[n_batches, color_channels, h, w]

        self.fill_val = fill_val
        self.h = self.img.shape[-2]
        self.w = self.img.shape[-1]
        
    def __call__(self, coords):
        # coords [n_batches, n_queries, 2]
        out_range_mask = (coords[...,0] < 0) | (coords[...,0] > self.h-1) | \
                            (coords[...,1] < 0) | (coords[...,1] > self.w-1)

        ## hotfix to prevent https://github.com/pytorch/pytorch/issues/24823
        ## this is non-ideal but works fine...
        coords[out_range_mask] = 0.5

        normalise_scale = immutable_tensor([(self.h-1)*0.5, (self.w-1)*0.5], dtype=coords.dtype).cuda()
        coords = coords / normalise_scale - 1
        coords = torch.flip(coords, dims=(-1,))
        
        ret = torch.nn.functional.grid_sample(self.img, coords.unsqueeze(2)).squeeze(-1).transpose(1,2) # [n_batches, n_queries, 3]
        
        ret[out_range_mask] = self.fill_val
        
        return ret

def replace_nan_with_inf(x):
    x = x.copy()
    x[np.isnan(x)]=np.inf
    return x
def convert_img_interp_cpu_to_gpu(li_imgs):
    img = immutable_tensor([replace_nan_with_inf(i.img) for i in li_imgs])
    return bilinear_image_interp_batch_gpu(img)

def convert_brdf_interp_cpu_to_gpu(brdf):
    ret = brdf_interp_gpu()
    ret.rhos = immutable_tensor(brdf.rhos).cuda()
    ret.gran = immutable_tensor(brdf.gran).cuda()
    return ret

def immutable_tensor(x, *args, **aargs):
    if (torch.from_numpy(np.array(x)).dtype.is_floating_point and 'dtype' not in aargs) or ('dtype' in aargs and aargs['dtype'].is_floating_point):
        aargs['dtype'] = torch.float32
    return torch.tensor(x, *args, requires_grad=False, **aargs)

def mutable_tensor(x, *args, **aargs):
    return torch.tensor(x, *args, requires_grad=True, **aargs)

def safe_load_mesh(mesh_str):
   mesh = trimesh.load_mesh(mesh_str)
   mesh.vertices[:,-1] = 0
   mesh.vertex_normals = mesh.vertex_normals * 0
   return mesh

def flip_brdf(brdf):
    brdf.rhos = brdf.rhos[::-1, ...]
    return brdf

def sample_unit_vectors(n, dev_d, dist_d='vonmises'):
    '''
    normal or uniformly distributed
    '''
    if dist_d == 'vonmises':
        zenith = np.random.vonmises(0, dev_d, n)
    elif dist_d == 'uniform':
        zenith = np.random.rand(n) * dev_d
    elif dist_d == 'normal':
        zenith = np.random.normal(0, dev_d, n)
    else:
        raise NotImplementedError

    azimuth = np.random.rand(n) * np.pi * 2
    
    z = np.cos(zenith)
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    
    return np.stack([x,y,z], axis=-1)
    


class brdf_interp_gpu_2d(object):
    def __init__(self, brdf_slice, log_scale=False, epsilon=1e-10):
        '''
        brdf_slice is a 2d slice, [theta_h, theta_d, 3], 90X90X3
        '''
        self.img = brdf_slice.unsqueeze(0).permute(0,3,1,2) #[n_batches, color_channels, h, w]
        if log_scale:
            self.img = torch.log(torch.max(self.img, immutable_tensor([epsilon], dtype=self.img.dtype).cuda()))
        self.h = self.img.shape[-2]
        self.w = self.img.shape[-1]
        
    def __call__(self, cos_theta_h, cos_theta_d):
        '''
        should make sure cos values are in range[0,1]
        '''
        # coords [n_batches, n_queries, 2]
        theta_h = torch.acos(torch.clamp(cos_theta_h, 0, 1)) * 180.0 /  np.pi
        theta_d = torch.acos(torch.clamp(cos_theta_d, 0, 1)) * 180.0 /  np.pi
        theta_h = torch.clamp(theta_h, 0, 89)
        theta_d = torch.clamp(theta_d, 0, 89)
        coords = torch.stack([theta_h, theta_d], dim=-1).unsqueeze(0) # [1, n_queries, 2]
        normalise_scale = immutable_tensor([(self.h-1)*0.5, (self.w-1)*0.5], dtype=coords.dtype).cuda()
        coords = coords / normalise_scale - 1
        coords = torch.flip(coords, dims=(-1,))
        ret = torch.nn.functional.grid_sample(self.img, coords.unsqueeze(2)).squeeze(-1).transpose(1,2) # [1, n_queries, 3]
        
        ret = ret.squeeze(0)
        return ret

class brdf_interp_2d(object):
    def __init__(self, brdf_slice, log_scale=False, epsilon=1e-10):
        self.gpu = brdf_interp_gpu_2d(immutable_tensor(brdf_slice).cuda(), log_scale, epsilon)
    def __call__(self, cos_theta_h, cos_theta_d):
        return self.gpu(immutable_tensor(cos_theta_h).cuda(), immutable_tensor(cos_theta_d).cuda()).cpu().numpy()
    def save(self, file):
        np.save(file, self.gpu.img.cpu().numpy())
    @staticmethod
    def load(file):
        slice_ = np.load(file)
        slice_ = np.swapaxes(np.swapaxes(slice_[0], 0,1), 1,2)
        return brdf_interp_2d(slice_)


def load_brdf_2d(material_str, brdf_param=None, log=False, epsilon=1e-10):
    # either brdf_param ([cos_thetas], [rgbs]), or load material_str
    if brdf_param is None:
        with h5py.File('BRDF_2d/{}.h5'.format(material_str), 'r') as f:
            brdf_param = f['BRDF'][:,:,:,90] # [3, theta_h, theta_d], at phi_d = 0
            brdf_param = np.swapaxes(np.swapaxes(brdf_param, 0,1), 1, 2) # [theta_h, theta_d,3]      
    return brdf_interp_2d(brdf_param, log_scale=log, epsilon=epsilon)

def convert_brdf_interp_cpu_to_gpu_2d(brdf_cpu):
    return brdf_cpu.gpu

def make_D2d():
    brdf_strs =  [s[8:-3] for s in sorted(glob('BRDF_2d/*.h5'))]
    params = []
    for material_str in brdf_strs:
        with h5py.File('BRDF_2d/{}.h5'.format(material_str), 'r') as f:
            brdf_param = f['BRDF'][:,:,:,90] # [3, theta_h, theta_d], at phi_d = 0
            brdf_param = np.swapaxes(np.swapaxes(brdf_param, 0,1), 1, 2) # [theta_h, theta_d,3]
            params.append(brdf_param)
    np.save('D2d.npy', np.stack(params, axis=2))


def cos_attenuation_map(K, width, height, mu=1.0, principal_axis=None, epsilon=1e-10):
    # K is in openGL convention
    # principal axis is in camera coorinates, openGL convention, default [0,0,-1]
    w, h = np.meshgrid(range(width), range(height))
    img_coords = np.stack([h,w,np.ones_like(w)], axis=-1).reshape((-1,3))
    camera_coords = img_coords.dot(np.linalg.inv(K[:3,:3]).T) # [height*width, 3]
    ray_camera_coords_xyz = (-camera_coords / camera_coords[:,2:]) # [height*width, 3]
    ray_camera_coords_xyz = normalise(ray_camera_coords_xyz)
    if principal_axis is None:
        principal_axis = np.array([0,0,-1], dtype=np.float64)
    principal_axis = normalise(principal_axis)

    cos_light = (ray_camera_coords_xyz * principal_axis).sum(-1) # [height*width]
    cos_light = np.clip(cos_light, epsilon, 1)
    return np.power(cos_light, mu).reshape([height, width])

