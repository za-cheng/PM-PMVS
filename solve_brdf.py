import scipy
from scipy.sparse import hstack, vstack, eye
from sklearn.linear_model import Lasso
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from params import *

def solve_huber_loss(D, y, delta, lambda_=0, p=2, solver='L-BFGS-B', positive=False, offset=True):
    # minimise E(x)=1/m * Huber(Dx-y)+lambda*|x|_p^ p w.r.t x, returns the minimum x
    # D [m,n]
    # y [m]
    # returns: x [n], and min energy value
    if p != 1 and p != 2: raise NotImplementedError
    
    def _eval_huber(x):
        raw_diff = D.dot(x) - y
        #############
        # raw_diff[y>np.log(MAX_PIXEL_INTENSITY-1)] = np.minimum(raw_diff[y>np.log(MAX_PIXEL_INTENSITY-1)], 0)
        #############
        diff = np.abs(raw_diff)
        d = np.minimum(diff, delta)
        if offset:
            x_ = x.copy()
            x_[-1] = 0 # remove the offset
        else:
            x_ = x
        loss = (d*(diff - 0.5*d) / len(D)).sum() * 1 + lambda_ * (np.abs(x_) ** p).sum()
        return loss 
    def _grad_huber(x):
        raw_diff = D.dot(x) - y
        #############
        # raw_diff[y>np.log(MAX_PIXEL_INTENSITY-1)] = np.minimum(raw_diff[y>np.log(MAX_PIXEL_INTENSITY-1)], 0)
        #############
        grads = np.clip(raw_diff, -delta, delta).dot(D) / len(D) * 1
        if offset:
            x_ = x.copy()
            x_[-1] = 0 # remove the offset
        else:
            x_ = x
        if p == 1:
            grads = grads + lambda_ * np.sign(x_)
        elif p == 2:
            grads = grads + lambda_ * 2 * x_
        return grads
    
    # initialise by least square solution w/o regularisation
    x_0, _ = solve_least_squared_with_squared_regularization(D, y, lambda_=lambda_, offset=offset)
    
    if offset:
        D = convert_to_homogeneous_coords(D)
    if positive:
        bounds = [(0, np.inf)] * len(x_0)
        if offset:
            bounds[-1] = (-np.inf, np.inf)
    else:
        bounds = None
    r = scipy.optimize.minimize(_eval_huber, x_0, method=solver, jac=_grad_huber, bounds=bounds)
    # if r.success: print 'success!'
    # else: print 'failed!'
    # print r.message
    # print 'gradients at converge: {}'.format( _grad_huber(r.x))
    return r.x, _eval_huber(r.x)
    


def solve_least_absolute_with_sparsity(D, y, lambda_=0, solver='gurobi', positive=True, offset=False):
    # minimise E(x)=1/m * |Dx-y|+lambda*|x| w.r.t x, returns the minimum x
    # D [m,n]
    # y [m]
    # returns: x [n], and min energy value
    m, n = D.shape
    
    
    # minimise |Dx-y|
    if solver == 'scipy':
        if offset or not positive:
            raise NotImplementedError
        D = np.concatenate([D,lambda_*np.eye(n)], axis=0) # [m+n, n]
        y = np.concatenate([y, np.zeros(n)], axis=0) # [m+n]
        
        c = np.concatenate([np.zeros(n), np.ones(m+n)], axis=0) # [m+2n]
        A_ub_upper = hstack([D, -eye(m+n)]) # [m+n, m+2n]
        b_ub_upper = y #[m+n]
        A_ub_lower = hstack([-D, -eye(m+n)]) # [m+n, m+2n]
        b_ub_lower = -y #[m+n]

        A_ub = vstack([A_ub_upper, A_ub_lower]) #[2m+2n, m+2n]
        b_ub = np.concatenate([b_ub_upper, b_ub_lower], axis=0) #[2m+2n]

        rst = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub)
        if rst.success:
            print("optimisation success")
        else:
            print("optimisation failed")
        x = rst.x[:n]
        min_E = rst.x[n:].sum()
    elif solver == 'gurobi':
        D = D / m
        y = y / m
        gm = gp.Model('lp')
        gm.setParam('OutputFlag', 0)
        # variables
        
        if positive:
            x = [gm.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY) for _ in range(n)] # [n]
        else:
            x = gm.addVars(n) # [n]
        
        if offset:
            offset_var = gm.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        else:
            offset_var = 0
            
        u = gm.addVars(2*n) # [2n]
        gm.update()
        
        x_np = np.array([x[i] for i in range(n)])
        E1 = np.dot(D, x_np) - y + offset_var # [n]
        
        for i in range(n):
            gm.addConstr(lhs=E1[i], sense=gp.GRB.LESS_EQUAL, rhs=u[i])
            gm.addConstr(lhs=-E1[i], sense=gp.GRB.LESS_EQUAL, rhs=u[i])
            gm.addConstr(lhs=lambda_*x[i], sense=gp.GRB.LESS_EQUAL, rhs=u[i+n])
            gm.addConstr(lhs=-lambda_*x[i], sense=gp.GRB.LESS_EQUAL, rhs=u[i+n])
        
        gm.setObjective(gp.quicksum(u))
        gm.optimize()
        x = gm.getAttr('x', x)
        x = np.array([x[i] for i in range(n)])
        u = gm.getAttr('x', u)
        u = np.array([u[i] for i in range(2*n)])
        
        if offset:
            x = np.concatenate([x, [m*offset_var.x]], axis=0)
            
        min_E = np.sum(u)
        
    
    return x, min_E

def solve_least_squared_with_sparsity(D, y, lambda_=0, solver='coord_descent', positive=True):
    # minimise lasso regression E(x)=1/m * |Dx-y|^2+lambda*|x| w.r.t x, returns the minimum x and min_E(x)
    # D [m,n]
    # y [m]
    # returns: x [n], and min energy value
    m = len(y)
    if solver == 'coord_descent':
        l = Lasso(alpha=lambda_, positive=positive)
        l.fit(D/np.sqrt(m), y/np.sqrt(m))
        x = l.coef_
        min_E = ((D.dot(x)-y)**2).mean()  + lambda_ * np.abs(x).sum()
    
    else:
        raise NotImplementedError
        
    return x, min_E

def solve_least_squared_with_squared_regularization(D, y, lambda_=0, offset=False):
    # minimise ridge regression E(x)=1/m * |Dx-y|^2+lambda*|x|^2 w.r.t x, returns the minimum x and min_E(x)
    # D [m,n]
    # y [m]
    # returns: x [n], and min energy value
    m = len(y)
    if offset:
        D = convert_to_homogeneous_coords(D) # [m, n+1], last column are all 1's
        I = np.diag([1]*(D.shape[1]-1) + [0]) # diagonal, all except bottom right elements are 1's, bottom-left is 0
    else:
        I = np.eye(D.shape[1])
    
    D = D / np.sqrt(m)
    y = y / np.sqrt(m)
    
    x = np.linalg.inv(D.T.dot(D) + np.eye(D.shape[1]) * lambda_).dot(D.T).dot(y)
    min_E = np.sum((D.dot(x)-y)**2) + np.sum(lambda_* (I.dot(x))**2 )
    return x, min_E

def solve_brdf_fix_shape(D, mu, point_coords, point_normals, view_coords, lhs_intensities, lambda_=0, order='L1', solve_only_gamma=False, epsilon=1e-10, select_mask=None):
    # D [90, no_materials*3] log scale
    # point_coords [no_points, 3]
    # point_normals [no_points, 3]
    # view_coords [no_views, 3]
    # lhs_intensities [no_views, no_points, 3], log-scale raw image intensities
    # returns estimated BRDF and the points used for estimation

    D = D.reshape([90, -1]) # make sure its 2-dimensional
    mu = mu.reshape([90, -1])
    # ref_brdf = np.median(D, axis=-1, keepdims=True) # [90, 1]
    view_vectors = view_coords.reshape([-1,1,3]) - point_coords.reshape([1,-1,3]) #[no_views, no_points, 3]
    view_angles = normalise(view_vectors) #[no_views, no_points, 3]
    view_distance = np.linalg.norm(view_vectors, axis=-1,keepdims=True) #[no_views, no_points, 1]
    lhs_intensities = lhs_intensities + np.log(np.maximum(view_distance, epsilon)) * 2
    # D = D - mu
    D_func = scipy.interpolate.interp1d(np.arange(90), D, axis=0)
    ref_func = scipy.interpolate.interp1d(np.arange(90), mu, axis=0)
    
    # optimise over c
    xs = (view_angles * point_normals).sum(axis=-1).flatten() # [no_views * no_points]
    xs = np.clip(xs, -1+epsilon, 1-epsilon)
    ys = lhs_intensities.reshape((-1,3)) # [no_views * no_points, 3]
    delta = np.zeros(view_vectors.shape[:2]) + DELTA # [no_views, no_points]
    #delta[0,:] = np.inf
    delta = delta.reshape([-1])


    if select_mask is not None:
        mask__ = select_mask.flatten()
        xs = xs[mask__]
        ys = ys[mask__]
        delta = delta[mask__]

    xs = xs[ys.sum(-1) < np.inf] # this is to handle nan values yielded by grid_samples() in torch, which appears to be a bug
    delta = delta[ys.sum(-1) < np.inf]
    ys = ys[ys.sum(-1) < np.inf]

    xs_ret = xs
    ys_ret = ys
    # ys = ys[xs<np.cos(10.0/180*np.pi)] # [N, 3]
    # delta = delta[xs<np.cos(10.0/180*np.pi)] # [N]
    # xs = xs[xs<np.cos(10.0/180*np.pi)] # [N]

    ys = ys[xs>np.cos(80*np.pi/180)] # [N, 3]
    delta = delta[xs>np.cos(80*np.pi/180)] # [N]
    xs = xs[xs>np.cos(80*np.pi/180)] # [N]


#     ys=ys[ np.logical_and(np.arccos(xs)*180/np.pi < 60, np.arccos(xs)*180/np.pi > 0)]
#     xs=xs[ np.logical_and(np.arccos(xs)*180/np.pi < 60, np.arccos(xs)*180/np.pi > 0)]
    
    xs = np.clip(np.arccos(xs) * 180 / np.pi, a_min=0, a_max=89) # [N]
    #ys = ys - np.log(np.cos(xs * np.pi / 180))[...,None]
    


    Dxs = D_func(xs) # [N, no_materials*3]
    refxs = ref_func(xs) # [N, 1]
    ys = ys - refxs
    if order == 'L1':
        if solve_only_gamma:
            raise NotImplementedError
        else:
            cs, errs = zip(*[solve_least_absolute_with_sparsity(Dxs, ys[:,0], lambda_, positive=True, offset=True), \
                            solve_least_absolute_with_sparsity(Dxs, ys[:,1], lambda_, positive=True, offset=True), \
                            solve_least_absolute_with_sparsity(Dxs, ys[:,2], lambda_, positive=True, offset=True) ])
    elif order == 'L2':
        if solve_only_gamma:
            raise NotImplementedError
        else:
            cs, errs = zip(*[solve_least_squared_with_squared_regularization(Dxs, ys[:,0], lambda_, offset=True), \
                         solve_least_squared_with_squared_regularization(Dxs, ys[:,1], lambda_, offset=True), \
                         solve_least_squared_with_squared_regularization(Dxs, ys[:,2], lambda_, offset=True) ])
    elif order == 'squared':
        if solve_only_gamma:
            raise NotImplementedError
        else:
            cs, errs = zip(*[solve_least_squared_with_sparsity(Dxs, ys[:,0], lambda_), \
                         solve_least_squared_with_sparsity(Dxs, ys[:,1], lambda_), \
                         solve_least_squared_with_sparsity(Dxs, ys[:,2], lambda_) ])
    elif order == 'huber-1':
        if solve_only_gamma:
            raise NotImplementedError
        else:
            cs, errs = zip(*[solve_huber_loss(Dxs, ys[:,0], delta=delta, lambda_=lambda_, p=1, solver='L-BFGS-B', positive=False, offset=True), \
                         solve_huber_loss(Dxs, ys[:,1], delta=delta, lambda_=lambda_, p=1, solver='L-BFGS-B', positive=False, offset=True), \
                         solve_huber_loss(Dxs, ys[:,2], delta=delta, lambda_=lambda_, p=1, solver='L-BFGS-B', positive=False, offset=True) ])
    elif order == 'huber-2':
        if solve_only_gamma:
            if lambda_ <= 0:
                raise NotImplementedError
            else:
                cs, errs = zip(*[solve_huber_loss(Dxs*0, ys[:,0], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True), \
                            solve_huber_loss(Dxs*0, ys[:,1], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True), \
                            solve_huber_loss(Dxs*0, ys[:,2], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True) ])
        else:
            cs, errs = zip(*[solve_huber_loss(Dxs, ys[:,0], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True), \
                         solve_huber_loss(Dxs, ys[:,1], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True), \
                         solve_huber_loss(Dxs, ys[:,2], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True) ])
    else:
        raise NotImplementedError

    c = np.stack(cs, axis=-1)# [no_materials*3, 3]

    brdf_param = np.cos(np.arange(90)*np.pi/180), D.dot(c[:-1]) + mu + c[-1] # if offset=True
    brdf_func = load_brdf(None, brdf_param, log=False, epsilon=epsilon)
    
    return brdf_func, xs_ret.copy(), ys_ret.copy(), errs

def solve_brdf_fix_shape_2d(D, mu, point_coords, point_normals, view_coords, lhs_intensities, lambda_=0, order='L1', solve_only_gamma=False, epsilon=1e-10, select_mask=None):
    # D [90, 90, no_materials*3] log scale
    # mu [90, 90, 3]
    # point_coords [no_points, 3]
    # point_normals [no_points, 3]
    # view_coords [no_views, 3, 2] last dimension [camera, light]
    # lhs_intensities [no_views, no_points, 3], log-scale raw image intensities
    # returns estimated BRDF and the points used for estimation

    D = D.reshape([90, 90, -1]) # make sure its 2-dimensional
    mu = mu.reshape([90, 90, -1])
    # ref_brdf = np.median(D, axis=-1, keepdims=True) # [90, 1]
    light_coords = view_coords[...,1]
    view_coords = view_coords[...,0]

    view_vectors = view_coords.reshape([-1,1,3]) - point_coords.reshape([1,-1,3]) #[no_views, no_points, 3]
    view_angles = normalise(view_vectors) #[no_views, no_points, 3]
    view_distance = np.linalg.norm(view_vectors, axis=-1,keepdims=True) #[no_views, no_points, 1]


    light_vectors = light_coords.reshape([-1,1,3]) - point_coords.reshape([1,-1,3]) #[no_views, no_points, 3]
    light_angles = normalise(light_vectors) #[no_views, no_points, 3]
    light_distance = np.linalg.norm(light_vectors, axis=-1,keepdims=True) #[no_views, no_points, 1]

    lhs_intensities = lhs_intensities + np.log(np.maximum(light_distance, epsilon)) * 2

    D_func = brdf_interp_2d(D)
    ref_func = brdf_interp_2d(mu)
    
    # optimise over c
    hs = normalise(view_angles + light_angles) #[no_views, no_points, 3]
    x_cos_hs = (hs * point_normals).sum(axis=-1).flatten() # [no_views * no_points]
    x_cos_ds = (hs * light_angles).sum(axis=-1).flatten() # [no_views * no_points]
    x_cos_falloff = np.clip((light_angles * point_normals).sum(axis=-1).flatten(), 1e-6, 1) # [no_views * no_points]
    
    ys = lhs_intensities.reshape((-1,3)) # [no_views * no_points, 3]
    ys = ys - np.log(x_cos_falloff)[...,None] # [no_views * no_points, 3]
    delta = np.zeros(view_vectors.shape[:2]) + DELTA # [no_views, no_points]
    delta[0,:] = np.inf
    delta = delta.reshape([-1]) # [no_views * no_points]


    mask__ = (light_angles * point_normals).sum(axis=-1).flatten() > 0.01 # remove adhere shadow points
    if select_mask is not None:
        mask__ = np.logical_and(mask__, select_mask.flatten())
    x_cos_hs = x_cos_hs[mask__]
    x_cos_ds = x_cos_ds[mask__]
    ys = ys[mask__]
    delta = delta[mask__]

    x_cos_hs = x_cos_hs[ys.sum(-1) < np.inf] # this is to handle nan values yielded by grid_samples() in torch, which appears to be a bug
    x_cos_ds = x_cos_ds[ys.sum(-1) < np.inf]
    delta = delta[ys.sum(-1) < np.inf]
    ys = ys[ys.sum(-1) < np.inf]

    ys = ys[x_cos_hs>0] # [N, 3]
    delta = delta[x_cos_hs>0] # [N]
    x_cos_ds = x_cos_ds[x_cos_hs>0] # [N]
    x_cos_hs = x_cos_hs[x_cos_hs>0] # [N]

    ys = ys[x_cos_ds>0] # [N, 3]
    delta = delta[x_cos_ds>0] # [N]
    x_cos_hs = x_cos_hs[x_cos_ds>0] # [N]
    x_cos_ds = x_cos_ds[x_cos_ds>0] # [N]



    xs_ret = (x_cos_hs.copy(), x_cos_ds.copy())
    ys_ret = ys.copy()
    
#     ys=ys[ np.logical_and(np.arccos(xs)*180/np.pi < 60, np.arccos(xs)*180/np.pi > 0)]
#     xs=xs[ np.logical_and(np.arccos(xs)*180/np.pi < 60, np.arccos(xs)*180/np.pi > 0)]
    


    Dxs = D_func(x_cos_hs, x_cos_ds) # [N, no_materials*3]
    refxs = ref_func(x_cos_hs, x_cos_ds) # [N, 1]
    ys = ys - refxs
    if order == 'L1':
        if solve_only_gamma:
            raise NotImplementedError
        else:
            cs, errs = zip(*[solve_least_absolute_with_sparsity(Dxs, ys[:,0], lambda_, positive=True, offset=True), \
                            solve_least_absolute_with_sparsity(Dxs, ys[:,1], lambda_, positive=True, offset=True), \
                            solve_least_absolute_with_sparsity(Dxs, ys[:,2], lambda_, positive=True, offset=True) ])
    elif order == 'L2':
        if solve_only_gamma:
            raise NotImplementedError
        else:
            cs, errs = zip(*[solve_least_squared_with_squared_regularization(Dxs, ys[:,0], lambda_, offset=True), \
                         solve_least_squared_with_squared_regularization(Dxs, ys[:,1], lambda_, offset=True), \
                         solve_least_squared_with_squared_regularization(Dxs, ys[:,2], lambda_, offset=True) ])
    elif order == 'squared':
        if solve_only_gamma:
            raise NotImplementedError
        else:
            cs, errs = zip(*[solve_least_squared_with_sparsity(Dxs, ys[:,0], lambda_), \
                         solve_least_squared_with_sparsity(Dxs, ys[:,1], lambda_), \
                         solve_least_squared_with_sparsity(Dxs, ys[:,2], lambda_) ])
    elif order == 'huber-1':
        if solve_only_gamma:
            raise NotImplementedError
        else:
            cs, errs = zip(*[solve_huber_loss(Dxs, ys[:,0], delta=delta, lambda_=lambda_, p=1, solver='L-BFGS-B', positive=False, offset=True), \
                         solve_huber_loss(Dxs, ys[:,1], delta=delta, lambda_=lambda_, p=1, solver='L-BFGS-B', positive=False, offset=True), \
                         solve_huber_loss(Dxs, ys[:,2], delta=delta, lambda_=lambda_, p=1, solver='L-BFGS-B', positive=False, offset=True) ])
    elif order == 'huber-2':
        if solve_only_gamma:
            if lambda_ <= 0:
                raise NotImplementedError
            else:
                cs, errs = zip(*[solve_huber_loss(Dxs*0, ys[:,0], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True), \
                            solve_huber_loss(Dxs*0, ys[:,1], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True), \
                            solve_huber_loss(Dxs*0, ys[:,2], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True) ])
        else:
            cs, errs = zip(*[solve_huber_loss(Dxs, ys[:,0], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True), \
                         solve_huber_loss(Dxs, ys[:,1], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True), \
                         solve_huber_loss(Dxs, ys[:,2], delta=delta, lambda_=lambda_, p=2, solver='L-BFGS-B', positive=False, offset=True) ])
    else:
        raise NotImplementedError

    c = np.stack(cs, axis=-1)# [no_materials*3, 3]

    brdf_param = D.reshape([8100, -1]).dot(c[:-1]).reshape([90,90,-1]) + mu + c[-1] # if offset=True
    brdf_func = load_brdf_2d(None, brdf_param, log=False, epsilon=epsilon)
    
    return brdf_func, xs_ret, ys_ret, errs