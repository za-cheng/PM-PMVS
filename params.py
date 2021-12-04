
DELTA = 0.1 # delta value for computing huber loss
N = -3 # number of input views selected by the min-sum in Eq.8, i.e. value of \mathcal{M}, if negative, number of views discarded by min-sum 


exp_name = 'steel-bunny' # name of the experiment, used when saving intermediate results
input_file = 'data/steel-bunny.npy' # path to input file that contains imgs and camera parameters
min_dist = 10  # min depth (distance of object from reference view point) in world unit
max_dist = 200 # max depth (distance of object from reference view point) in world unit
# min_dist and max_dist are used for limiting the range from which depth variables can be searched
# they can be very loose but object must be confined in range (min_dist, max_dist)
# E.g. in the bunny example, we set min_dist=10 and max_dist=200, when object's size is actually 10 and about 40 units away from reference camera

kernel_connectivity=20 # how many neighbors are used for patchmatch's propagation step, must be one of {4,8,20,32}
lambda_c=5e-3 # weight for brdf regularization term
no_outter_iterations=50 # how many total iterations 
no_inner_iterations=10 # how many iterations per "fix brdf solve shape" subproblem
no_qpm_iterations=10 # how many iterations per QPM
no_patchmatch_iterations=10 # how many iterations per PatchMatch
theta_d_dev = 10 * 3.1415926 / 180 # angular std for PatchMatch's random seach step, in radians
len_dev = 1 # depth std for PatchMatch's random seach step, in world unit (it's recommended to set this in same order of magnitude as object's size)
inner_attenuation=1 # attenuation rate for PatchMatch, leave it to 1 unless there's specific reason to change it
no_candidates=10 # number of random candidates for PatchMatch's random seach step, use greater values for increased accuracy but slower run time
lambda_n = 50 # weight for normal-depth consistency, i.e. lambda_s in paper
lambda_z_init = 5 # initial weight for quadratic penalty, i.e. \sigma^{(0)} in Eq.12 
lambda_l = 0 # weight for depth laplacian, may change to non-zero value to encourage surface smoothness, not used in paper
lambda_z_increase_rate_inner = 2 # ratio by which QPM penalty increases, i.e. \kappa in Eq.12 
epsilon=1e-10 # small constant for handling zero-related arithmetics, e.g. log(0) etc.



#### you may play with above parameters to control trade-off between performance, robustness and efficiency
#### E.g. following setting leads to 10 times faster optimisation and results are probably be just as good
no_outter_iterations = 30
kernel_connectivity = 8
no_inner_iterations = 15
no_qpm_iterations = 1
lambda_z_increase_rate_inner = 1.3
