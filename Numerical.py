import autograd.numpy as np
from Kernels import *
from utils import *
from Functions import *

from Functions import Gibbs, GradGibbs


def one_d_feynman_kac(process):
    u_start = process["u_start"]
    x_low, x_high, t_low, t_high = process["domain"] # closed interval for spatial and time
    delta_x, delta_t = process["delta"] # deltas for spatial and temporal dimensions

    # get potential_function and gradient
    U, grad_U = get_potential(process)

    # init u
    u = np.array([[u_start(x) for x in range(x_low, x_high)]]) # 1D is time and rest are space

    for t in range(t_low, t_high+1):
        u_t = []
        for x in range(x_low, x_high+1):

            # compute the update step
            
