import numpy as np


#define potential for second proccess
def U_second(U, k, kernel, particles):
    def U_second_helper(x):
        return U(x) + k*V(x, kernel, particles)
    return U_second_helper


def grad_U_second(grad_U, k, grad_kernel, particles):
    return U_second(grad_U, k, grad_kernel, particles)


# Approximating density with the particles
def V(x, K, particles):
    N = len(particles)

    ret_sum = 0
    for p in particles:
        ret_sum += K(x, p)
    return 1 / float(N) * ret_sum


def grad_V(x, grad_K, particles):
    return V(x, grad_K, particles)

def particles_converged(p_paths, epsilon):
    for p in p_paths:
        if not ((len(p) > 2) and (np.linalg.norm(p[-1] - p[-2]) < epsilon)):
            return False
    return True

def hyper_cube_enforcer(upper_bound=32.768, lower_bound=-32.768):
    def helper(x):
        filter_lower = x < lower_bound
        x[filter_lower] = lower_bound +1

        filter_upper = x > upper_bound
        x[filter_upper] = upper_bound - 1
        return x, any(filter_lower) or any(filter_upper)
    return helper

def percent_endpoint(x_star, end_points, epsilon):
    num_reached = 0
    for p in end_points:
        if np.linalg.norm(x_star - p) < epsilon:
            num_reached += 1
    return num_reached * 1.0 / len(end_points)


def filter_to_goal(x_star, epsilon, analytics):
    filter_distance = np.array([np.linalg.norm(p - x_star) < epsilon for p in analytics["end_point"].values
                                ])

    return filter_distance


if __name__ == "__main__":
    from Kernels import *

    x = np.array([0])
    y = np.array([0, 1, 4, 65, 123, 65])
    cov = np.eye(y.shape[0])

    cov = np.eye(1)
    grad_k = grad_gaussian(cov)

    V(x, grad_k, [x])