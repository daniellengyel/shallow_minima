import autograd.numpy as np
from Functions import *

# ------
# Process util stuff
def get_potential(process):
    # get potential_function and gradient
    if process["potential_function"]["name"] == "gaussian":
        potential_params = process["potential_function"]["params"]
        U = gaussian_sum(potential_params)
        grad_U = grad_gaussian_sum(potential_params)
    elif process["potential_function"]["name"] == "Ackley":
        U = AckleyProblem
        grad_U = GradAckleyProblem
    elif process["potential_function"]["name"] == "2d_gaussian_symmetric":
        potential_params = process["potential_function"]["params"]
        one_d_U = gaussian_sum(potential_params)
        one_d_grad_U = grad_gaussian_sum(potential_params)
        U = lambda inp: one_d_U(np.array([inp[0]])) + one_d_U(np.array([inp[1]]))
        grad_U = lambda inp: np.array([one_d_grad_U(np.array([inp[0]]))[0],  one_d_grad_U(np.array([inp[1]]))[0]])
    else:
        raise ValueError("Does not support given function {}".format(process["potential_function"]["name"]))
    return U, grad_U


def get_particles(process):
    # get start_pos
    if process["particle_init"]["name"] == "1d_uniform":
        num_particles = process["particle_init"]["num_particles"]
        particles = [[np.random.uniform(process["x_range"][0], process["x_range"][1])] for _ in range(num_particles)]
    elif process["particle_init"]["name"] == "2d_uniform": # for now same range for all dimensions
        num_particles = process["particle_init"]["num_particles"]
        x_low, x_high = process["x_range"]
        particles = [[np.random.uniform(x_low, x_high), np.random.uniform(x_low, x_high)] for _ in range(num_particles)]
    else:
        raise ValueError("Does not support given function {}".format(process["particle_init"]["name"]))
    return particles

def get_resample_function(process):
    if process["resample_function"]["name"] == "softmax":
        resample_beta = process["resample_function"]["params"]["beta"]
        p_resample_func = lambda w, end_p: resample_positions_softmax(w, end_p, beta=resample_beta)
    elif process["resample_function"]["name"] == "none":
        p_resample_func = lambda w, end_p: end_p
    else:
        raise ValueError("Does not support given function {}".format(process["resample_function"]["name"]))
    return p_resample_func

def get_weight_function(process):
    if process["weight_function"]["name"] == "norm":
        p_weight_func = lambda U, grad_U, x, curr_weights: weight_function_discounted_norm(U, grad_U, x, curr_weights, 1)
    elif process["weight_function"]["name"] == "discounted_norm":
        weight_gamma = process["weight_function"]["params"]["gamma"]
        p_weight_func = lambda U, grad_U, x, curr_weights: weight_function_discounted_norm(U, grad_U, x, curr_weights,
                                                                                             weight_gamma)
    elif process["weight_function"]["name"] == "partial_norm":
        partials = process["weight_function"]["params"]["partials"]
        p_weight_func = lambda U, grad_U, x, curr_weights: weight_function_discounted_norm(U, grad_U, x, curr_weights,
                                                                                           partials=partials)
    else:
        raise ValueError("Does not support given function {}".format(process["weight_function"]["name"]))
    return p_weight_func

# -------
# diffusion stuff
def resample_positions_softmax(weights, positions, beta=1):
    probabilities = softmax(weights, beta)
    pos_filter = np.random.choice(list(range(len(positions))), len(positions), p=probabilities)
    return np.array(positions)[np.array(pos_filter)]

def softmax(weights, beta=1):
    sum_exp_weights = sum([np.exp(beta*w) for w in weights])
    probabilities = np.array([np.exp(beta*w) for w in weights]) / sum_exp_weights
    return probabilities

def weight_function_discounted_norm(U, grad_U, x, curr_weights, gamma=1, partials=None):
    grad = grad_U(x.T)
    if partials is not None:
        grad = grad[partials]
    return gamma * curr_weights + np.linalg.norm(grad, axis=0)


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

def hyper_cube_enforcer(lower_bound=-32.768, upper_bound=32.768, reflective_strength=1):
    def helper(x):
        filter_lower = x < lower_bound
        x[filter_lower] = lower_bound + reflective_strength

        filter_upper = x > upper_bound
        x[filter_upper] = upper_bound - reflective_strength
        return x, np.any(filter_lower) or np.any(filter_upper)
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

def resample_positions_softmax(weights, positions, beta=1):
    probabilities = softmax(weights, beta)
    pos_filter = np.random.choice(list(range(len(positions))), len(positions), p=probabilities)
    return np.array(positions)[np.array(pos_filter)]

def softmax(weights, beta=1):
    sum_exp_weights = sum([np.exp(beta*w) for w in weights])
    probabilities = np.array([np.exp(beta*w) for w in weights]) / sum_exp_weights
    return probabilities


if __name__ == "__main__":
    from Kernels import *

    x = np.array([0])
    y = np.array([0, 1, 4, 65, 123, 65])
    cov = np.eye(y.shape[0])

    cov = np.eye(1)
    grad_k = grad_gaussian(cov)

    V(x, grad_k, [x])