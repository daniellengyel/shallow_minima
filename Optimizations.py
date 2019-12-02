import numpy as np
from Kernels import *
from utils import *
from Functions import Gibbs, GradGibbs


def grad_descent(func, grad_func, x_curr, eps, gamma, start_t=0, end_t=float("inf"), verbose=False, domain_enforcer=None):
    x_curr = np.array(x_curr, dtype=np.float)
    if domain_enforcer is not None:
        x_curr, went_outside_domain = domain_enforcer(x_curr)
    path = [x_curr]
    t = start_t

    went_outside_domain = False
    while t < end_t:
        x_next = x_curr - gamma(t) * grad_func(x_curr)
        if domain_enforcer is not None:
            x_next, went_outside_domain = domain_enforcer(x_next)

        path.append(x_next)

        if (np.linalg.norm(x_next - x_curr)) < eps and not went_outside_domain:  # TODO check what happens with more samples
            if verbose:
                print(grad_func(x_curr))
            break
        if (t % 50) == 0 and verbose:
            print("Iteration", t)
            print("diff", np.abs(func(x_next) - func(x_curr)))
        x_curr = x_next
        t += 1
    return np.array(path)


def simulated_annealing(func, grad_func, x_curr, eps, gamma, temperature, start_t=0, end_t=float("inf"), verbose=False, domain_enforcer=None):
    x_curr = np.array(x_curr, dtype=np.float)
    if domain_enforcer is not None:
        x_curr, went_outside_domain = domain_enforcer(x_curr)
    path = [x_curr]
    t = start_t

    went_outside_domain = False
    while t < end_t:
        x_next = x_curr + gamma(t) * (
                    -grad_func(x_curr)) + temperature(t) * np.array([[np.random.normal()] for _ in range(x_curr.shape[0])]).reshape(x_curr.shape)
        if domain_enforcer is not None:
            x_next, went_outside_domain = domain_enforcer(x_next)

        path.append(x_next)

        if np.linalg.norm(x_next) < eps: #(eps != 0) and (np.linalg.norm(x_next - x_curr) < eps) and (not went_outside_domain):  # TODO check what happens with more samples
            if verbose:
                print(grad_func(x_curr))
            break

        if (t % 50) == 0 and verbose:
            print("Iteration", t)
            print("diff", np.abs(func(x_next) - func(x_curr)))

        x_curr = x_next
        t += 1
    return np.array(path)


def interactive_diffusion(U, grad_U, first_process, second_process, total_iter, k, verbose=False, domain_enforcer=None):
    first_start = first_process["start"]
    first_gamma = first_process["gamma"]
    first_temperature = first_process["temperature"]
    first_num_particles = len(first_start)
    first_num_iter = first_process["num_iter"]
    first_epsilon = first_process["epsilon"]

    second_start = second_process["start"]
    second_gamma = second_process["gamma"]
    second_temperature = second_process["temperature"]
    second_num_particles = len(second_start)
    second_num_iter = second_process["num_iter"]  # TODO not used
    second_epsilon = second_process["epsilon"]

    dim = len(second_start[0])

    kernel = multi_gaussian(3*np.eye(dim))
    grad_kernel = grad_gaussian(3*np.eye(dim))

    # init num_particles
    first_paths = [np.array([p]) for p in first_start]

    # init second_particles
    second_paths = [np.array([p]) for p in second_start]

    for t in range(total_iter):
        # run n particles N_particles_iter times
        for i in range(first_num_particles):
            p = first_paths[i][-1]
            path = simulated_annealing(U, grad_U, p, first_epsilon, first_gamma, first_temperature,
                                             start_t=t * first_num_iter, end_t=(t + 1) * first_num_iter, domain_enforcer=domain_enforcer)
            first_paths[i] = np.concatenate((first_paths[i], path[1:]), axis=0)

        # update second diffusion proccess
        interacting_particles = [p[-1] for p in first_paths]
        U_second_use = U_second(U, k, kernel, interacting_particles)
        grad_U_second_use = grad_U_second(grad_U, k, grad_kernel, interacting_particles)
        for i in range(second_num_particles):
            p = second_paths[i][-1]
            if np.linalg.norm(p) < second_epsilon: #(len(second_paths[i]) > 2) and (np.linalg.norm(second_paths[i][-1] - second_paths[i][-2]) < second_epsilon):
                continue
            path = simulated_annealing(U_second_use, grad_U_second_use, p, second_epsilon, second_gamma,
                                             second_temperature, start_t=t, end_t=t + 1, domain_enforcer=domain_enforcer)
            second_paths[i] = np.concatenate((second_paths[i], path[1:]), axis=0)

        # if particles_converged(second_paths, second_epsilon):
        #     break

        if verbose:
            if t % 100 == 0:
                print("Iter: ", t)

    return np.array(first_paths), np.array(second_paths)


def interactive_diffusion_gibbs(U, grad_U, process, total_iter, k, sig, verbose=False, domain_enforcer=None):
    p_start = process["start"]
    p_gamma = process["gamma"]
    p_temperature = process["temperature"]
    p_num_particles = len(p_start)
    p_num_iter = process["num_iter"]
    p_epsilon = process["epsilon"]

    dim = len(p_start[0])

    # init num_particles
    p_paths = [np.array([p]) for p in p_start]

    U_second_use = U
    grad_U_second_use = lambda x: grad_U(x) - k*GradGibbs(x, U, grad_U, sig)

    for i in range(p_num_particles):
        p = p_paths[i][-1]
        path = simulated_annealing(U_second_use, grad_U_second_use, p, p_epsilon, p_gamma,
                                         p_temperature, start_t=0, end_t=total_iter, verbose=verbose, domain_enforcer=domain_enforcer)
        p_paths[i] = path


    return np.array(p_paths)




