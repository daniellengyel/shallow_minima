import autograd.numpy as np

def d_gaussian(x, y, sig):
    d_abs = -1 if x < y else 1
    return - np.abs(x - y) / sig * d_abs / np.sqrt(2 * np.pi * sig) * np.exp(- np.abs(x - y) ** 2 / (2 * sig))


def multi_gaussian(cov):
    def multi_guassian_helper(x, y):
        k = x.shape[0]
        return 1 / np.sqrt(pow(2 * np.pi, k) * np.linalg.det(cov)) * np.exp(
            -0.5 * (x - y).T.dot(np.linalg.inv(cov).dot(x - y)))

    return multi_guassian_helper


def grad_gaussian(cov):
    def grad_gaussian_helper(x, y):
        mg = multi_gaussian(cov)(x, y)
        grad_term = - np.linalg.inv(cov).dot(x - y)
        return mg * grad_term

    return grad_gaussian_helper