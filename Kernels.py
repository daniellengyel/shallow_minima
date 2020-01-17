import autograd.numpy as np

def d_gaussian(x, y, sig):
    d_abs = -1 if x < y else 1
    return - np.abs(x - y) / sig * d_abs / np.sqrt(2 * np.pi * sig) * np.exp(- np.abs(x - y) ** 2 / (2 * sig))


def multi_gaussian(cov):
    def multi_gaussian_helper(inp, mu):
        """same mu for every datapoint given in ipn"""
        k = inp.shape[0]
        diff = (inp.T - mu).T
        return 1 / np.sqrt(pow(2 * np.pi, k) * np.linalg.det(cov)) * np.exp(
            -0.5 * np.sum(diff*(np.linalg.inv(cov).dot(diff)), axis=0))
    return multi_gaussian_helper

def multi_gaussian_unnormalized(cov):
    def multi_gaussian_helper(inp, mu):
        """same mu for every datapoint given in ipn"""
        k = inp.shape[0]
        diff = (inp.T - mu).T
        return np.exp(-0.5 * np.sum(diff*(np.linalg.inv(cov).dot(diff)), axis=0))
    return multi_gaussian_helper


def grad_multi_gaussian(cov):
    def grad_gaussian_helper(inp, mu):
        """respect to x"""
        k = inp.shape[0]
        diff = (inp.T - mu).T

        mg = multi_gaussian(cov)(inp, mu)
        grad_term = - np.linalg.inv(cov).dot(diff)
        return mg * grad_term

    return grad_gaussian_helper