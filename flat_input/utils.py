import numpy as np


def sample_index_softmax(weights, positions, beta=1):
    probabilities = softmax(weights, beta)
    pos_filter = np.random.choice(list(range(len(positions))), len(positions), p=probabilities)
    return pos_filter


def softmax(weights, beta=-1):
    # normalize weights:
    weights /= np.sum(weights)

    sum_exp_weights = sum([np.exp(beta * w) for w in weights])
    probabilities = np.array([np.exp(beta * w) for w in weights]) / sum_exp_weights
    return probabilities


def weight_function(grad, curr_weight):
    input_shape = grad.shape  # batch, filters, x_dim, y_dim
    grad = grad.reshape((input_shape[0], np.product(input_shape[1:]))).T

    return curr_weight + np.sum(np.linalg.norm(grad, axis=0))