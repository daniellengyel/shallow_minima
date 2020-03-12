import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torchvision
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from utils import *
from nets.Nets import SimpleNet

from torch.utils.data import DataLoader
from viz.plots import single_feature_plt
from experiments.dataloaders import GaussianMixture

def get_data(config):
    means = config["data"]["params"]["means"]
    covs = config["data"]["params"]["covs"]
    nums = config["data"]["params"]["nums"]


    training_gaussian = GaussianMixture(means, covs, len(means) * [nums[0]])
    test_gaussian = GaussianMixture(means, covs, len(means) * [nums[1]])

    train_loader = DataLoader(training_gaussian, batch_size=config["batch_size"]["train_size"], shuffle=True)
    test_loader = DataLoader(test_gaussian, batch_size=config["batch_size"]["test_size"], shuffle=True)

    return train_loader, test_loader


def train(config):
    # init torch
    torch.backends.cudnn.enabled = False
    if config["torch_random_seed"] is not None:
        torch.manual_seed(config["torch_random_seed"])

    # get data
    train_loader, test_loader = get_data(config)


    # Init neural nets and weights
    num_nets = config["num_nets"]
    net_params = config["net"]["params"]
    nets = [SimpleNet(*net_params) for _ in range(num_nets)]
    nets_weights = np.zeros(num_nets)

    #  Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = [optim.SGD(nets[i].parameters(), lr=config["SGD_params"]["learning_rate"],
                            momentum=config["SGD_params"]["momentum"]) for i in range(num_nets)]

    # Set algorithm params
    weight_type = config["weight_type"]

    beta = config["softmax_beta"]

    writer = SummaryWriter()

    # get train loaders for each net
    net_data_loaders = [iter(enumerate(train_loader, 0)) for _ in range(num_nets)]

    # train
    for epoch in range(config["num_epochs"]):
        is_training_epoch = True
        while is_training_epoch:
            for idx_net in range(num_nets):
                # get net and optimizer
                net = nets[idx_net]
                optimizer = optimizers[idx_net]

                # get the inputs; data is a list of [inputs, labels]
                try:
                    i, data = next(net_data_loaders[idx_net])
                except:
                    is_training_epoch = False
                    break
                inputs, labels = data
                labels = labels.reshape(len(labels)).long()

                # Compute gradients for input.
                inputs.requires_grad = True

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs.float(), labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                # update weights
                if weight_type == "input_output_forbenius":
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # get input gradients
                    output_forb = torch.norm(outputs)
                    output_forb.backward()
                    input_grads = inputs.grad

                    curr_weight = weight_function_input_jacobian(input_grads)
                    nets_weights[idx_net] += curr_weight
                elif weight_type == "loss_gradient_weights":

                    param_grads = get_grad_params_vec(net)
                    curr_weight = np.linalg.norm(param_grads)
                    nets_weights[idx_net] += curr_weight
                else:
                    raise NotImplementedError()

                # store training accuracy current
                writer.add_scalar('Loss/train/net {}'.format(idx_net), loss, i + epoch*len(train_loader))

                writer.add_scalar('Weight/net {}'.format(idx_net), curr_weight, i + epoch*len(train_loader))


            writer.add_scalar('Kish/', kish_effs(nets_weights), i + epoch*len(train_loader))

            # Get variation of network weights
            writer.add_scalar('WeightVarTrace/', np.trace(get_params_var(nets)), i + epoch*len(train_loader))


            # Check resample
            if kish_effs(nets_weights) < config["ess_threshold"]:
                # resample particles
                sampled_idx = sample_index_softmax(nets_weights, nets, beta=beta)
                # init nets etc
                nets = [copy.deepcopy(nets[i]) for i in sampled_idx]
                optimizers = [optim.SGD(nets[i].parameters(), lr=config["SGD_params"]["learning_rate"],
                                        momentum=config["SGD_params"]["momentum"]) for i in range(num_nets)]
                nets_weights = np.zeros(num_nets)
                print(sampled_idx)

        # get test error
        correct = 0
        _sum = 0

        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = nets[0](test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('accuracy: {:.2f}'.format(correct / _sum))
        torch.save(nets[0], 'models/mnist_{:.2f}.pkl'.format(correct / _sum))

        # get train loaders for each net
        net_data_loaders = [iter(enumerate(train_loader, 0)) for _ in range(num_nets)]



    return nets



