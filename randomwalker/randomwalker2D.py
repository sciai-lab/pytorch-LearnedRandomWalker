#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from sksparse.cholmod import cholesky
import torch
from torch.autograd import Function
from randomwalker import randomwalker_tools


class RandomWalker2D(Function):
    def __init__(self, num_grad=1000, max_backprop=True):
        super(RandomWalker2D, self).__init__()
        """
        num_grad: Number of sampled gradients
        max_backprop: Compute the loss only on the absolute maximum
        """
        self.num_grad = num_grad
        self.max_backprop = max_backprop
        self.lap_u = None
        self.pu = None
        self.gradout = None
        self.ch_lap = None
        self.c_max = None

    def forward(self, elap_input, seeds_input):
        """
        input : essential Laplacian (s, e edge for each pixel) shape: N_pixels x 2

        output: instances probability shape: N_pixels x N_seeds
        """
        # Pytorch Tensors to numpy
        elap = elap_input.clone().numpy()
        seeds = seeds_input.numpy()
        elap = np.squeeze(elap)

        # Building laplacian and running the RW
        self.pu, self.lap_u = randomwalker_tools.standard_RW(elap, seeds)

        # Fill seeds predictions
        p = randomwalker_tools.pu2p(self.pu, seeds)
        # save for backward
        self.save_for_backward(seeds_input)

        return torch.from_numpy(p)

    def backward(self, grad_output):
        """
        input : grad from loss
        output: grad from the laplacian backprop
        """

        # Pytorch Tensors to numpy
        gradout = grad_output.numpy()
        seeds = self.saved_tensors[0].numpy()

        # Remove seeds from grad_output
        self.gradout = randomwalker_tools.p2pu(gradout, seeds)

        # Back propagation
        grad_input = self.dlap_df()

        grad_input = randomwalker_tools.grad_fill(grad_input, seeds, 2).reshape(-1, seeds.shape[0], seeds.shape[1])

        grad_input = grad_input[None, ...]

        return torch.FloatTensor(grad_input), None

    def dlap_df(self):
        """
        Sampled back prop implementation
        grad_input: The gradient input for the previous layer
        """

        # Solver + sampling
        grad_input = np.zeros((2, self.pu.shape[0]))
        lap_u = coo_matrix(self.lap_u)
        ind_i, ind_j = lap_u.col, lap_u.row

        # mask n and w direction
        mask = (ind_j - ind_i) > 0
        ind_i, ind_j = ind_i[mask], ind_j[mask]

        # find the edge direction
        mask = ind_j - ind_i == 1
        dir_e = np.zeros_like(ind_i)
        dir_e[mask] = 1

        # Sampling
        if self.num_grad < np.unique(ind_i).shape[0]:
            u_ind = np.unique(ind_i)
            grad2do = np.random.choice(u_ind, size=self.num_grad, replace=False)
        else:
            grad2do = np.unique(ind_i)

        # Compute the choalesky decomposition
        self.ch_lap = cholesky(csc_matrix(self.lap_u))

        # find maxgrad for each region
        if self.max_backprop:
            self.c_max = np.argmax(np.abs(self.gradout), axis=1)
        else:
            # only biggest 10
            self.c_max = np.argsort(np.abs(self.gradout), axis=1)

        # Loops around all the edges
        for k, l, e in zip(ind_i, ind_j, dir_e):
            if k in grad2do:
                grad_input[e, k] = self.compute_grad(k, l)

        return grad_input

    def compute_grad(self, k, l):
        """
        k, l: pixel indices, referred to the unseeded laplacian
        ch_lap: choaleshy decomposition of the undseeded laplacian
        pu: unseeded output probability
        gradout: previous layer jacobian
        return: grad for the edge k, l
        """
        dl = np.zeros_like(self.pu)
        dl[l] = self.pu[k] - self.pu[l]
        dl[k] = self.pu[l] - self.pu[k]

        partial_grad = self.ch_lap.solve_A(dl[:, self.c_max[k]])
        grad = np.sum(self.gradout[:, self.c_max[k]] * partial_grad)
        return grad
