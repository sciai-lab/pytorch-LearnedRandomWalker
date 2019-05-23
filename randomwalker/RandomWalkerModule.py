#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from .randomwalker2D import RandomWalker2D as RW2D


class RandomWalker(nn.Module):
    def __init__(self, num_grad, max_backprop=True):
        """
        num_grad: Number of sampled gradients
        max_backprop: Compute the loss only on the absolute maximum
        """
        super(RandomWalker, self).__init__()
        self.rw = RW2D
        self.num_grad = num_grad
        self.max_backprop = max_backprop

    def forward(self, e, seeds):
        """
        e: must be a torch tensors [b x 2 x X, Y]
        seeds: must be a torch tensors [b x X, Y]
        """
        out_probabilities = []
        for batch in range(e.size(0)):
            out_probabilities_ = self.rw(self.num_grad, self.max_backprop)(e[batch].cpu(), seeds[batch])

            out_probabilities_ = torch.transpose(out_probabilities_, 0, 1)
            out_probabilities_ = out_probabilities_.view(1, -1, seeds.size(1), seeds.size(2))
            out_probabilities.append(out_probabilities_)

        return out_probabilities
