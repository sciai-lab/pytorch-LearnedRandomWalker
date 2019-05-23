#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from .build_laplacian import build_laplacian2D


def pu2p(pu, seeds):
    """
    :param pu: unseeded output probability
    :param seeds: RW seeds, must be the same size as the image
    :return: p: the complete output probability
    """
    seeds_r = seeds.ravel()
    mask_u = seeds_r == 0
    p = np.zeros((seeds_r.shape[0], pu.shape[-1]), dtype=np.float32)

    for s in range(seeds.max()):
        pos_s = np.where(seeds_r == s + 1)
        p[pos_s, s] = 1
        p[mask_u, s] = pu[:, s]

    return p


def grad_fill(gradu, seeds, edges=2):
    """
    :param gradu: unseeded output probability
    :param seeds: RW seeds, must be the same size as the image
    :param edges: number of affinities for each pixel
    :return: p: the complete output probability
    """
    seeds_r = seeds.ravel()
    mask_u = seeds_r == 0
    grad = np.zeros((edges, seeds_r.shape[0]))
    grad[:, mask_u] = gradu

    return grad


def p2pu(p, seeds):
    """
    :param p: output probability
    :param seeds: RW seeds, must be the same size as the image
    :return: pu: unseeded output probability
    """
    mask_u = seeds.ravel() == 0
    pu = p[mask_u]
    return pu


def lap2lapu_bt(lap, seeds):
    mask_u = seeds.ravel() == 0
    return lap[mask_u][:, mask_u], - lap[mask_u][:, ~mask_u]


def sparse_laplacian(elap, size_image):
    """
    :param elap: Graph weights elements, must be (num pixels, 2)
    :param size_image: size original image, must be a 2D tuple
    :return: graph laplacian as a csc matrix
    """
    if elap.shape[0] == 2:
        e, i_ind, j_ind = build_laplacian2D(elap, size_image)
        laplacian = csc_matrix((e, (i_ind, j_ind)), shape=(np.prod(size_image), np.prod(size_image)))
        return laplacian
    else:
        raise NotImplementedError


def sparse_pm(seeds):
    """
    :param seeds: RW seeds, must be the same size as the image
    :return: poss matrix for the standard RW
    """
    k = np.where(seeds.ravel() != 0)[0]
    i_ind, j_ind = np.arange(k.shape[0]), seeds.ravel()[k] - 1
    val = np.ones_like(k, dtype=np.float)
    return csc_matrix((val, (i_ind, j_ind)), shape=(k.shape[0], j_ind.max() + 1))


def standard_RW(elap, seeds):
    """
    laplacian: Graph laplacian
    seeds: RW seeds, must be the same size as the image
    return: the output probability for each pixel and instances
    """
    laplacian = sparse_laplacian(elap, seeds.shape)
    lap_u, B_T = lap2lapu_bt(laplacian, seeds)
    pm = sparse_pm(seeds)

    # Random Walker Solution
    pu = spsolve(lap_u, B_T.dot(pm), use_umfpack=True)

    # Save out_put for backward
    if type(pu) == np.ndarray:
        return np.array(pu, dtype=np.float32), lap_u
    else:
        return np.array(pu.toarray(), dtype=np.float32), lap_u
