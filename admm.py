# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/17 15:59

import torch
import tensorly as tl
from tensorly.decomposition import partial_tucker

import numpy as np
from numpy.linalg import svd
from ttd import ten2tt, tt2ten


class ADMM:
    def __init__(self, model, rho, hp_dict, format, device, verbose=False, log=False):
        tl.set_backend('pytorch')
        self.model = model
        self.init_rho = rho
        self.hp_dict = hp_dict
        self.format = format
        self.device = device
        self.verbose = verbose
        self.log = log
        if self.log:
            self.logger = {}

        if format == 'none':
            raise Exception('ERROR: Tensor format should be specified!')

        self.rho = self.init_rho

        self.u = {}
        self.z = {}

        for name, param in self.model.named_parameters():
            if name in self.hp_dict.ranks.keys():
                self.u[name] = torch.zeros(param.shape).to(device)
                self.z[name] = torch.Tensor(param.data.cpu().clone().detach()).to(device)
                if self.log:
                    self.logger[name] = []

    def update(self, update_u=True):
        for name, param in self.model.named_parameters():
            if name in self.hp_dict.ranks.keys():
                z = param.data + self.u[name]
                # if 'conv' in name:
                if len(param.shape) == 4:
                    if self.format == 'tk':
                        self.z[name].data = self.prune_conv_rank_tk(z, name)
                    elif self.format == 'tt':
                        self.z[name] = torch.from_numpy(
                            self.prune_conv_rank_tt(z.detach().cpu().numpy(), name)).to(self.device)
                    else:
                        self.z[name] = torch.from_numpy(
                            self.prune_conv_rank_svd(z, name)).to(self.device)
                # elif 'fc' in name or 'qkv' in name or 'proj' in name:
                elif len(param.shape) == 2:
                    if self.format == 'tk':
                        self.z[name] = torch.from_numpy(
                            self.prune_linear_rank_tk(z.detach().cpu().numpy(), name)).to(self.device)
                    elif self.format == 'tt':
                        self.z[name] = torch.from_numpy(
                            self.prune_linear_rank_tt(z.detach().cpu().numpy(), name)).to(self.device)
                    else:
                        self.z[name] = torch.from_numpy(
                            self.prune_linear_rank_svd(z, name)).to(self.device)
                else:
                    raise Exception('ERROR: unsupported layer in ADMM!')

                if update_u:
                    with torch.no_grad():
                        diff = param.data - self.z[name]
                        self.u[name] += diff
                        if self.log:
                            self.logger[name].append(float(torch.norm(diff)))
                        if self.verbose:
                            print('*INFO: {} in ADMM, norm(w-z)={}'.format(name, torch.norm(diff, p=2)))

    def append_admm_loss(self, loss):
        for name, param in self.model.named_parameters():
            if name in self.hp_dict.ranks.keys():
                loss += 0.5 * self.rho * (torch.norm(param - self.z[name] + self.u[name], p=2)) ** 2
                # dense_loss += 0.5 * rho * p.norm()
        return loss

    def adjust_rho(self, epoch, epochs, factor=5):
        if epoch > int(0.85 * epochs):
            self.rho = factor * self.init_rho

    def prune_conv_rank_tt(self, z, name):
        kernel_shape = z.shape
        filter_dim = kernel_shape[2] * kernel_shape[3]
        tt_ranks = self.hp_dict.ranks[name]
        tt_shapes = self.hp_dict.tt_shapes[name]
        t = np.transpose(np.reshape(z, (kernel_shape[0], kernel_shape[1], filter_dim)), (0, 2, 1))
        tt_cores = ten2tt(t, tt_shapes, tt_ranks)
        updated_z = tt2ten(tt_cores, (kernel_shape[0], filter_dim, kernel_shape[1]))
        updated_z = np.reshape(np.transpose(updated_z, (0, 2, 1)), kernel_shape)

        return updated_z

    def prune_linear_rank_tt(self, z, name):
        weight_shape = z.shape
        tt_ranks = list(self.hp_dict.ranks[name])
        tt_shapes = self.hp_dict.tt_shapes[name]
        t = np.reshape(z, tt_shapes)
        tt_cores = ten2tt(t, tt_shapes, tt_ranks)
        updated_z = tt2ten(tt_cores, weight_shape)

        return updated_z

    def prune_conv_rank_tk(self, z, name):
        ranks = self.hp_dict.ranks[name]
        core_tensor, (last_factor, first_factor) = partial_tucker(z, modes=[0, 1], rank=ranks, init='svd')
        updated_z = tl.tucker_to_tensor((core_tensor, (last_factor, first_factor)))

        return updated_z

    def prune_linear_rank_tk(self, z, name):
        tl.set_backend('numpy')
        ranks = self.hp_dict.ranks[name]
        core_tensor, (last_factor, first_factor) = partial_tucker(z, modes=[0, 1], rank=ranks, init='svd')
        updated_z = tl.tucker_to_tensor((core_tensor, (last_factor, first_factor)))

        return updated_z

    def prune_conv_rank_svd(self, z, name):
        rank = self.hp_dict.ranks[name]
        u, s, v = np.linalg.svd(z.detach().squeeze().cpu().numpy(), full_matrices=False)
        u = u[:, :rank]
        s = s[:rank]
        v = v[:rank, :]
        updated_z = u @ np.diag(s) @ v
        updated_z = np.expand_dims(updated_z, -1)
        updated_z = np.expand_dims(updated_z, -1)

        return updated_z

    def prune_linear_rank_svd(self, z, name):
        rank = self.hp_dict.ranks[name]
        u, s, v = np.linalg.svd(z.detach().cpu().numpy(), full_matrices=False)
        u = u[:, :rank]
        s = s[:rank]
        v = v[:rank, :]
        updated_z = u @ np.diag(s) @ v

        return updated_z
