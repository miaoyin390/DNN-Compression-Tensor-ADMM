# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/11/15 17:58

import torch


def append_double_l2_loss(model, loss, rho, device):
    for name, param in model.named_parameters():
        if 'first_kernel' in name or 'last_kernel' in name or \
                'first_factor' in name or 'last_factor' in name or \
                'left_kernel' in name:
            if param.shape[0] < param.shape[1]:
                eye = torch.eye(param.shape[0]).to(device)
                loss += 0.5 * rho * (torch.norm(torch.mm(torch.squeeze(param), torch.squeeze(param).t()) - eye, p=2)) ** 2
            else:
                eye = torch.eye(param.shape[1]).to(device)
                loss += 0.5 * rho * (torch.norm(torch.mm(torch.squeeze(param).t(), torch.squeeze(param)) - eye, p=2)) ** 2
    return loss
