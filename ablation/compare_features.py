import sys
import timm
import torch
import resnet_cifar_tk
import resnet_cifar
from parse_args import parse_args
import numpy as np
import matplotlib.pyplot as plt

from datasets import get_data_loader

import utils

hp_dict = utils.get_hp_dict('tkc_resnet32', ratio='2')

tk_model = timm.create_model('tkc_resnet32', hp_dict=hp_dict, decompose=None, pretrained=True, path='../saved_models/tkc_resnet32_cifar10_0129-213040_model.pt')

args = parse_args()
args.batch_size = 32
args.num_workers = 1
args.dataset = 'cifar10'
data_loader = get_data_loader(False, args)

for images, _ in data_loader:
    xs = images
    break

y, tk_features = tk_model.forward_features(xs)
# for item in tk_features.items():
#     print(item[0])
#     for f in item[1]:
#         print(f.shape)

sigma = 0.95

for layer in tk_features.keys():
    print(layer)

for layer in tk_features.keys():

    f = tk_features[layer]
    f1 = f[2].detach().numpy()
    n1 = np.linalg.norm(f1[0,10,:,:], 'nuc')
    u, s, v = np.linalg.svd(f1[0,10,:,:], full_matrices=False)
    sum_s1 = sum(s)
    # sum_ = 0
    # for i in range(len(s)):
    #     sum_ += s[i]
    #     if sum_ > sigma * sum_s1:
    #         print(i/len(s))
    #         break

    f11 = f[1].detach().numpy()[10]
    f11 = np.reshape(f11, [f11.shape[0], -1])
    _, s, _ = np.linalg.svd(f11, full_matrices=False)
    sum_s11 = sum(s)
    sum_ = 0
    for i in range(len(s)):
        sum_ += s[i]
        if sum_ > sigma * sum_s11:
            print(i/len(s))
            break

print('****')
model = timm.create_model('resnet32', pretrained=True, path='../baselines/resnet32_cifar10_0929-221750_model.pt')
x = torch.randn(32, 3, 32, 32)
_, features = model.forward_features(xs)
# for item in features.items():
#     print(item[0], item[1].shape)
for layer in features.keys():
    f = features[layer]
    f2 = f.detach().numpy()
    n2 = np.linalg.norm(f2[0,10,:,:], 'nuc')
    u, s, v = np.linalg.svd(f2[0,10,:,:], full_matrices=False)
    sum_s2 = sum(s)
    # sum_ = 0
    # for i in range(len(s)):
    #     sum_ += s[i]
    #     if sum_ > sigma * sum_s2:
    #         print(i/len(s))
    #         break

    f22 = f2[10]
    f22 = np.reshape(f22, [f22.shape[0], -1])
    _, s, _ = np.linalg.svd(f22, full_matrices=False)
    sum_s22 = sum(s)
    sum_ = 0
    for i in range(len(s)):
        sum_ += s[i]
        if sum_ > sigma * sum_s22:
            print(i/len(s))
            break
