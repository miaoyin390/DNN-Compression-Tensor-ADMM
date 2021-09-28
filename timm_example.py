# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/15 23:14

import torch
import timm


model = timm.create_model('resnet18', pretrained=True)