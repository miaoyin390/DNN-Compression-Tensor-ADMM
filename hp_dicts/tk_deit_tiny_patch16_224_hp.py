# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/22 19:05


class HyperParamsDictRatio2x:
    depth = 12
    num_heads = 3
    embed_dim = 192
    ranks = {
        # state_dict key: (output channel, input channel)
        'blocks.0.attn.qkv.weight':   (128, 72),
        'blocks.0.attn.proj.weight':  (72,  72),
        'blocks.0.mlp.fc1.weight':    (128, 72),
        'blocks.0.mlp.fc2.weight':    (72, 128),
        'blocks.1.attn.qkv.weight':   (128, 72),
        'blocks.1.attn.proj.weight':  (72,  72),
        'blocks.1.mlp.fc1.weight':    (128, 72),
        'blocks.1.mlp.fc2.weight':    (72, 128),
        'blocks.2.attn.qkv.weight':   (128, 72),
        'blocks.2.attn.proj.weight':  (72,  72),
        'blocks.2.mlp.fc1.weight':    (128, 72),
        'blocks.2.mlp.fc2.weight':    (72, 128),
        'blocks.3.attn.qkv.weight':   (128, 72),
        'blocks.3.attn.proj.weight':  (72,  72),
        'blocks.3.mlp.fc1.weight':    (128, 72),
        'blocks.3.mlp.fc2.weight':    (72, 128),
        'blocks.4.attn.qkv.weight':   (128, 72),
        'blocks.4.attn.proj.weight':  (72,  72),
        'blocks.4.mlp.fc1.weight':    (128, 72),
        'blocks.4.mlp.fc2.weight':    (72, 128),
        'blocks.5.attn.qkv.weight':   (128, 72),
        'blocks.5.attn.proj.weight':  (72,  72),
        'blocks.5.mlp.fc1.weight':    (128, 72),
        'blocks.5.mlp.fc2.weight':    (72, 128),
        'blocks.6.attn.qkv.weight':   (128, 72),
        'blocks.6.attn.proj.weight':  (72,  72),
        'blocks.6.mlp.fc1.weight':    (128, 72),
        'blocks.6.mlp.fc2.weight':    (72, 128),
        'blocks.7.attn.qkv.weight':   (128, 72),
        'blocks.7.attn.proj.weight':  (72,  72),
        'blocks.7.mlp.fc1.weight':    (128, 72),
        'blocks.7.mlp.fc2.weight':    (72, 128),
        'blocks.8.attn.qkv.weight':   (128, 72),
        'blocks.8.attn.proj.weight':  (72,  72),
        'blocks.8.mlp.fc1.weight':    (128, 72),
        'blocks.8.mlp.fc2.weight':    (72, 128),
        'blocks.9.attn.qkv.weight':   (128, 72),
        'blocks.9.attn.proj.weight':  (72,  72),
        'blocks.9.mlp.fc1.weight':    (128, 72),
        'blocks.9.mlp.fc2.weight':    (72, 128),
        'blocks.10.attn.qkv.weight':  (128, 72),
        'blocks.10.attn.proj.weight': (72,  72),
        'blocks.10.mlp.fc1.weight':   (128, 72),
        'blocks.10.mlp.fc2.weight':   (72, 128),
        'blocks.11.attn.qkv.weight':  (128, 72),
        'blocks.11.attn.proj.weight': (72,  72),
        'blocks.11.mlp.fc1.weight':   (128, 72),
        'blocks.11.mlp.fc2.weight':   (72, 128),
    }
