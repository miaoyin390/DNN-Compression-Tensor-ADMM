# from transformer.compressed_modeling_tt import CompressedBertModel
# from transformer.compressed_modeling_tt_57 import CompressedBertForSequenceClassification
# from transformer.modeling import BertForSequenceClassification
# model = CompressedBertForSequenceClassification.from_scratch('models/bert-td-36-128')
# size = 0
# emb_size = 0
# for n, p in model.named_parameters():
#     if 'embeddings' in n:
#         emb_size += p.nelement()
#     size += p.nelement()
#
# print(model)
#
# # print(model)
# print('Total student parameters: {}'.format(size))
# print('Embedding parameters: {}'.format(emb_size))
# print('Backbone parameters: {}'.format(size - emb_size))
# print(85.6*(10**6)/(size - emb_size))

import torch

n = torch.randn([100, 100])
print(torch.norm(n, p='fro'))
print(torch.norm(n, p='nuc'))
print(torch.norm(n, p=1))
print(torch.norm(n, p=2))