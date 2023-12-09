import numpy as np
from numpy.linalg import svd
import torch
from torch.nn import Module, Parameter, Embedding
from torch.nn import init


class SVDEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, rank=None, compression_ratio=None, weights=None):
        super(SVDEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if compression_ratio is not None:
            self.rank = int((num_embeddings * embedding_dim) / (compression_ratio * (num_embeddings + embedding_dim)))
        else:
            assert rank is not None
            self.rank = rank
        self.first_factor = Parameter(torch.Tensor(self.num_embeddings, self.rank))
        self.last_factor = Parameter(torch.Tensor(self.rank, self.embedding_dim))
        if weights is not None:
            u, s, v = svd(weights.detach().cpu().numpy(), full_matrices=False)
            u = u[:, :self.rank]
            s = s[:self.rank]
            v = v[:self.rank, :]
            self.first_factor.data = torch.from_numpy(u)
            self.last_factor.data = torch.from_numpy(np.dot(np.diag(s), v))
        else:
            self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.first_factor)
        init.xavier_uniform_(self.last_factor)

    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1)
        select_factors = torch.index_select(self.first_factor, 0, x)
        res = select_factors @ self.last_factor
        res = res.view(x.shape[0], -1)
        res_shape = list(x_shape) + [self.embedding_dim, ]
        res = res.view(*res_shape)
        return res.to(x.device)


if __name__ == '__main__':
    b = Embedding(num_embeddings=1000, embedding_dim=800)
    a = SVDEmbedding(num_embeddings=1000, embedding_dim=800, rank=800, weights=b.weight)
    input = torch.tensor([[3, 202, 333], [11, 422, 366]])
    print(b(input))
    print(a(input))
    print((a(input) - b(input)).abs().sum())
