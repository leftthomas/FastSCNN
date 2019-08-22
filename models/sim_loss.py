import torch
import torch.nn.functional as F
from torch import nn


def pairwise_similarity(x, y=None):
    if y is None:
        y = x
    # normalization
    y = normalize(y)
    x = normalize(x)
    # similarity
    similarity = torch.mm(x, y.t())
    return similarity


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


class SimLoss(nn.Module):
    def __init__(self, margin=0):
        super(SimLoss, self).__init__()
        self.margin = margin

    def forward(self, embed1, embed2):
        # Compute similarity matrix
        sim_mat = pairwise_similarity(embed1, embed2)
        loss = F.relu(sim_mat.diag().mean() - self.margin)
        return loss

# sim = SimLoss(margin=0.5)
# b = sim(torch.randn(4,512),torch.randn(4,512))
# print(b)
