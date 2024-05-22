import torch
from torch.nn import functional as F


def dist(output1, vector, label=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_norm2 = F.pairwise_distance(output1, vector).to(device)
    d_norm1 = torch.sum(torch.pow(torch.abs(output1 - vector), 1)).to(device)
    d_norm3 = torch.sum(torch.pow(torch.abs(output1 - vector), 3)).to(device)
    d_infinity = torch.max(torch.abs(output1 - vector)).to(device)
    d_cosine = F.cosine_similarity(output1, vector).to(device)
    weights = F.softmax(
        torch.Tensor(
            [
                d_norm1 * 0.001,
                d_norm2 * 0.04,
                d_norm3,
                d_infinity * 0.04,
                d_cosine * len(output1) * 0.01,
            ]
        ),
        dim=0,
    ).to(device)
    norms = torch.Tensor([d_norm1, d_norm2, d_norm3, d_infinity, d_cosine]).to(device)
    sum = torch.dot(weights, norms)
    return sum


def L1_dist(vec1, vec2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.sum(torch.abs(vec1 - vec2)).to(device)


def L2_dist(vec1, vec2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return F.pairwise_distance(vec1, vec2).to(device)


def L3_dist(vec1, vec2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.sum(torch.pow(torch.abs(vec1 - vec2), 3)).to(device)


def L4_dist(vec1, vec2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.sum(torch.pow(torch.abs(vec1 - vec2), 4)).to(device)


def L_inf(vec1, vec2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.max(torch.abs(vec1 - vec2)).to(device)
