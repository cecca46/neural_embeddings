import torch
import torch.nn as nn

def square_distance(t1_emb, t2_emb):
    """
    Compute the square distance between two tensors.

    Args:
    t1_emb (torch.Tensor): First tensor.
    t2_emb (torch.Tensor): Second tensor.

    Returns:
    torch.Tensor: The square distance between the two tensors.
    """
    D = t1_emb - t2_emb
    d = torch.sum(D * D, dim=-1)
    return d


def euclidean_distance(t1_emb, t2_emb):
    """
    Compute the Euclidean distance between two tensors.

    Args:
    t1_emb (torch.Tensor): First tensor.
    t2_emb (torch.Tensor): Second tensor.

    Returns:
    torch.Tensor: The Euclidean distance between the two tensors.
    """
    D = t1_emb - t2_emb
    d = torch.norm(D, dim=-1)
    return d


def cosine_distance(t1_emb, t2_emb):
    """
    Compute the cosine distance between two tensors.

    Args:
    t1_emb (torch.Tensor): First tensor.
    t2_emb (torch.Tensor): Second tensor.

    Returns:
    torch.Tensor: The cosine distance between the two tensors.
    """
    return 1 - nn.functional.cosine_similarity(t1_emb, t2_emb, dim=-1, eps=1e-6)


def manhattan_distance(t1_emb, t2_emb):
    """
    Compute the Manhattan distance between two tensors.

    Args:
    t1_emb (torch.Tensor): First tensor.
    t2_emb (torch.Tensor): Second tensor.

    Returns:
    torch.Tensor: The Manhattan distance between the two tensors.
    """
    D = t1_emb - t2_emb
    d = torch.sum(torch.abs(D), dim=-1)
    return d


def hyperbolic_distance(u, v, epsilon=1e-7):  
    """
    Compute the hyperbolic distance between two tensors.

    Args:
    u (torch.Tensor): First tensor.
    v (torch.Tensor): Second tensor.
    epsilon (float, optional): Small value for numerical stability. Defaults to 1e-7.

    Returns:
    torch.Tensor: The hyperbolic distance between the two tensors.
    """
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = torch.sqrt(x ** 2 - 1)
    return torch.log(x + z)
