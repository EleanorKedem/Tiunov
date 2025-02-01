import numpy as np
import torch
import matplotlib.pyplot as plt


def energy(J, b, s):
    """
    Compute the energy of a given solution.

    Args:
        J (torch.Tensor): Interaction matrix.
        b (torch.Tensor): Bias vector.
        s (torch.Tensor): Binary solution vector.

    Returns:
        torch.Tensor: Computed energy value.
    """
    return -0.5 * torch.einsum('in,ij,jn->n', s, J, s) - torch.einsum('in,ik->n', s, b)


def length(order, lengths):
    """
    Compute the total path length for a given order of cities.

    Args:
        order (torch.Tensor): Order of cities visited.
        lengths (torch.Tensor): Distance matrix between cities.

    Returns:
        torch.Tensor: Total path length.
    """
    order1 = torch.zeros(order.shape[0], dtype=torch.long)
    order1[:-1] = order[1:]
    order1[-1] = order[0]
    return torch.sum(lengths[(order, order1)])


def Qubo(lengths, A, B):
    """
    Construct a QUBO matrix for the traveling salesman problem.

    Args:
        lengths (numpy.ndarray): Distance matrix between cities.
        A (float): Penalty coefficient for city constraints.
        B (float): Penalty coefficient for path constraints.

    Returns:
        tuple: (Q, b), where Q is the QUBO matrix and b is the bias vector.
    """
    N_cities = lengths.shape[0]
    Q = np.zeros((N_cities, N_cities, N_cities, N_cities))
    inds0 = np.arange(N_cities)
    inds1 = np.concatenate((inds0[1:], inds0[0:1]))
    Q[:, inds0, :, inds1] += B * np.repeat(lengths.reshape(1, N_cities, N_cities), N_cities, axis=0)
    dims = np.arange(N_cities)
    Q[dims, :, dims, :] += A
    inds0 = dims.reshape(N_cities, 1).repeat(N_cities, axis=1).flatten()
    inds1 = dims.reshape(1, N_cities).repeat(N_cities, axis=0).flatten()
    Q[inds0, inds1, inds0, inds1] -= Q[inds0, inds1, inds0, inds1]
    Q[:, dims, :, dims] += A
    Q[inds1, inds0, inds1, inds0] -= Q[inds1, inds0, inds1, inds0]
    b = -np.ones((N_cities, N_cities)) * 2 * A
    return Q, b


def get_Jh(lengths, A, B):
    """
    Convert the QUBO matrix for TSP into the J and h parameters for SimCIM.

    Args:
        lengths (numpy.ndarray): Distance matrix between cities.
        A (float): Penalty coefficient for city constraints.
        B (float): Penalty coefficient for path constraints.

    Returns:
        tuple: (J, h), where J is the interaction matrix and h is the bias vector.
    """
    N_variables = lengths.shape[0]
    Q, b = Qubo(lengths, A, B)
    Q = torch.tensor(Q.reshape(N_variables ** 2, N_variables ** 2), dtype=torch.float32)
    b = torch.tensor(b.reshape(N_variables ** 2), dtype=torch.float32)
    Q = 0.5 * (Q + Q.t())
    J = -0.5 * Q
    h = -0.5 * (Q.sum(1) + b)
    h = h.reshape(-1, 1)
    return J, h


# def H(Q, b, x):
#     return torch.einsum('ij,ni,nj->n', Q, x, x) + x @ b
#
#
# def int2base(nums, base, N_digits):
#     nums_cur = torch.clone(nums)
#     res = torch.empty((nums.shape[0], N_digits))
#     for i in range(N_digits):
#         res[:, N_digits - 1 - i] = torch.remainder(nums_cur, base)
#         nums_cur = (nums_cur / base).type(torch.long)
#     return res


#def get_order(x, lengths, A, B):
#    Q, b = Qubo(lengths, A, B)
#    Q = torch.tensor(Q.reshape(N_cities ** 2, N_cities ** 2), dtype=torch.float32)
#    b = torch.tensor(b.reshape(N_cities ** 2), dtype=torch.float32)
#    Q = 0.5 * (Q + Q.t())
#    ind_min = torch.argmin(H(Q, b, x.type(torch.float32)))
#    inds_nonzero = np.nonzero(x[ind_min].reshape(N_cities, N_cities))
#    inds_order = (inds_nonzero[:, 1].sort()[1])
#    order = inds_nonzero[:, 0][inds_order]
#    return order

def get_order_simcim(s_min, N_cities):
    """
    Extract the city visit order from the SimCIM solution.

    Args:
        s_min (torch.Tensor): Solution vector from SimCIM.
        N_cities (int): Number of cities in the TSP problem.

    Returns:
        torch.Tensor: Order of cities visited.
    """
    inds_nonzero = np.nonzero((0.5 * (s_min + 1)).reshape(N_cities, N_cities))
    inds_order = (inds_nonzero[:, 1].sort()[1])
    order = inds_nonzero[:, 0][inds_order]
    return order