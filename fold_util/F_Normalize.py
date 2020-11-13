import numpy as np
import torch as t
import scipy.sparse as sp


def normalize_feat(feat_np: np.ndarray) -> np.ndarray:
    """
    row-normalization
    """
    feat_csr = sp.csr_matrix(feat_np)  # NdArray -> Csr_matrix
    rowsum = np.array(feat_csr.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)  #
    feat_nor_np = r_mat_inv.dot(feat_csr)
    return feat_nor_np.A


def normalize_adj(adj_np: np.ndarray) -> np.ndarray:
    """input adj  -> output : D^-0.5 * A * D^-0.5"""
    adj_csr = sp.csr_matrix(adj_np)

    # add identity matrix
    adj_eye_csr = adj_csr + sp.eye(adj_np.shape[1])
    adj_eye_np = adj_eye_csr.A

    # compute  Degree Matrix
    row_sum = np.array(adj_eye_np.sum(1))
    # compute D^-0.5
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    mx_dot = adj_eye_csr.dot(r_mat_inv_sqrt)  # A*D^0.5
    mx_dot_trans = mx_dot.transpose()
    mx_dot_trans_dot = mx_dot_trans.dot(r_mat_inv_sqrt)
    mx_dot_trans_dot_np = mx_dot_trans_dot.A

    return mx_dot_trans_dot_np

def nor_sub_adj_eye(p_sub_adj, p_sub_d):
    """
    1. A = A+I
    2. d^-0.5 * A * d^-0.5
    """
    # compute d^-0.5
    d_inv_sqrt = np.power(np.diagonal(p_sub_d), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt_sp = sp.diags(d_inv_sqrt)

    # adj_csr
    adj_eye_csr = sp.csr_matrix(p_sub_adj + np.eye(p_sub_adj.shape[0]))

    # D^-0.5 * A * D^-0.5
    mx_dot = adj_eye_csr.dot(d_inv_sqrt_sp)  # A*D^0.5
    mx_dot_trans = mx_dot.transpose()
    mx_dot_trans_dot = mx_dot_trans.dot(d_inv_sqrt_sp)
    mx_dot_trans_dot_np = mx_dot_trans_dot.A

    return mx_dot_trans_dot_np

def normalize_adj_degree(adj_np: np.ndarray) -> np.ndarray:
    """
    D^-0.5
    :param adj: Adjacency Matrix
    :return: D^-0.5
    """
    adj_csr = sp.csr_matrix(adj_np)
    rowsum = np.array(adj_csr.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    degree_np = r_mat_inv_sqrt.A
    return degree_np