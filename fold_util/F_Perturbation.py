import numpy as np

def Per_add_fake_node(adj_np: np.ndarray, feat_np: np.ndarray, attack_node_index: int, node_num=1):
    """
    add fake nodes to the attack nodes
    :param adj_np:
    :param target_index:
    :param node_num:
    :return: adj_fake_np, feat_fake_np
    """

    shape = adj_np.shape[0]
    # initialize adj matrix
    adj_fake_np = np.zeros((shape + node_num, shape + node_num))
    adj_fake_np[:shape, :shape] = adj_np

    # add link between attack node and fake nodes
    for i in range(node_num):
        adj_fake_np[attack_node_index, shape + i] = 1
        adj_fake_np[shape + i, attack_node_index] = 1

    # add feature to feature matrix
    feat_num = feat_np.shape[1]
    feat_fake_np = np.zeros((shape + node_num, feat_num))
    feat_fake_np[:shape, :feat_num] = feat_np
    return adj_fake_np, feat_fake_np


def Per_add_fake_feat_based_on_grad_multi_attack_nodes(p_grad_np: np.ndarray, p_feat_fake: np.ndarray):
    """
    according to the derivation, modify feature of fake nodes
    """

    # initialize
    fake_node_num = p_grad_np.shape[0]
    feat_num = p_grad_np.shape[1]
    total_node_num = p_feat_fake.shape[0]
    feat_fake = np.copy(p_feat_fake)

    """find the Grad(i,j)"""
    grad_exclude_sum_sort = np.unique(np.sort(p_grad_np.flatten()))

    # find the max, if not, check next maxi value
    for ii in range(50):
        temp_feat_arg = grad_exclude_sum_sort[ii]
        temp_feat_idx_np = np.where(p_grad_np.flatten() == temp_feat_arg)[0]

        # check if there are same grad value
        for jj in range(temp_feat_idx_np.shape[0]):
            temp_feat_node_idx = int(temp_feat_idx_np[jj] / feat_num) + total_node_num - fake_node_num
            temp_feat_idx = int(temp_feat_idx_np[jj] % feat_num)

            if feat_fake[temp_feat_node_idx, temp_feat_idx] == 1:
                continue
            else:
                feat_fake[temp_feat_node_idx, temp_feat_idx] = 1
                return feat_fake

    return feat_fake









