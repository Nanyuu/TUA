import numpy as np


def find_neighbor_idx(p_adj: np.ndarray, p_hops: int, p_node_idx: int):
    """
    Find neighbors within K-hop
    :param p_adj: adjacency matrix  e.g [2708,2708] for cora
    :param p_hops: hops
    :param p_node_idx: target node
    :return: adjacency matrix
    """

    neighbor_matrix = np.array([p_node_idx], dtype=np.int16)
    for lp_hop in range(p_hops):
        # loop top hop
        for lp_node_id in range(neighbor_matrix.shape[0]):

            # find neighbor
            temp_neighbors = np.where(p_adj[neighbor_matrix[lp_node_id]] != 0)[0]

            # check each node
            for idx in temp_neighbors:
                if neighbor_matrix.__contains__(idx):
                    continue
                else:
                    neighbor_matrix = np.append(neighbor_matrix, idx)

    return np.sort(neighbor_matrix)


def construct_sub_graph(p_adj, p_feat, p_node_set: np.ndarray):
    """
    construct subgraph based on node set
    :param p_adj: adjacency matrix
    :param p_feat:feature matrix
    :param p_node_set:ndoe set
    :return: sub_Adj, sub_Degree ,sub_Feat
    """

    # create correspondence between original Graph and subGraph
    proj_o_to_s = {}  # origin to sub
    proj_s_to_o = {}  # sub to origin
    for lp_set_id in range(p_node_set.shape[0]):
        proj_s_to_o[lp_set_id] = p_node_set[lp_set_id]
        proj_o_to_s[p_node_set[lp_set_id]] = lp_set_id

    # initialize sub_Adj
    sub_adj = np.zeros([p_node_set.shape[0], p_node_set.shape[0]])

    # construct Sub_adj
    for lp_node_i in p_node_set:
        for lp_node_j in p_node_set:
            if p_adj[lp_node_i, lp_node_j] == 1:
                sub_idx_i = proj_o_to_s[lp_node_i]
                sub_idx_j = proj_o_to_s[lp_node_j]
                sub_adj[sub_idx_i, sub_idx_j] = 1

    # compute degree matrix based on origin_ADJ
    sub_d = np.diag(p_adj[p_node_set].sum(1))

    # construct Sub_feat
    sub_feat = np.copy(p_feat[p_node_set])

    # Return sub_Adj, sub_Degree ,sub_Feat
    return sub_adj, sub_d, sub_feat
