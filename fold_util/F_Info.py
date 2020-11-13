import numpy as np
from config import opts


class C_per_info():
    def __init__(self, p_adj: np.ndarray, p_feat: np.ndarray, p_label: np.ndarray, p_idx_train: np.ndarray,
                 p_idx_val: np.ndarray, p_idx_test: np.ndarray, p_opts: opts = opts):
        self.adj_np = p_adj
        self.feat_np = p_feat
        self.label_np = p_label
        self.idx_train = p_idx_train
        self.idx_val = p_idx_val
        self.idx_test = p_idx_test
        self.opt = p_opts
        self.random_seed = self.opt.np_random_seed

    def F_get_random_n_nodes_from_each_class(self, node_num):
        label = F_one_hot_to_label(self.label_np)
        class_num = label.max() + 1

        # initialize node_index
        node_index = np.zeros([class_num, node_num], np.int16)

        # random choose node
        for ii in range(class_num):
            temp_index_np = np.where(label == ii)[0]
            np.random.seed(self.random_seed)
            node_index[ii] = np.random.choice(temp_index_np, node_num)
        return node_index

    def F_idx_to_class(self, idx):
        """node idx -> label"""
        return np.where(self.label_np)[1][idx]

    def F_get_K_random_idx_of_single_class(self, target_class: int, node_num=10) -> np.ndarray:
        """find K nodes from one class"""
        label_not_one_hot = np.where(self.label_np)[1]
        idx_target_class = np.where(label_not_one_hot == target_class)[0]
        np.random.seed(self.random_seed)
        idx_sample_np = np.random.choice(idx_target_class, node_num)
        return idx_sample_np

    def F_get_K_random_idx_except_one_class(self, except_class: int, node_num=10) -> np.ndarray:
        label_not_one_hot = np.where(self.label_np)[1]
        idx_target_class = np.where(label_not_one_hot != except_class)[0]
        np.random.seed(self.random_seed)
        idx_sample_np = np.random.choice(idx_target_class, node_num)
        return idx_sample_np

    def F_from_label_to_idx(self, label_id: int) -> np.ndarray:
        """through label derive index"""
        label_not_one_hot = F_one_hot_to_label(self.label_np)
        idx_label_np = np.where(label_not_one_hot == label_id)[0]
        return idx_label_np


def F_one_hot_to_label(label: np.ndarray):
    """label"""
    return np.where(label)[1]
