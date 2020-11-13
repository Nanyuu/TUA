import torch as t
import numpy as np
import os
import scipy.sparse as sp

from config import opts
from fold_util import F_Normalize as F_Nor
from train import F_accuracy


def test(p_adj_np: np.ndarray, p_feat_np: np.ndarray, p_labels_np: np.ndarray, p_index_test: np.ndarray,
         opt:opts):
    print("\nTesting")
    use_gpu = t.cuda.is_available()

    if opt.feature_Nor:
        feat_nor_np = F_Nor.normalize_feat(p_feat_np)
    else:
        feat_nor_np = p_feat_np

    # A = A + I
    p_adj_np = (sp.csr_matrix(p_adj_np) + sp.eye(p_adj_np.shape[1])).A

    # Degree matrix
    degree_np = F_Nor.normalize_adj_degree(p_adj_np)

    # Numpy -> Tensor
    adj_tensor = t.from_numpy(p_adj_np).float()
    feat_nor_tensor = t.from_numpy(feat_nor_np).float()
    label_tensor = t.from_numpy(np.where(p_labels_np)[1]).long()
    idx_test_tensor = t.from_numpy(p_index_test).long()
    degree_tensor = t.from_numpy(degree_np).float()

    '''dataset info'''
    print("\nObtain(Adj,Feat,Label) matrix")
    print("| Adj : {}".format(p_adj_np.shape))
    print("| Feat: {}".format(feat_nor_np.shape))
    print("| label:{}".format(p_labels_np.shape))

    opt.model_path = "./checkpoint"
    load_model = t.load("{}/{}/{}.t7".format(opt.model_path, opt.dataset, opt.model))
    model = load_model['model'].cpu()
    acc_val = load_model['acc']
    print("best epoch was : {}".format(load_model['epoch']))

    if use_gpu:
        model.cuda()
        adj_tensor, feat_nor_tensor, label_tensor, idx_test_tensor, degree_tensor = list(map(lambda x: x.cuda(), [adj_tensor, feat_nor_tensor, label_tensor, idx_test_tensor, degree_tensor]))

    adj_tensor, feat_nor_tensor, label_tensor, degree_tensor = list(
        map(lambda x: t.autograd.Variable(x), [adj_tensor, feat_nor_tensor, label_tensor, degree_tensor]))

    feat_nor_tensor.requires_grad = True

    # normalize
    D_Adj_tensor = t.mm(degree_tensor, adj_tensor).cuda()
    adj_nor_tensor = t.mm(D_Adj_tensor, degree_tensor).cuda()

    # Test mode
    model.eval()
    output = model(feat_nor_tensor, adj_nor_tensor)

    # ACC
    acc_test = F_accuracy(output[idx_test_tensor], label_tensor[idx_test_tensor])

    print("test_acc = {}".format(acc_test))
    print("val_acc = {}".format(acc_val))

    return acc_test


