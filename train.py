import torch as t
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from config import opts
import random
import os
import sys
import torch.optim as optim
from torch.autograd import Variable

from fold_model import GCN
from fold_util import F_Normalize as F_Nor


def train(p_adj_np: np.ndarray, p_feat_np: np.ndarray, p_labels_np: np.ndarray, p_idx_train_np, p_idx_val_np, opt:opts):
    use_gpu = t.cuda.is_available()
    random.seed(opt.np_random_seed)
    np.random.seed(opt.np_random_seed)
    t.manual_seed(opt.np_random_seed)
    best_acc = 0
    label_tensor = t.LongTensor(np.where(p_labels_np)[1])

    # A = A + I
    p_adj_np = (sp.csr_matrix(p_adj_np) + sp.eye(p_adj_np.shape[1])).A

    # Degree Matrix
    degree_np = F_Nor.normalize_adj_degree(p_adj_np)

    '''Row-Normalized feature matrix'''
    if opt.feature_Nor:
        feat_nor_np = F_Nor.normalize_feat(p_feat_np)
    else:
        feat_nor_np = p_feat_np

    """Initialize Model"""
    model = GCN(
        nfeat=feat_nor_np.shape[1],
        nhid=opt.num_hiden_layer,
        nclass=label_tensor.max().item() + 1,
        dropout=opt.drop_out,
        init=opt.init_type
    )

    """Optimizer"""
    if opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError

    """Numpy to Tensor"""
    adj_tensor = t.from_numpy(p_adj_np).float()
    feat_nor_tensor = t.from_numpy(feat_nor_np).float()

    idx_train_tensor = t.from_numpy(p_idx_train_np).long()
    idx_val_tensor = t.from_numpy(p_idx_val_np).long()
    degree_tensor = t.from_numpy(degree_np).float()

    """Tensor CPU -> GPU"""
    if use_gpu:
        model.cuda()
        adj_tensor, feat_nor_tensor, label_tensor, idx_train_tensor, idx_val_tensor, degree_tensor = \
            list(map(lambda x: x.cuda(),
                     [adj_tensor, feat_nor_tensor, label_tensor, idx_train_tensor, idx_val_tensor, degree_tensor]))

    adj_tensor, feat_nor_tensor, label_tensor, degree_tensor = list(
        map(lambda x: Variable(x), [adj_tensor, feat_nor_tensor, label_tensor, degree_tensor]))

    feat_nor_tensor.requires_grad = True

    #  D^-0.5 * A * D^-0.5
    D_Adj_tensor = t.mm(degree_tensor, adj_tensor).cuda()  # D^-0.5 * A
    adj_nor_tensor = t.mm(D_Adj_tensor, degree_tensor).cuda()  # D^-0.5 * A * D^-0.5

    """Save Model"""
    save_point = os.path.join('./checkpoint', opt.dataset)
    if not os.path.isdir(save_point):
        os.mkdir(save_point)

    for epoch in np.arange(1, opt.epoch + 1):
        model.train()  # Training

        optimizer.lr = F_lr_scheduler(epoch, opt)  # Lr decay
        optimizer.zero_grad()

        output = model(feat_nor_tensor, adj_nor_tensor)  # model output
        loss_train = F.nll_loss(output[idx_train_tensor], label_tensor[idx_train_tensor])
        acc_train = F_accuracy(output[idx_train_tensor], label_tensor[idx_train_tensor])

        loss_train.backward()  # Backpropagation
        optimizer.step()

        # Validation
        model.eval()
        output = model(feat_nor_tensor, adj_nor_tensor)
        acc_val = F_accuracy(output[idx_val_tensor], label_tensor[idx_val_tensor])

        if acc_val > best_acc:
            best_acc = acc_val
            state = {
                'model': model,
                'acc': best_acc,
                'epoch': epoch,
            }
            t.save(state, os.path.join(save_point, '%s.t7' % opt.model))  # Save as .t7 eg. GCN.t7
        if epoch % 10 == 0:
            sys.stdout.flush()
            sys.stdout.write('\r')
            sys.stdout.write(" => Training Epoch #{}".format(epoch))
            sys.stdout.write(" | Training acc : {:6.2f}%".format(acc_train.data.cpu().numpy() * 100))
            sys.stdout.write(" | Learning Rate: {:6.4f}".format(optimizer.lr))
            sys.stdout.write(" | Best acc : {:.2f}".format(best_acc.data.cpu().numpy() * 100))


def F_lr_scheduler(epoch, opt):
    return opt.lr * (0.5 ** (epoch / opt.lr_decay_epoch))


# calculate ACC
def F_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


