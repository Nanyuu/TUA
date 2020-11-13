import numpy as np


class opts():
    """Training setting"""
    lr = 0.0045
    num_hiden_layer = 8
    init_type = 'xavier'  # initialization [norm, xavier, kaiming]
    drop_out = 0.6
    optim = "adam"  # optimizer [adam, sgd]
    weight_decay = 5e-4
    dataset = 'cora'  # datasets [cora, citeseer, pubmed]
    epoch = 800  # training epoch
    lr_decay_epoch = 5000
    feature_Nor = True  # row normalization for features X(i,j) = X(i,j)/ ΣX(i), i= 1..N , j = 1..d
    model = 'GCN'
    """ Attack settings"""
    data_path = "./fold_data/Data/Planetoid"
    model_path = "./checkpoint"
    np_random_seed = 100  # random seed
    use_cuda = True
    fake_node_num_each_attack_node = 2  # fake node number
    limit_fake_feat = 25  # budget
    attack_node_num = 3
    flag_att_group_nodes = 1

    # Node number for each dataset
    adj_num = {"cora":2708,"citeseer":3327, "pubmed":19717}
