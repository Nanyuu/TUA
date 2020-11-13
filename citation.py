import argparse

from fold_data import c_dataset_loader
from config import opts
from train import train
from test import test

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='cora', help="training dataset name [\'cora\',\'citeseer\',\'pubmed\']")
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.6, help="dropout, default=0.6")
parser.add_argument("--epoch", type=int, default=800, help="training epoch")
parser.add_argument("--seed", type=int, default=100, help="random initializing seed")

if __name__ == '__main__':
    args = parser.parse_args()
    opt = opts()

    opt.dataset = args.dataset
    opt.lr = args.lr
    opt.drop_out = args.dropout
    opt.epoch = args.epoch
    opt.np_random_seed = args.seed

    # load data
    data_loader = c_dataset_loader(opt.dataset, opt.data_path)
    adj, feat, label, idx_train, idx_val, idx_test = data_loader.process_data()

    # Train model
    train(adj, feat, label, idx_train, idx_val, opt)

    # Test
    test(adj, feat, label, idx_test, opt)



