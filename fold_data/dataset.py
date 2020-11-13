import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


class c_dataset_loader():
    """
    data is from:
    https://github.com/kimiyoung/planetoid
    """
    def __init__(self, dataset_name, data_path):
        self.path = data_path
        self.dataset = dataset_name

    def get_train(self):
        '''
        load training data
        :return: ndarry - x，ndarry - y
        '''
        index_train = ['x', 'y']
        objects = []
        for i in range(len(index_train)):
            with open("{}/ind.{}.{}".format(self.path, self.dataset, index_train[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))
        x, y = tuple(objects)  # x- csr_matrix, y - ndarray
        x = x.A  # csr_matrix -> ndarray
        return x, y

    def get_test(self):
        """
        load validation matrix
        :return: ndarray -tx , ndarray - ty
        """
        index_val = ['tx', 'ty']
        objects = []
        for i in range(len(index_val)):
            with open("{}/ind.{}.{}".format(self.path, self.dataset, index_val[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))
        tx, ty = tuple(objects)  # tx- csr_matrix, y - ndarray
        tx = tx.A  # csr_matrix -> ndarray
        return tx, ty

    def get_all(self):
        """
        :return: ndarray - allx, ndarray - ally
        """
        index_test = ['allx', 'ally']
        objects = []
        for i in range(len(index_test)):
            with open("{}/ind.{}.{}".format(self.path, self.dataset, index_test[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))
        allx, ally = tuple(objects)
        allx = allx.A
        return allx, ally

    def parse_index_file(self, filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def get_adj(self):
        """
        :return: graph - NetworkX
        """
        with open("{}/ind.{}.{}".format(self.path, self.dataset, "graph"), 'rb') as f:
            graph = pkl.load(f, encoding='latin1')

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        print("| # of nodes:{}".format(adj.shape[0]))
        print("| # of edges:{}".format(adj.sum().sum() / 2))
        return adj.A

    def process_data(self):
        """
        Get adj, feature, label, index_train, index_val , index_test
        :return: adj, feature, label, index_train, index_val , index_test
        """
        # Get Data NdArray
        x, y = self.get_train()  # e.g. x - NdArray ~ [140,1433] ; y - NdArray ~ [140,7]  Cora
        tx, ty = self.get_test()  # e.g. tx - NdArray ~ [1000,1433] ; y - NdArray ~ [1000,7]   Cora
        allx, ally = self.get_all()  # e.g. allx - NdArray ~ [1708,1433] ; ally - NdArray ~ [1708,1433]   Cora
        adj = self.get_adj()

        # reorder test set
        test_idx_reorder = self.parse_index_file("{}/ind.{}.test.index".format(self.path, self.dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if self.dataset == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extend = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extend[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extend

            ty_extend = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extend[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extend

        # construct feature matrix
        features_lil = sp.lil_matrix(sp.vstack((sp.lil_matrix(allx), sp.lil_matrix(tx))))
        features_lil[test_idx_reorder, :] = features_lil[test_idx_range, :]
        features_np_noNor = features_lil.toarray()

        # reorder labels
        labels_np = np.vstack((ally, ty))
        labels_np[test_idx_reorder, :] = labels_np[test_idx_range, :]

        if self.dataset == 'citeseer':
            save_label = np.where(labels_np)[1]

        # set train,val,test set
        index_train = np.arange(len(y))  # train index
        index_val = np.arange(len(y), len(y) + 500)  # validation index
        index_test = test_idx_range  # test index

        def missing_elements(L):
            start, end = L[0], L[-1]
            return sorted(set(range(start, end + 1)).difference(L))

        # check
        if self.dataset == 'citeseer':
            L = np.sort(index_test)
            missing = missing_elements(L)
            for element in missing:
                save_label = np.insert(save_label, element, 0)
            labels_np = save_label
            label_class_num = labels_np.max() + 1
            labels_np = np.eye(label_class_num)[labels_np]

        return adj, features_np_noNor, labels_np, index_train, index_val, index_test

