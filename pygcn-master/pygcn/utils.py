import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    #去重
    classes = set(labels)
    #针对每个元素设置对应的onehot向量
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    #使用map函数遍历原先label的每一个值，用遍历到的元素作为字典get函数的输入(即字典的key)，返回对应的value(即key对应的onehot向量)
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    #读取cora.content文件信息，把每一行的元素根据\t拆分并转化为str类型，得到2708*1435的矩阵
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #把上面的矩阵每一行信息中从1-1434提取出来作为特征矩阵，且每个元素都是float32类型
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)#使用numpy的array数组把数据的每一行第一个元素(即节点序号)取出并转化为int32类型
    idx_map = {j: i for i, j in enumerate(idx)}#对每一个节点的序号进行编码
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32) #从文件中读取边信息 矩阵为5429*2
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), #flaten()是把5423*2的矩阵从第二行开始，每一行都放在第一行结尾，最后形成5429*2=10858的一维list
                     dtype=np.int32).reshape(edges_unordered.shape) #利用map函数遍历faltten后的list中每一个值，用遍历到的元素作为节点编号字典中get函数的输入(即字典的key)，返回对应的value(即key对应的节点编号)，然后再恢复成原先的矩阵形状
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), #因为labels.shape[0]恰好就是样本数量，即这个数据集的总节点数目，因此可以直接构建对应的邻接矩阵
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #把邻接矩阵进行对称化

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1]) #获取每个样本对应的label，这个label就不是one-hot向量，而是这个向量中非0元素的位置。直接确定出class的类别？
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) #对传入的矩阵每一行进行相加得到结果存入一个rowsum的list中，.sum(1)表示对行操作
    r_inv = np.power(rowsum, -1).flatten() #对每一个和取倒数，这样在后面相乘就类似于除以每一行的总和
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) #构建度矩阵
    mx = r_mat_inv.dot(mx) #矩阵点乘运算
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
