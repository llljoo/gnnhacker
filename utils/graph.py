import torch
import dgl
import networkx as nx
import numpy as np
import random
from trojan.prop import matrix_sample


def numpy_to_graph(A, type_graph='dgl', node_features=None, to_cuda=True, sample=False):
    '''Convert numpy arrays to graph

    Parameters
    ----------
    A : mxm array
        Adjacency matrix
    type_graph : str
        'dgl' or 'nx'
    node_features : dict
        Optional, dictionary with key=feature name, value=list of size m
        Allows user to specify node features

    Returns

    -------
    Graph of 'type_graph' specification
    '''
    '''
    将一个邻接矩阵（以NumPy数组形式给出）转换为图对象。函数支持生成NetworkX或DGL（Deep Graph Library）图对象，并且可以为图中的节点附加特征。
    这对于使用图神经网络处理图数据尤其有用，因为许多图神经网络库（如DGL）要求数据以图形式表示。
    '''
    if sample:
        n = A.shape[0]
        k = int(0.1 * n * (n - 1) / 2)
        mask = matrix_sample(n, k)
        A *= mask
        # print('------Ramdomly sample')


    G = nx.from_numpy_array(A)
    
    if node_features != None:
        for n in G.nodes():
            for k, v in node_features.items():
                G.nodes[n][k] = v[n]
    
    if type_graph == 'nx':
        return G
    
    G = G.to_directed()
    
    if node_features != None:
        node_attrs = list(node_features.keys())
    else:
        node_attrs = []
        
    g = dgl.from_networkx(G)#, node_attrs=node_attrs, edge_attrs=['weight'])
    if to_cuda:
        g = g.to(torch.device('cuda'))
    return g


