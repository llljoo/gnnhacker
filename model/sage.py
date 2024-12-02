import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
#from dgl import DGLGraph, transform
from dgl.nn.pytorch.conv import SAGEConv
from utils.graph import numpy_to_graph


# Used for inductive case (graph classification) by default.
class GraphSAGE(nn.Module):  
    def __init__(self, in_dim, out_dim,
                 hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
                 dropout=0.2,
                 activation=F.relu,
                 aggregator_type='gcn'):   # mean/gcn/pool/lstm
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        # 首先添加一个输入层
        self.layers.append(SAGEConv(in_dim, hidden_dim[0], aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        # 然后根据hidden_dim中的维度添加隐藏层
        for i in range(len(hidden_dim) - 1):
            self.layers.append(SAGEConv(hidden_dim[i], hidden_dim[i+1], aggregator_type, feat_drop=dropout, activation=activation))
        
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        # 添加一个全连接层（FC层）来将最后一个隐藏层的输出转换为目标输出维度。
        fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self, data, random_smooth=False):
        x, B, N = self.process_batch(data, random_smooth)

        F_prime = x.shape[-1]
        # 将图卷积后的节点特征矩阵x重新整形为(B, N, F_prime)，这里B是批次大小，N是每个图的节点数，F_prime是卷积层输出的特征维度
        x = x.reshape(B, N, F_prime)
        # 在节点维度上（dim=1）对特征进行最大池化，从而得到每个图的一个聚合特征表示。通常比平均池化性能更好，因为它能够更有效地捕捉图中最显著的信号。
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        # x = torch.mean(x, dim=1).squeeze()
        # 将汇聚后的图级特征通过定义在GCN模型中的全连接层进行最终的预测
        x = self.fc(x)
        return x

    def get_embeddings(self, data, random_smooth=False):
        x, B, N = self.process_batch(data, random_smooth)
        return x.reshape(B, N, -1)


    def process_batch(self, data, random_smooth=False):
        batch_g = []
        # 根据邻接矩阵创建一批DGL图对象batch_g
        for adj in data[1]:
            # cannot use tensor init DGLGraph
            batch_g.append(numpy_to_graph(adj.cpu().detach().T.numpy() , to_cuda=adj.is_cuda, sample=random_smooth))
        batch_g = dgl.batch(batch_g)

        # 掩码mask用于在图卷积后保留有效节点的特征，并将无效节点的特征置零。
        mask = data[2]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2) # (B,N,1)  
        
        B, N, F = data[0].shape[:3]
        # 将节点特征矩阵x和掩码mask调整为合适的形状以适配图卷积操作
        x = data[0].reshape(B*N, F)
        mask = mask.reshape(B*N, 1)
        # 通过GraphSAGE的每一层，逐层更新节点特征x，并应用掩码
        for layer in self.layers:
            x = layer(batch_g, x)
            x = x * mask
        

        return x, B, N

