import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from utils.graph import numpy_to_graph

class GINLayer(nn.Module):
    def __init__(self, in_feats, out_feats, eps=0, learn_eps=False):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        self.eps = nn.Parameter(torch.Tensor([eps])) if learn_eps else eps

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            # 在GIN中，我们聚合自己的特征和邻居的特征
            g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='neigh'))
            # h = (1 + eps) * 自己的特征 + 聚合的邻居特征
            g.ndata['h'] = (1 + self.eps) * g.ndata['h'] + g.ndata['neigh']
            # 将更新后的节点特征通过MLP
            return self.mlp(g.ndata['h'])


class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.2, activation=F.relu, learn_eps=False):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()

        # 创建多个GIN层
        self.layers.append(GINLayer(in_dim, hidden_dim[0], learn_eps=learn_eps))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GINLayer(hidden_dim[i], hidden_dim[i + 1], learn_eps=learn_eps))

        # 最终的全连接层
        fc = [nn.Dropout(dropout)] if dropout > 0 else []
        fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self, data, random_smooth=False):
        x, B, N = self.process_batch(data, random_smooth)
        x = x.reshape(B, N, -1)
        x = torch.max(x, dim=1)[0].squeeze()
        x = self.fc(x)
        return x

    def get_embeddings(self, data, random_smooth=False):
        x, B, N = self.process_batch(data, random_smooth)
        return x.reshape(B, N, -1)

    def process_batch(self, data, random_smooth=False):
        '''
        data，一个包含图数据的列表，其中data[1]是邻接矩阵的列表，data[0]是节点特征的批次，data[2]是用于节点级别的mask。
        '''
        batch_g = []
        for adj in data[1]:
            # 转换成DGL图对象
            batch_g.append(numpy_to_graph(adj.cpu().detach().T.numpy(), to_cuda=adj.is_cuda, sample=random_smooth))
        # 将这些图对象批量化
        batch_g = dgl.batch(batch_g)
        # 在图神经网络的上下文中，mask通常用于指示每个图中有效节点的位置，以便在执行操作如汇聚（pooling）时忽略无效（或填充的）节点。
        mask = data[2]  # mask的形状为(B, N)，其中B是批次大小（即图的数量），N是节点的最大数量
        # 在mask张量的第三个维度上增加一个维度，将其形状从(B, N)变为(B, N, 1)。
        # 这样做的目的是为了确保mask的维度与图卷积网络处理的其他张量兼容，使得可以通过元素乘法（element-wise multiplication）等操作将mask应用于节点特征或其他相应的张量。
        '''
        在处理图数据时，特别是当以批处理方式处理多个图时，不同图可能具有不同数量的节点。为了能够将这些图放在同一个批次中处理，通常需要对节点进行填充以保持统一的维度。
        mask张量就是用来指示每个图中哪些节点是实际存在的，哪些是为了维度对齐而添加的填充节点。通过在计算中应用mask，模型可以仅对实际存在的节点进行操作，忽略填充节点，从而保持计算的准确性。
        例如，在进行节点级别的特征聚合或汇聚操作时，mask可以用来确保只有实际存在的节点的特征被考虑在内，而填充节点的特征被忽略。这对于维持图神经网络在处理不同结构图时的性能和准确度是至关重要的。
        '''
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)  # (B,N,1)
        '''
        B：批次大小（Batch size），即一次处理的图的数量。
        N：节点数（Number of nodes），每个图中的最大节点数量。
        F：特征数（Features per node），每个节点的特征向量的维度。
        '''
        B, N, F = data[0].shape[: 3]
        # 将原本按图组织的节点特征展平为一个二维数组，其中每行代表一个节点的特征向量，总共有B*N行，每行F个特征。这种展平操作是为了便于后续在单个大矩阵上执行图卷积操作，而不是在每个图上单独进行。
        x = data[0].reshape(B * N, F)
        # mask用于指示哪些节点是有效的（例如，在图中实际存在的节点），哪些节点是为了保持批次中图大小一致而添加的填充节点。
        mask = mask.reshape(B * N, 1)

        for layer in self.layers:
            # 将图结构和节点特征结合起来，为每个节点生成新的特征表示
            x = layer(batch_g, x)
            # 将前一步得到的节点特征与mask进行逐元素乘法（Element-wise multiplication）。这样做的目的是将那些填充的节点的特征值置零，确保它们不会影响后续的计算
            x = x * mask

        # 返回节点级的表示
        return x, B, N
