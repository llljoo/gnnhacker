import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from utils.graph import numpy_to_graph

# 直接将源节点（发送消息的节点）的特征复制作为消息。在这里，它将节点数据'h'（源特征）复制到消息'm'中。简单来说，每个节点会将自己的特征作为消息发送给邻居节点。
gcn_msg = fn.copy_src(src='h', out='m') # DGL内建的消息传递函数
# 将所有进入目标节点的消息（'m'）进行求和操作，然后将结果保存到目标节点的特征'h'中。这意味着每个节点会收集其所有邻居节点发送的特征信息，并将它们累加起来以更新自己的状态。
gcn_reduce = fn.sum(msg='m', out='h') # DGL内建的聚合函数

# Used for inductive case (graph classification) by default.
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        '''
        g：一个DGL图对象，表示当前处理的图。
        feature：图中每个节点的特征矩阵。
        '''
        # 确保在这个forward方法执行期间对图g所做的任何修改（比如添加节点数据'h'）都不会影响到外部环境。当退出这个作用域时，这些临时修改会被自动清除。
        with g.local_scope():
            # 将输入特征矩阵赋值给图的节点数据'h'
            g.ndata['h'] = feature
            # 使用两个函数gcn_msg和gcn_reduce来执行图卷积操作。
            # gcn_msg定义了如何从源节点向目标节点发送消息（即如何计算边上的信息）
            # gcn_reduce定义了如何在目标节点聚合消息（即如何使用邻居节点的信息更新自己的信息）
            g.update_all(gcn_msg, gcn_reduce)
            # 在完成消息传递和聚合后，h = g.ndata['h']获取更新后的节点特征
            h = g.ndata['h']
            # 通过之前定义的线性层进一步转换这些特征，从而生成最终的输出特征矩阵
            return self.linear(h)


# 2 layers by default
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim,
                 hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
                 dropout=0.2,
                 activation=F.relu):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNLayer(hidden_dim[i], hidden_dim[i+1]))
    
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))

        '''
        self.fc = nn.Sequential(*fc)：这行代码使用nn.Sequential构造了一个序列模型。
        nn.Sequential是PyTorch中的一个容器，按照其中模块的顺序进行前向传播计算。使用*fc将fc列表中的所有元素（在这个例子中只有一个全连接层，如果有dropout则为两个元素）
        解包作为参数传递给nn.Sequential。这样构建的self.fc序列模型可以被直接用于前向传播，处理从图卷积层来的特征，生成最终的预测输出。
        
        在图神经网络中，经过一系列图卷积层处理后，节点的特征被聚合（比如通过平均或最大池化）成图级别的表示。
        这个图级别的表示包含了整个图的信息，可以用于图分类、图回归等任务。self.fc全连接层负责将这个图级别的表示映射到最终的输出空间（比如分类任务中的类别标签）。
        这是模型架构中的一个关键步骤，因为它直接关联到模型的预测性能。
        '''
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
        mask = data[2] # mask的形状为(B, N)，其中B是批次大小（即图的数量），N是节点的最大数量
        # 在mask张量的第三个维度上增加一个维度，将其形状从(B, N)变为(B, N, 1)。
        # 这样做的目的是为了确保mask的维度与图卷积网络处理的其他张量兼容，使得可以通过元素乘法（element-wise multiplication）等操作将mask应用于节点特征或其他相应的张量。
        '''
        在处理图数据时，特别是当以批处理方式处理多个图时，不同图可能具有不同数量的节点。为了能够将这些图放在同一个批次中处理，通常需要对节点进行填充以保持统一的维度。
        mask张量就是用来指示每个图中哪些节点是实际存在的，哪些是为了维度对齐而添加的填充节点。通过在计算中应用mask，模型可以仅对实际存在的节点进行操作，忽略填充节点，从而保持计算的准确性。
        例如，在进行节点级别的特征聚合或汇聚操作时，mask可以用来确保只有实际存在的节点的特征被考虑在内，而填充节点的特征被忽略。这对于维持图神经网络在处理不同结构图时的性能和准确度是至关重要的。
        '''
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2) # (B,N,1)  
        '''
        B：批次大小（Batch size），即一次处理的图的数量。
        N：节点数（Number of nodes），每个图中的最大节点数量。
        F：特征数（Features per node），每个节点的特征向量的维度。
        '''
        B, N, F = data[0].shape[: 3]
        # 将原本按图组织的节点特征展平为一个二维数组，其中每行代表一个节点的特征向量，总共有B*N行，每行F个特征。这种展平操作是为了便于后续在单个大矩阵上执行图卷积操作，而不是在每个图上单独进行。
        x = data[0].reshape(B*N, F)
        # mask用于指示哪些节点是有效的（例如，在图中实际存在的节点），哪些节点是为了保持批次中图大小一致而添加的填充节点。
        mask = mask.reshape(B*N, 1)

        for layer in self.layers:
            # 将图结构和节点特征结合起来，为每个节点生成新的特征表示
            x = layer(batch_g, x)
            # 将前一步得到的节点特征与mask进行逐元素乘法（Element-wise multiplication）。这样做的目的是将那些填充的节点的特征值置零，确保它们不会影响后续的计算
            x = x * mask

        # 返回节点级的表示
        return x, B, N
    
