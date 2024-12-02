import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph import numpy_to_graph

# implemented from https://arxiv.org/abs/1710.10903

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        g = g.to(h.device)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')#, g.edata['e']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_head, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_head):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs), dim=0)


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, 
                hidden_dim=[64, 32],
                dropout=0.2,
                num_head=2):
        super(GAT, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(MultiHeadGATLayer(in_dim, hidden_dim[0], num_head, merge='mean'))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(MultiHeadGATLayer(hidden_dim[i], hidden_dim[i+1], num_head, merge='mean'))
    
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
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
        for adj in data[1]:
            batch_g.append(numpy_to_graph(adj.cpu().detach().T.numpy(), to_cuda=adj.is_cuda, sample=random_smooth))
        batch_g = dgl.batch(batch_g)
        
        mask = data[2]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2) # (B,N,1)  
        
        B, N, F = data[0].shape[:3]
        x = data[0].reshape(B*N, F)
        mask = mask.reshape(B*N, 1)
        for layer in self.layers:
            x = layer(batch_g, x)
            x = x * mask
        

        return x, B, N