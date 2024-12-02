import torch
import numpy as np

def gen_input(args, datareader, bkd_gids):
    """
    Prepare inputs for GTN, topo input and feat input together.
    
    About inputs (of this function):
    - args: control adapt-input type 
    
    Note: Extend input size as (N, N) / (N, F) where N is max node num among all graphs

    args: 包含控制参数，特别是 gtn_input_type 决定了图的输入类型。
    datareader: 数据读取器，包含图数据，如邻接列表和节点特征。
    bkd_gids: 图的标识符集合，指定哪些图将被处理。
    """
    As = {}
    Xs = {}
    # 从 datareader 中提取每个图的邻接矩阵 (As) 和特征矩阵 (Xs)
    for gid in bkd_gids:
        if gid not in As: As[gid] = torch.tensor(datareader.data['adj_list'][gid], dtype=torch.float)
        if gid not in Xs: Xs[gid] = torch.tensor(datareader.data['features'][gid], dtype=torch.float)

    Ainputs = {}
    Xinputs = {}
    # 1hop: 直接使用邻接矩阵和特征矩阵
    if args.gtn_input_type == '1hop':
        for gid in bkd_gids:
            if gid not in Ainputs: Ainputs[gid] = As[gid].clone().detach()
            if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid], Xs[gid])

    # 2hop: 计算邻接矩阵的二次方，用于表示两跳邻居的连接。
    # 将结果转换为0或1的形式，以表示连接的存在或不存在，并清除对角线上的值（移除自环）。
    elif args.gtn_input_type == '2hop':
        for gid in bkd_gids:
            As[gid] = torch.add(As[gid], torch.mm(As[gid], As[gid]))
            As[gid] = torch.where(As[gid]>0, torch.tensor(1.0, requires_grad=True),
                                             torch.tensor(0.0, requires_grad=True))
            As[gid].fill_diagonal_(0.0)
            
        for gid in bkd_gids:
            if gid not in Ainputs: Ainputs[gid] = As[gid].clone().detach()
            if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid], Xs[gid])
    
    # 直接使用邻接矩阵和特征矩阵,额外考虑了节点的度（通过邻接矩阵的行和来计算）进行归一化，用于处理图中的节点度分布
    elif args.gtn_input_type == '1hop_degree': 
        rowsums = [torch.add(torch.sum(As[gid], dim=1), 1e-6) for gid in bkd_gids]
        re_Ds = [torch.diag(torch.pow(rowsum, -1)) for rowsum in rowsums]
        
        for i in range(len(bkd_gids)):
            gid = bkd_gids[i]
            if gid not in Ainputs: Ainputs[gid] = torch.mm(re_Ds[i], As[gid])
            if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid], Xs[gid])
                
    # 计算邻接矩阵的二次方,额外考虑了节点的度（通过邻接矩阵的行和来计算）进行归一化，用于处理图中的节点度分布
    elif args.gtn_input_type == '2hop_degree':
        for gid in bkd_gids:
            As[gid] = torch.add(As[gid], torch.mm(As[gid], As[gid]))
            As[gid] = torch.where(As[gid]>0, torch.tensor(1.0, requires_grad=True),
                                             torch.tensor(0.0, requires_grad=True))
            As[gid].fill_diagonal_(0.0)
            
        rowsums = [torch.add(torch.sum(As[gid], dim=1), 1e-6) for gid in bkd_gids]
        re_Ds = [torch.diag(torch.pow(rowsum, -1)) for rowsum in rowsums]
        
        for i in range(len(bkd_gids)):
            gid = bkd_gids[i]
            if gid not in Ainputs: Ainputs[gid] = torch.mm(re_Ds[i], As[gid])
            if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid], Xs[gid])
                                
    else: raise NotImplementedError('not support other types of aggregated inputs')

    # pad each input into maxi possible size (N, N) / (N, F)
    # 确定所有图中的最大节点数 (NodeMax) 和特征维度 (FeatDim)，以便将所有输入矩阵填充到这个最大尺寸。
    NodeMax = int(datareader.data['n_node_max'])
    FeatDim = np.array(datareader.data['features'][0]).shape[1]
    for gid in Ainputs.keys():
        a_input = Ainputs[gid]
        x_input = Xinputs[gid]
        
        add_dim = NodeMax - a_input.shape[0]
        # 使用 np.pad 对邻接矩阵和特征矩阵进行填充，确保它们的尺寸为 (N, N) 或 (N, F)，其中 N 是最大节点数，F 是特征维度。
        Ainputs[gid] = np.pad(a_input, ((0, add_dim), (0, add_dim))).tolist()
        Xinputs[gid] = np.pad(x_input, ((0, add_dim), (0, 0))).tolist()
        Ainputs[gid] = torch.tensor(Ainputs[gid])
        Xinputs[gid] = torch.tensor(Xinputs[gid])
    # 函数返回两个字典，分别包含处理并填充后的邻接矩阵和特征矩阵，键为图的标识符。
    return Ainputs, Xinputs
    