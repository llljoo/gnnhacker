import sys, os
sys.path.append(os.path.abspath('..'))

import torch, gc
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from utils.datareader import GraphData, DataReader
from utils.batch import collate_batch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from tqdm import tqdm

'''
evaluate 函数提供了一个更全面的性能评估，包括准确率和置信度，而forwarding函数则更注重于损失的计算和可能的一次性前向传播需求。
'''

# run on CUDA
# 一个模型前向传播的过程，用于评估模型在一个数据集上的性能
def forwarding(args, bkd_dr: DataReader, model, gids, criterion, flag='topo'):
    # model为原始GNN，bkd_dr为后门训练数据集，gids为训练数据集的id，criterion为交叉熵损失函数
    assert torch.cuda.is_available(), "no GPU available"
    cuda = torch.device(f'cuda:{args.device}')
    gc.collect()

    
    gdata = GraphData(bkd_dr, gids) # 创建一个图数据对象
    # 创建一个数据加载器，用于批量加载图数据
    loader = DataLoader(gdata,
                        batch_size=args.batch_size,
                        shuffle=False,   
                        collate_fn=collate_batch)
    # 如果模型的参数不在CUDA设备上，则将模型移动到CUDA设备上。
    if not next(model.parameters()).is_cuda:
        model.to(cuda)

    model.eval() # 将模型设置为评估模式

    all_loss, n_samples = 0.0, 0.0
    loss_sim = torch.tensor(0.0)
    # 遍历后门数据集，计算用GNN模型预测的结果与真实标签之间的损失
    for batch_idx, data in enumerate(loader):
        if args.randomly_smooth:
            # TODO:在这里对data进行下采样，要完成d次下采样
            data[1] = randomly_sample(data[1], args)

        torch.cuda.empty_cache()

#         assert batch_idx == 0, "In AdaptNet Train, we only need one GNN pass, batch-size=len(all trainset)"
        # TODO：添加拓扑生成约束，在forwarding过程中添加，修改损失计算规则
        # 计算节点之间的余弦相似度
        # 使用余弦相似性计算特征相似性矩阵
        if flag == 'topo':
            for i in range(len(data[0])):
                features_np = data[0][i]
                similarity_matrix = torch.tensor(cosine_similarity(features_np))
                loss_fn = torch.nn.MSELoss()
                loss_sim += loss_fn(similarity_matrix, data[1][i])


        # # TODO：添加特征生成约束，在forwarding过程中添加，修改损失计算规则
        # # 使用结构保留损失函数计算特征生成损失（让相邻节点的具有更相似的特征）
        # cnt = 0
        # if flag == 'feat':
        #     for i in range(len(data[0])):
        #         features = data[0][i]
        #         adj_matrix = data[1][i]
        #         n = features.size(0)
        #         for k in range(n):
        #             for j in range(k + 1, n):
        #                 if adj_matrix[k, j] > 0:  # 只有邻接矩阵中i和j是相连的，才计算loss
        #                     loss_sim += torch.nn.functional.mse_loss(features[k], features[j], reduction='mean')
        #                     cnt += 1
        #         loss_sim = torch.div(loss_sim, cnt)



        # 将每个数据元素移动到CUDA设备上
        for i in range(len(data)):
            data[i] = data[i].to(cuda)
        output = model(data, args.randomly_smooth) # 使用模型对调整后的数据进行前向传播，得到输出
        
        if len(output.shape) == 1:
            output = output.unsqueeze(0)

        # 使用提供的损失准则计算损失值
        loss = criterion(output, data[4])  # only calculate once
        # 更新累加的总损失和样本数量
        all_loss = torch.add(torch.mul(loss, len(output)), all_loss)  # cannot be loss.item()
        n_samples += len(output)

    # 计算所有样本的平均损失
    all_loss = torch.div(all_loss + loss_sim, n_samples)
    return all_loss


def train_model(args, dr_train: DataReader, model, pset, nset):
    assert torch.cuda.is_available(), "no GPU available"

    cuda = torch.device(f'cuda:{args.device}')
    cpu = torch.device('cpu')
    model.to(cuda)


    gids = {'pos': pset, 'neg': nset}
    gdata = {}
    loader = {}
    for key in ['pos', 'neg']:
        gdata[key] = GraphData(dr_train, gids[key]) # 创建一个图数据对象，用于存储当前类别的样本数据。
        # 创建一个数据加载器，用于批量加载当前类别的样本数据。
        loader[key] = DataLoader(gdata[key],
                                batch_size=args.batch_size,
                                shuffle=False,   
                                collate_fn=collate_batch)


    # 获取模型中所有需要梯度更新的参数。
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # 创建一个Adam优化器，用于优化模型参数
    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    # 创建一个学习率调度器，用于在特定步骤调整学习率
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)

    final_loss = None
    for epoch in range(args.train_epochs):
        loss = train(args, model, optimizer, epoch, loader)
        scheduler.step() # 调整学习率
        final_loss = loss  # 记录最后一个 epoch 的损失
    model.to(cpu)
    return final_loss  # 返回最后一个 epoch 的损失



def train(args, model, optimizer, epoch, loader):
    cuda = torch.device(f'cuda:{args.device}')
    cpu = torch.device('cpu')
    # 定义交叉熵损失函数
    loss_fn = F.cross_entropy
    model.train()


    losses = {'pos': 0.0, 'neg': 0.0}  # 存储每类样本的损失
    n_samples = {'pos': 0.0, 'neg': 0.0}  # 存储每类样本的样本数量
    loss_accsum = 0

    pbar = tqdm(range(args.iter_per_epoch), unit='batch')

    for pos in pbar:
        for key in ['pos', 'neg']:
            for batch_idx, data in enumerate(loader[key]):
                if args.randomly_smooth:
                    # TODO:在这里对data进行下采样，要完成d次下采样
                    data[1] = randomly_sample(data[1], args)
                torch.cuda.empty_cache()

                # 将数据移动到GPU上
                for i in range(len(data)):
                    data[i] = data[i].to(cuda)
                output = model(data, args.randomly_smooth)  # 计算模型输出
                if len(output.shape) == 1:
                    # [N] 的一维数据，转换为 [1, N] 的二维数据
                    output = output.detach().unsqueeze(0)
                # 计算损失，并累加到相应类别的总损失中
                losses[key] += loss_fn(output.detach(), data[4].detach()) * len(output.detach())
                # 计算该类别的总样本数
                n_samples[key] += len(output.detach())
                # 将数据移回CPU
                for i in range(len(data)):
                    data[i] = data[i].to(cpu)

            # 对每一类样本，计算平均损失
            losses[key] = torch.div(losses[key], n_samples[key])
        # 计算总损失，考虑正负样本的损失和正样本损失的权重
        loss = losses['pos'] + args.lambd * losses['neg']
        loss_accsum += loss.item()
        if optimizer is not None:
            optimizer.zero_grad()  # 清零梯度
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)  # 执行反向传播
            optimizer.step()  # 更新模型参数

        pbar.set_description('epoch: %d' % (epoch))
    average_loss = loss_accsum / args.iter_per_epoch
    print("train GNN avg loss: %f" % (average_loss))

    return torch.tensor(average_loss, requires_grad=True)



def randomly_sample(adj_list, args):
    '''
    随机下采样数据,data[0]是X，data[1]是A，data[2]是graph_support，data[3]是nodenums，data[4]是labels
    采样方法：
    1、对边进行采样，按照比例直接对边进行下采样
    2、对节点进行采样，被采样的节点的相连边会被删除，节点数量不变，以孤立节点的方式阻止信息传递
    '''
    if args.sample_method == 'edge':
        for i in range(len(adj_list)):
            adj = adj_list[i].detach()
            n = adj_list[i].shape[0]
            k = int(args.randomly_preserve * n * (n - 1) / 2)
            mask = matrix_sample(n, k)
            # tmp1 = adj.numpy()
            # tmp = adj.numpy() * mask
            adj_list[i] = torch.tensor(adj.numpy() * mask)
            # tmp2 = data[1][i].numpy()



    elif args.sample_method == 'node':
        for i in range(len(adj_list)):
            adj = adj_list[i]
            n = adj_list[i].shape[0]
            k = int(args.randomly_preserve * n)
            mask = np.ones((n, n))
            # 随机选择 k 个节点
            selected_indices = np.random.choice(n, n - k, replace=False)
            # 将这些节点相连的边位置标记为 0
            for idx in selected_indices:
                mask[idx, : ] = 0
                mask[ : ,idx] = 0  # 如果是无向图，也要标记对称位置为 0
            adj_list[i] = torch.tensor(adj.numpy() * mask)

    return adj_list


def matrix_sample(n, k):
    # n*n symmetrix matrix sample k units
    random.seed(1)
    sample_list = random.sample(range(1, int(n * (n - 1) / 2 + 1)), k)
    A = np.zeros((n, n))
    for m in sample_list:
        i = 0
        l = n - 1
        while (m > l):
            m -= l
            i += 1
            l -= 1
        j = i + m
        A[i][j] = A[j][i] = 1
    return A



# def TrainGNN_v2(args,
#              dr_train,
#              model,
#              fold_id,
#              train_gids,
#              use_optim='Adam',
#              need_print=False):
#     assert torch.cuda.is_available(), "no GPU available"
#     cuda = torch.device('cuda')
#     cpu = torch.device('cpu')
                       
#     model.to(cuda)
                       
#     gdata = GraphData(dr_train,
#                       fold_id,
#                       'train',
#                       train_gids)
#     loader = DataLoader(gdata,
#                         batch_size=args.batch_size,
#                         shuffle=False,   
#                         collate_fn=collate_batch)
    
#     train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
#     if use_optim=='Adam':
#         optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
#     else:
#         optimizer = optim.SGD(train_params, lr=args.lr)
#     predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
#     loss_fn = F.cross_entropy

#     model.train()
#     for epoch in range(args.epochs):
#         optimizer.zero_grad()
        
#         loss = 0.0
#         n_samples = 0
#         correct = 0
#         for batch_idx, data in enumerate(loader):
#             for i in range(len(data)):
#                 data[i] = data[i].to(cuda)
#             output = model(data)
#             if len(output.shape)==1:
#                 output = output.unsqueeze(0)
#             loss += loss_fn(output, data[4])*len(output)
#             n_samples += len(output)

#             for i in range(len(data)):
#                 data[i] = data[i].to(cpu)
#             torch.cuda.empty_cache()
            
#             pred = predict_fn(output)
#             correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
#         acc = 100. * correct / n_samples
#         loss = torch.div(loss, n_samples)
        
#         if need_print and epoch%5==0:
#             print("Epoch {} | Loss {:.4f} | Train Accuracy {:.4f}".format(epoch, loss.item(), acc))
#         loss.backward()
#         optimizer.step()
#     model.to(cpu)


# 用于评估模型在测试集上的性能
# 计算和报告模型在给定测试数据上的平均损失、准确率和平均置信度
def evaluate(args, dr_test: DataReader, model, gids):  
    # separate bkd_test/clean_test gids
    # 使用Softmax函数处理模型输出，它将模型的原始输出转换为概率分布
    softmax = torch.nn.Softmax(dim=1)
    
    model.cuda() # 将模型移至CUDA设备以利用GPU加速
    gdata = GraphData(dr_test, gids) # 创建图数据对象
    # 创建数据加载器
    loader = DataLoader(gdata,
                        batch_size=args.batch_size,
                        shuffle=False,   
                        collate_fn=collate_batch)
    
    loss_fn = F.cross_entropy # 定义交叉熵损失函数
    # 基于模型的输出确定每个样本的预测类别
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    
    model.eval() # 将模型设置为评估模式
    test_loss, correct, n_samples, confidence = 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            loss = torch.tensor(0.0)
            if args.randomly_smooth:
                data[1] = randomly_sample(data[1], args)

            for i in range(len(data)):
                data[i] = data[i].cuda()
            output = model(data, args.randomly_smooth)  # not softmax yet
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            loss += loss_fn(output.detach().cpu(), data[4].detach().cpu(), reduction='sum')  # 计算损失
            test_loss += loss.item() # 计算并累加损失
            n_samples += len(output) # 计算总样本数
            pred = predict_fn(output) # 获取预测类别

            correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item() # 计算并累加正确预测的数量
            confidence += torch.sum(torch.max(softmax(output), dim=1)[0]).item() # 计算并累加平均置信度

    acc = 100. * correct / n_samples # 计算准确率
    confidence = confidence / n_samples # 计算平均置信度
    
    print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.2f%s), Average Confidence %.4f' % (
        test_loss / n_samples, correct, n_samples, acc, '%', confidence))
    model.cpu()
    return acc