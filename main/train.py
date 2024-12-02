import sys, os
sys.path.append(os.path.abspath('..'))

import time
import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from utils.datareader import GraphData, DataReader
from utils.batch import collate_batch
from model.gcn import GCN
from model.gat import GAT
from model.sage import GraphSAGE
from model.gin import GIN
from config import parse_args
from utils.poisonNodeSelect import add_noise_to_graph
from trojan.prop import randomly_sample
import copy
import shap





def compute_saliency_and_shap(model, test_loader, args):
    model.eval()
    data = next(iter(test_loader))  # 获取一批测试数据

    # 解析数据（确保格式正确）
    node_features = data[0].to('cpu')  # 节点特征
    node_features.requires_grad = True

    # 前向传播，获取模型输出
    output = model(data, random_smooth=args.randomly_smooth)

    # 获取目标类别输出并计算梯度
    batch_size = output.shape[0]
    target_class = output.argmax(dim=1)
    selected_output = output[torch.arange(batch_size), target_class]
    selected_output.sum().backward()

    # 计算 Saliency Map：特征梯度的绝对值
    saliency = node_features.grad.abs().sum(dim=2).numpy()
    # 聚合每个节点的重要性
    node_scores = saliency.sum(axis=1)

    # 按照重要性排序，选择最重要的节点
    important_nodes = node_scores.argsort()[::-1]  # 从大到小排序
    print(important_nodes)

    return important_nodes, node_scores









def run(args):
    assert torch.cuda.is_available(), 'no GPU available'
    cpu = torch.device('cpu')
    cuda = torch.device(f'cuda:{args.device}')

    # load data into DataReader object
    dr = DataReader(args)

    loaders = {}
    for split in ['train', 'test']:
        if split == 'train':
            gids = dr.data['splits']['train']
        else:
            gids = dr.data['splits']['test']
        gdata = GraphData(dr, gids) # 加载一批数据
        # DataLoader为PyTorch提供的一个用于读取数据的工具，它能够自动对数据进行批处理
        loader = DataLoader(gdata,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_batch)
        # data in loaders['train/test'] is saved as returned format of collate_batch()
        loaders[split] = loader # 将train与test数据分别存储在loaders['train']与loaders['test']中
    print('train %d, test %d' % (len(loaders['train'].dataset), len(loaders['test'].dataset)))

    # prepare model
    in_dim = loaders['train'].dataset.num_features # 输入维度
    out_dim = loaders['train'].dataset.num_classes # 输出维度
    if args.model == 'gcn':
        model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.model == 'gat':
        model = GAT(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, num_head=args.num_head)
    elif args.model == 'sage':
        model = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.model == 'gin':
        model = GIN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        raise NotImplementedError(args.model)

    # print('\nInitialize model')
    # print(model)
    '''
    model.parameters()：这是一个生成器（generator），它会遍历模型中所有的参数。这包括权重（weights）和偏差（biases）等所有可训练的参数。
    lambda p: p.requires_grad：这是一个匿名函数，用于检查参数p是否需要计算梯度。
    在PyTorch中，如果一个参数的requires_grad属性设置为True，那么在反向传播时将会计算该参数的梯度。通常情况下，模型的可训练参数默认requires_grad就是True。
    filter(function, iterable)：filter函数接受两个参数，第一个是一个函数，第二个是一个可迭代对象。
    它会迭代第二个参数提供的每个元素，将其传递给第一个参数所指定的函数。如果这个函数返回True，则该元素会被包含在返回的迭代器中。
    list()：最后，使用list()函数将filter的结果转换为列表。这是因为filter函数返回的是一个迭代器，而优化器（如Adam）在初始化时需要参数列表。
    '''
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

    # training
    loss_fn = F.cross_entropy
    '''
    匿名函数predict_fn，用于从模型输出中提取预测结果
    output.max(1, keepdim=True)[1]：这个操作执行了两个主要步骤。首先，output.max(1, ...)在给定维度（这里是1，通常对应于模型输出的特征或类别维度）上查找每个样本最大的值。
    keepdim=True参数保持输出张量的维度不变，这有助于保持后续操作的维度一致性。
    output.max(...)函数返回一个元组，其中第一个元素是每行的最大值，第二个元素（[1]所获取的）是这些最大值所在的索引，即预测的类别。
    .detach()：此方法用于从当前计算图中分离出梯度信息。在预测时，通常不需要计算梯度，因此使用.detach()来避免梯度的计算和存储，从而节省内存和计算资源。这一步是优化性能的常见做法。
    .cpu()：将数据移动到CPU。如果模型和数据在GPU上，预测结果在处理或保存前通常需要被移动到CPU上。.cpu()方法确保了无论当前设备是什么，数据都会被转移到CPU上。
    '''
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    '''
    train_params：是一个包含模型参数的列表，这些参数是需要在训练过程中更新的。
    lr=args.lr：是学习率（Learning Rate），一个控制参数更新步长大小的超参数
    weight_decay=args.weight_decay：权重衰减（Weight Decay）是一种正则化技术，用于防止模型过拟合
    betas=(0.5, 0.999)：是Adam优化器的超参数，用于控制梯度的指数加权平均和平方梯度的指数加权平均
    '''
    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    '''
    创建了一个MultiStepLR学习率调度器，它用于在训练过程中按预定的多个步骤调整学习率。
    随着训练的进行，适时地减小学习率可以帮助模型更好地收敛，特别是当训练接近最优解时。
    optimizer：这是之前创建的Adam优化器，学习率调度器将会调整这个优化器的学习率。
    args.lr_decay_steps：这是一个确定学习率衰减发生步骤的列表或元组。它定义了在训练的哪些epoch学习率应该被减少。
        例如，如果它被设置为[30, 60]，这意味着在第30个和第60个epoch后学习率会被降低。
    gamma=0.1：这是学习率衰减的因子。在每个预定的衰减步骤，学习率会被乘以这个因子。
    '''
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
    
    model.to(cuda)
    for epoch in range(args.train_epochs):
        model.train() # model.train()：将模型设置为训练模式。这个方法通知模型在训练过程中需要计算梯度。
        start = time.time()
        train_loss, n_samples = 0, 0
        # loaders['train']是一个DataLoader实例，负责按批次提供训练数据。
        for batch_id, data in enumerate(loaders['train']):
            # if args.randomly_smooth:
            #     data[1] = randomly_sample(data[1], args)
            # TODO:随机平滑防御，对图添加噪声
            # data = add_noise_to_graph(copy.deepcopy(data), args)

            # 循环确保批次中的所有数据（如节点特征、邻接矩阵）都被转移到指定的设备
            for i in range(len(data)):
                data[i] = data[i].to(cuda)
            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])
            # 在新的迭代开始前清空（之前迭代的）梯度。
            optimizer.zero_grad()
            # 执行前向传播，计算模型对当前批次数据的输出。
            output = model(data, args.randomly_smooth)
            # 如果输出是一维的，需要将其转换为二维的。因为后续的损失函数可能预期输入是二维的（例如，[批次大小, 输出类别数]）。
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            # 计算模型输出和真实标签之间的损失。data[4]假设是批次中的标签。
            loss = loss_fn(output, data[4]) # loss_fn = F.cross_entropy
            loss.backward() # 执行反向传播，计算模型参数的梯度。
            optimizer.step() # 执行一步优化，根据计算出的梯度更新模型的参数。
            scheduler.step() # 根据预定的策略调整学习率。

            time_iter = time.time() - start
            train_loss += loss.item() * len(output)  # 累加批次的损失，以便计算整个周期的平均损失。
            n_samples += len(output) # 更新处理过的样本总数。

        # 确保只在特定的周期（由args.log_every确定）或在最后一个训练周期打印训练日志。
        if args.train_verbose and (epoch % args.log_every == 0 or epoch == args.train_epochs - 1):
            print('Train Epoch: %d\tLoss: %.4f (avg: %.4f) \tsec/iter: %.2f' % (
                epoch + 1, loss.item(), train_loss / n_samples, time_iter / (batch_id + 1)))



        # 在特定的周期（由args.eval_every确定）或在最后一个训练周期评估模型。
        if (epoch + 1) % args.eval_every == 0 or epoch == args.train_epochs-1:
            # 将模型设置为评估模式，关闭Dropout和BatchNorm层的随机行为。
            model.eval()
            start = time.time()
            test_loss, correct, n_samples = 0, 0, 0
            for batch_id, data in enumerate(loaders['test']):
                if args.randomly_smooth:
                    data[1] = randomly_sample(data[1], args)
                for i in range(len(data)):
                    data[i] = data[i].to(cuda)
                # if args.use_org_node_attr:
                #     data[0] = norm_features(data[0])
                output = model(data, args.randomly_smooth)
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                # 使用loss_fn计算输出和真实标签之间的损失。这里使用reduction='sum'参数来累加批次内的损失，以便最后计算整个测试集的平均损失。
                loss = loss_fn(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                # 使用predict_fn函数从模型输出中提取预测结果，然后将这些预测与实际标签进行比较以计算准确率。
                pred = predict_fn(output)

                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

            eval_acc = 100. * correct / n_samples
            print('Test set (epoch %d): Average loss: %.4f, Accuracy: %d/%d (%.2f%s) \tsec/iter: %.2f' % (
                epoch + 1, test_loss / n_samples, correct, n_samples, 
                eval_acc, '%', (time.time() - start) / len(loaders['test'])))
    
    model.to(cpu)

    # Compute Saliency Map and SHAP explanations on test data
    # compute_saliency_and_shap(model, loaders['test'], args)
    
    if args.save_clean_model:
        save_path = args.clean_model_save_path
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, '%s-%s-%s.t7' % (args.model, args.dataset, str(args.train_ratio)))
        
        torch.save({
                    'model': model.state_dict(),
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'eval_acc': eval_acc,
                    }, save_path)
        print('Clean trained GNN saved at: ', os.path.abspath(save_path))

    return dr, model


if __name__ == '__main__':
    args = parse_args()
    run(args)