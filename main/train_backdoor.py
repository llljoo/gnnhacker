import sys, os

sys.path.append(os.path.abspath('..'))

import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from utils.datareader import DataReader
from utils.bkdcdd import select_cdd_graphs, select_cdd_nodes, select_poisoned_nodes, select_poisoned_nodes_2
from utils.mask import gen_mask, recover_mask
import main.benign as benign
import trojan.GTA as gta
from trojan.input import gen_input
from trojan.prop import train_model, evaluate
from config import parse_args
from model.gcn import GCN
from model.gat import GAT
from model.sage import GraphSAGE
from model.gin import GIN
from utils.datareader import GraphData, DataReader
from utils.batch import collate_batch
from torch.utils.data import DataLoader
import time

os.environ["OMP_NUM_THREADS"] = '1'


class GraphBackdoor:
    def __init__(self, args) -> None:
        '''
        初始化 GraphBackdoor 实例时，要求系统必须有 GPU 支持（使用 torch.cuda.is_available() 检查）。
        定义两个设备：CPU 和 GPU，用于在必要时将数据或模型移动到适当的设备上。
        '''
        self.args = args

        assert torch.cuda.is_available(), 'no GPU available'
        self.cpu = torch.device('cpu')
        self.cuda = torch.device('cuda')

    def run(self):
        '''
        这是类的核心方法，实现了后门攻击的完整流程：
        '''
        # train a benign GNN / 首先训练一个未经篡改的良性图形神经网络模型。
        '''
        通过调用 benign.run(self.args) 来训练一个未经篡改的良性图形神经网络（GNN）模型。
        这个步骤返回两个结果：一个是数据读取器 self.benign_dr，它包含了训练数据；另一个是训练好的模型 self.benign_model。
        '''
        # self.benign_dr, self.benign_model = benign.run(self.args)
        # model = copy.deepcopy(self.benign_model).to(self.cuda)

        assert torch.cuda.is_available(), 'no GPU available'

        # load data into DataReader object
        self.benign_dr = DataReader(args)
        loaders = {}
        for split in ['train', 'test']:
            if split == 'train':
                gids = self.benign_dr.data['splits']['train']
            else:
                gids = self.benign_dr.data['splits']['test']
            gdata = GraphData(self.benign_dr, gids)  # 加载一批数据
            # DataLoader为PyTorch提供的一个用于读取数据的工具，它能够自动对数据进行批处理
            loader = DataLoader(gdata,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=collate_batch)
            # data in loaders['train/test'] is saved as returned format of collate_batch()
            loaders[split] = loader  # 将train与test数据分别存储在loaders['train']与loaders['test']中
        print('train %d, test %d' % (len(loaders['train'].dataset), len(loaders['test'].dataset)))

        # prepare model
        in_dim = loaders['train'].dataset.num_features  # 输入维度
        out_dim = loaders['train'].dataset.num_classes  # 输出维度
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

        # 遍历良性数据读取器中的所有邻接矩阵，计算出所有图中的最大节点数 nodemax 和特征的维度 featdim。这两个值用于初始化攻击生成器的输入维度。
        nodenums = [adj.shape[0] for adj in self.benign_dr.data['adj_list']]
        nodemax = max(nodenums)
        featdim = np.array(self.benign_dr.data['features'][0]).shape[1]

        '''初始化触发器生成器'''
        # init two generators for topo/feat
        # 一个用于生成拓扑（结构）攻击掩码的图形Trojan网络，其输入维度为最大节点数 nodemax，层数由 self.args.gtn_layernum 指定
        toponet = gta.GraphTrojanNet(nodemax, self.args.gtn_layernum)
        # 一个用于生成特征攻击掩码的图形Trojan网络，其输入维度为特征维度 featdim，层数也由 self.args.gtn_layernum 指定
        featnet = gta.GraphTrojanNet(featdim, self.args.gtn_layernum)

        # TODO:加载训练好的模型
        real_model_path = '../save/model/clean/' + args.real_model + '-' + args.dataset + '-' + str(
            args.train_ratio) + '.t7'

        if args.real_model == 'gcn':
            real_model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        elif args.real_model == 'gat':
            real_model = GAT(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
                             num_head=args.num_head)
        elif args.real_model == 'sage':
            real_model = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        elif args.real_model == 'gin':
            real_model = GIN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        checkpoint = torch.load(real_model_path)

        # 提取模型参数
        model_state_dict = checkpoint['model']
        # 提取其他信息
        lr = checkpoint['lr']
        batch_size = checkpoint['batch_size']
        eval_acc = checkpoint['eval_acc']

        # 加载模型参数
        real_model.load_state_dict(model_state_dict)

        self.benign_model = real_model

        for rs_step in range(self.args.resample_steps):  # for each step, choose different samples
            # randomly select new graph backdoor samples
            # 从训练集中随机选择图作为后门攻击的目标
            bkd_gids_train, bkd_nids_train, bkd_nid_groups_train = self.bkd_cdd('train', args.select_method)

            # positive/negtive sample set
            # 构建正样本集 (pset) 和负样本集 (nset)。正样本集包含后门图ID，而负样本集包含训练集中未被选为后门目标的图。
            # 如果设置了正负样本比率 (pn_rate)，则相应地调整正负样本的数量，保证它们之间的比例。
            pset = bkd_gids_train
            nset = list(set(self.benign_dr.data['splits']['train']) - set(pset))

            # 检查是否设置了正负样本比例，如果设置了，则调整正负样本的数量
            if self.args.pn_rate != None:
                if len(pset) > len(nset):
                    # 如果正样本集 (pset) 的大小大于负样本集 (nset) 的大小，计算需要重复负样本集多少次以达到指定的比例。
                    # 计算公式为 ceil(len(pset) / (len(nset) * pn_rate))，意味着将负样本集重复足够次数，使得负样本集的大小至少与正样本集大小乘以比例的倒数相等。
                    # 使用 ceil 函数确保重复次数为整数，并且负样本集大小不小于所需的最小值。
                    repeat = int(np.ceil(len(pset) / (len(nset) * self.args.pn_rate)))
                    nset = list(nset) * repeat
                else:
                    # 如果负样本集的大小大于或等于正样本集的大小，则计算需要将正样本集重复的次数，
                    # 这次是通过 len(nset) * self.args.pn_rate / len(pset) 计算得出。
                    # 重复次数也是向上取整，然后将正样本集重复指定的次数以增加其大小
                    repeat = int(np.ceil((len(nset) * self.args.pn_rate) / len(pset)))
                    pset = list(pset) * repeat

            # init train data
            # NOTE: for data that can only add perturbation on features, only init the topo value
            # 通过 self.init_trigger 方法初始化含后门的训练数据，并生成对应的拓扑和特征掩码及输入数据。
            init_dr_train = self.init_trigger(self.args, copy.deepcopy(self.benign_dr), bkd_gids_train,
                                              bkd_nid_groups_train, 0.0, 0.0)
            bkd_dr_train = copy.deepcopy(init_dr_train)

            # 生成拓扑（结构）和特征的掩码，这些掩码指示了在哪些位置可以对图的结构和节点的特征进行修改，以嵌入后门触发器
            topomask_train, featmask_train = gen_mask(init_dr_train, bkd_gids_train, bkd_nid_groups_train)
            # 根据注入了后门触发器的数据 init_dr_train 和后门图ID列表 bkd_gids_train，准备好供生成器使用的输入数据
            Ainput_train, Xinput_train = gen_input(self.args, init_dr_train, bkd_gids_train)

            '''
            第一层涉及到通过toponet和featnet生成修改后的图数据，旨在优化生成器以产生对模型训练最有影响的图变化。
            第二层是基于这些生成的数据训练得到一个后门GNN模型，目的是提高模型在面对修改后的数据时的性能或鲁棒性。

            # 在每次重采样中执行 self.args.bilevel_steps 次双层优化步骤。每步中，使用训练好的生成器（toponet 和 featnet）修改训练数据的结构和特征，然后训练后门模型
            for bi_step in range(self.args.bilevel_steps):
                # 打印当前的重采样步骤（rs_step）和双层优化步骤（bi_step）
                print("Resampling step %d, bi-level optimization step %d" % (rs_step, bi_step))

                # step1：调用train_gtn函数训练拓扑和特征生成器。这一步是用于优化生成器，以便它们能够更好地修改图数据，实现目标后门攻击或其他目标。
                toponet, featnet = gta.train_gtn(self.args, model, toponet, featnet, pset, nset, topomask_train,
                                                 featmask_train,
                                                 init_dr_train, bkd_dr_train, Ainput_train, Xinput_train)

                # get new backdoor datareader for training based on well-trained generators
                # 循环遍历后门训练集bkd_gids_train中的每个图ID
                for gid in bkd_gids_train:
                    #使用上层训练好的触发器结构生成模型 和 触发器特征生成模型，为后门训练集中的每个图生成新的拓扑结构和特征，也即更新子图触发器过程
                    # 利用训练好的toponet和featnet为每个图生成新的拓扑结构和特征
                    # 生成的新拓扑结构和特征是通过调用toponet和featnet的输出得到的，然后与原始数据相加，更新bkd_dr_train数据读取器中的adj_list和features。
                    # 使用.detach().cpu()是为了确保生成的数据从计算图中分离，并移动到CPU上，这样做通常是为了避免在未来的计算中保留梯度信息，也为了确保数据能够在不支持CUDA的环境中使用。
                    rst_bkdA = toponet(Ainput_train[gid], topomask_train[gid], self.args.topo_thrd, self.cpu,
                                       self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_train[gid], 'topo')
                    # bkd_dr_train.data['adj_list'][gid] = torch.add(rst_bkdA, init_dr_train.data['adj_list'][gid])
                    bkd_dr_train.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]].detach().cpu(), init_dr_train.data['adj_list'][gid])

                    rst_bkdX = featnet(Xinput_train[gid], featmask_train[gid], self.args.feat_thrd, self.cpu,
                                       self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_train[gid], 'feat')
                    # bkd_dr_train.data['features'][gid] = torch.add(rst_bkdX, init_dr_train.data['features'][gid])
                    bkd_dr_train.data['features'][gid] = torch.add(rst_bkdX[:nodenums[gid]].detach().cpu(),
                                                                   init_dr_train.data['features'][gid])

                # step2：train GNN，model是benign的GNN模型，bkd_dr_train是后门训练数据读取器，pset和nset是正负样本集合
                #更新了子图触发器后，用该批数据继续训练GNN模型
                train_model(self.args, bkd_dr_train, model, list(set(pset)), list(set(nset)))
            '''
            ####################################################################################################
            optimizer = torch.optim.Adam(
                list(toponet.parameters()) + list(featnet.parameters()) + list(model.parameters()),
                lr=self.args.lr
            )

            # 联合优化步骤，代替双层优化
            for joint_step in range(self.args.bilevel_steps):  # 将双层循环改为单一循环
                start_time_joint = time.time()  # 开始计时
                print("Resampling step %d, joint optimization step %d" % (rs_step, joint_step))

                optimizer.zero_grad()  # 清空梯度

                # 更新后门图的拓扑和特征
                for gid in bkd_gids_train:
                    rst_bkdA = toponet(Ainput_train[gid], topomask_train[gid], self.args.topo_thrd, self.cpu,
                                       self.args.topo_activation, 'topo')
                    bkd_dr_train.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]].detach().cpu(),
                        torch.as_tensor(init_dr_train.data['adj_list'][gid]))

                    rst_bkdX = featnet(Xinput_train[gid], featmask_train[gid], self.args.feat_thrd, self.cpu,
                                       self.args.feat_activation, 'feat')
                    bkd_dr_train.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]].detach().cpu(), torch.as_tensor(init_dr_train.data['features'][gid]))

                # 在更新后的后门数据上训练GNN模型
                loss = train_model(self.args, bkd_dr_train, model, list(set(pset)), list(set(nset)))
                loss.backward()  # 反向传播
                optimizer.step()  # 更新所有参数

                end_time_joint = time.time()  # 结束计时
                joint_duration = end_time_joint - start_time_joint  # 计算时间
                print(f"一轮联合优化耗时: {joint_duration:.2f} 秒")

                # 评估模型在测试集上的攻击效果和准确率
                if joint_step == self.args.bilevel_steps - 1 or abs(loss.item() - 0) < 1e-4:
                    # 获取后门测试集和掩码
                    bkd_gids_test, bkd_nids_test, bkd_nid_groups_test = self.bkd_cdd('test', args.select_method)
                    init_dr_test = self.init_trigger(self.args, copy.deepcopy(self.benign_dr), bkd_gids_test,
                                                     bkd_nid_groups_test, 0.0, 0.0)
                    bkd_dr_test = copy.deepcopy(init_dr_test)
                    topomask_test, featmask_test = gen_mask(init_dr_test, bkd_gids_test, bkd_nid_groups_test)
                    Ainput_test, Xinput_test = gen_input(self.args, init_dr_test, bkd_gids_test)

                    for gid in bkd_gids_test:
                        rst_bkdA = toponet(Ainput_test[gid], topomask_test[gid], self.args.topo_thrd, self.cpu,
                                           self.args.topo_activation, 'topo')
                        bkd_dr_test.data['adj_list'][gid] = torch.add(
                            rst_bkdA[:nodenums[gid], :nodenums[gid]],
                            torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))

                        rst_bkdX = featnet(Xinput_test[gid], featmask_test[gid], self.args.feat_thrd, self.cpu,
                                           self.args.feat_activation, 'feat')
                        bkd_dr_test.data['features'][gid] = torch.add(rst_bkdX[:nodenums[gid]],
                                                                      torch.as_tensor(copy.deepcopy(
                                                                          init_dr_test.data['features'][gid])))

                    clean_graphs_test = list(set(self.benign_dr.data['splits']['test']) - set(bkd_gids_test))
                    # 计算攻击成功率和清洁准确率
                    bkd_acc = evaluate(self.args, bkd_dr_test, real_model, bkd_gids_test)
                    clean_acc = evaluate(self.args, bkd_dr_test, real_model, clean_graphs_test)

                    print(f'bkd_acc: {bkd_acc}, clean_acc: {clean_acc}')
                    # if abs(bkd_acc - 100) < 1e-4:
                    #     print("Early Termination for 100% Attack Rate")
                    #     break
                ######################################################################################################

                # pick up initial candidates / 选取一些图（节点）作为潜在的攻击目标。
                '''
                通过调用 self.bkd_cdd('test') 选取一批图（节点）作为潜在的攻击目标，这里针对的是测试集。
                该函数返回三个列表：bkd_gids_test（后门图的ID）、bkd_nids_test（后门节点的ID）、bkd_nid_groups_test（后门节点的分组），这些都是为后续的攻击准备的。
                '''
                bkd_gids_test, bkd_nids_test, bkd_nid_groups_test = self.bkd_cdd('test', args.select_method)

                # init test data / 选择一些图（节点）作为潜在的攻击目标，这些目标将被用来测试和训练。
                # NOTE: for data that can only add perturbation on features, only init the topo value
                # 初始化触发器，并将中毒图的标签改为目标标签
                init_dr_test = self.init_trigger(self.args, copy.deepcopy(self.benign_dr), bkd_gids_test,
                                                 bkd_nid_groups_test, 0.0, 0.0)
                bkd_dr_test = copy.deepcopy(init_dr_test)  # 深拷贝，用于训练触发器生成模型
                # gen_mask 函数生成拓扑（结构）和特征的掩码。这些掩码定义了在哪些区域可以对图进行修改，以嵌入后门。
                # 输入参数包括初始化后的数据读取器 init_dr_test 和后门图ID bkd_gids_test 及其节点组 bkd_nid_groups_test
                topomask_test, featmask_test = gen_mask(init_dr_test, bkd_gids_test, bkd_nid_groups_test)
                # gen_input 函数根据初始化后的数据和后门图ID bkd_gids_test 生成用于后门攻击的输入数据。
                # 这可能涉及到从原始图数据中提取特定的结构和特征信息，以及准备适合生成
                Ainput_test, Xinput_test = gen_input(self.args, init_dr_test, bkd_gids_test)  # 批数据

                # ----------------- Evaluation -----------------#
                for gid in bkd_gids_test:
                    # 对于测试集中的每个图ID gid，分别使用训练好的toponet和featnet来生成新的邻接矩阵rst_bkdA和特征矩阵rst_bkdX
                    # 这些生成的数据与原始测试数据（通过深复制得到）相加，以产生最终用于评估的测试数据。
                    # 通过这种方式，评估过程能夠考量模型对于经过特定方式修改的数据的处理能力
                    rst_bkdA = toponet(Ainput_test[gid], topomask_test[gid], self.args.topo_thrd, self.cpu,
                                       self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_test[gid], 'topo')
                    # bkd_dr_test.data['adj_list'][gid] = torch.add(rst_bkdA,
                    #     torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid]))
                    bkd_dr_test.data['adj_list'][gid] = torch.add(rst_bkdA[:nodenums[gid], :nodenums[gid]],
                                                                  torch.as_tensor(copy.deepcopy(
                                                                      init_dr_test.data['adj_list'][gid])))

                    rst_bkdX = featnet(Xinput_test[gid], featmask_test[gid], self.args.feat_thrd, self.cpu,
                                       self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_test[gid], 'feat')
                    # bkd_dr_test.data['features'][gid] = torch.add(
                    #     rst_bkdX, torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))
                    bkd_dr_test.data['features'][gid] = torch.add(rst_bkdX[:nodenums[gid]], torch.as_tensor(
                        copy.deepcopy(init_dr_test.data['features'][gid])))

                '''
                通过区分测试集中的图：
                yt_gids：原本就属于目标类别的图ID集合。
                yx_gids：原本不属于目标类别的图ID集合。
                clean_graphs_test：未被选为后门攻击目标的干净（未修改）图ID集合。
                '''

                # graph originally in target label
                yt_gids = [gid for gid in bkd_gids_test if self.benign_dr.data['labels'][gid] == self.args.target_class]
                # graph originally notin target label
                yx_gids = list(set(bkd_gids_test) - set(yt_gids))
                clean_graphs_test = list(set(self.benign_dr.data['splits']['test']) - set(bkd_gids_test))

                # feed into GNN, test success rate
                # 使用所有修改后的测试数据（包括原本属于和不属于目标类别的图）评估模型，得到模型在处理后门数据上的准确率
                print('bkd_acc(attack success rate in backdoor test dataset):')
                bkd_acc = evaluate(self.args, bkd_dr_test, real_model, bkd_gids_test)
                # 仅使用原本不属于目标类别但被修改以尝试归为目标类别的图ID集合yx_gids进行评估，得到翻转率。
                # 这表示模型被欺骗将非目标类别的图错误识别为目标类别的比例。

                # print('flip_rate(attack success rate in non-target-class samples of backdoor test dataset):')
                # flip_rate = evaluate(self.args, bkd_dr_test, real_model, yx_gids)
                # 使用未进行任何修改的干净图ID集合clean_graphs_test进行评估，得到模型在干净数据上的准确率。
                # 这反映了模型在没有遭受后门攻击时的性能。
                print('clean_acc(acc of clean test dataset):')
                clean_acc = evaluate(self.args, bkd_dr_test, real_model, clean_graphs_test)

                # save gnn
                '''
                在每轮重采样步骤（rs_step）的最后一个双层优化步骤（bi_step==self.args.bilevel_steps-1）或当后门攻击成功率（bkd_acc）接近100%时，
                检查是否设置了保存模型的参数（self.args.save_bkd_model）。
                如果满足保存条件，将模型的状态字典（model.state_dict()）、后门攻击成功率（bkd_acc）、翻转率（flip_rate），
                以及在干净数据上的准确率（clean_acc）保存到指定路径（save_path）。这里使用os.makedirs(save_path, exist_ok=True)确保保存路径存在，
                os.path.join构造包含各种参数的文件名，以便区分不同配置下训练的模型。
                '''
                if rs_step == 0 and (joint_step == self.args.bilevel_steps - 1 or abs(bkd_acc - 100) < 1e-4):
                    if self.args.save_bkd_model:
                        save_path = self.args.bkd_model_save_path
                        os.makedirs(save_path, exist_ok=True)
                        save_path = os.path.join(save_path, '%s-%s-%f-%f-%d-%d.t7' % (
                            self.args.model, self.args.dataset, self.args.train_ratio,
                            self.args.bkd_gratio_train, self.args.bkd_num_pergraph, self.args.bkd_size))

                        torch.save({'model': model.state_dict(),
                                    'asr': bkd_acc,
                                    # 'flip_rate': flip_rate,
                                    'clean_acc': clean_acc,
                                    }, save_path)
                        print("Trojaning model is saved at: ", save_path)

                        save_topo_path = save_path.replace(save_path, '%s-%s-%f-%f-%d-%d_topo.t7' % (
                            self.args.model, self.args.dataset, self.args.train_ratio,
                            self.args.bkd_gratio_train, self.args.bkd_num_pergraph, self.args.bkd_size))
                        save_feat_path = save_path.replace(save_path, '%s-%s-%f-%f-%d-%d_feat.t7' % (
                            self.args.model, self.args.dataset, self.args.train_ratio,
                            self.args.bkd_gratio_train, self.args.bkd_num_pergraph, self.args.bkd_size))
                        torch.save({'toponet': toponet.state_dict()}, save_topo_path)
                        print("Toponet model is saved at: ", save_topo_path)
                        torch.save({'featnet': featnet.state_dict()}, save_feat_path)
                        print("Featnet model is saved at: ", save_feat_path)

                '''
                如果在任意重采样步骤的任意双层优化步骤中，后门攻击成功率接近100%（abs(bkd_acc-100) <1e-4），打印消息，
                并使用break提前终止训练过程。这意味着模型已经被成功地注入了后门，进一步的训练可能不会提供更多的价值。
                '''
                if abs(bkd_acc-100) < 1e-4:
                    # bkd_dr_tosave = copy.deepcopy(bkd_dr_test)
                    print("Early Termination for 100% Attack Rate")
                    break
        print('Done')

    def bkd_cdd(self, subset: str, select_methond='random'):
        '''
        这个方法用于选择作为后门攻击目标的图（图集）和节点。它基于特定的标准（如图的特性或节点的重要性）来选择这些目标。
        '''
        # - subset: 'train', 'test'
        # find graphs to add trigger (not modify now)

        bkd_gids = select_cdd_graphs(self.args, self.benign_dr.data['splits'][subset], self.benign_dr.data['adj_list'],
                                     subset)
        # find trigger nodes per graph
        # same sequence with selected backdoored graphs

        # TODO:重要性节点选择
        if select_methond == 'random':
            print('randomly select nodes...')
            bkd_nids, bkd_nid_groups = select_cdd_nodes(self.args, bkd_gids, self.benign_dr.data['adj_list'])

        if select_methond == 'cluster_degree':
            print('Selecting nodes for backdoor attack...')
            bkd_nids, bkd_nid_groups = select_poisoned_nodes(self.args, bkd_gids, self.benign_dr.data['adj_list'],
                                                             self.benign_dr, self.benign_model,
                                                             self.benign_dr.data['nlabels'])
            print('node selection done.')

        if select_methond == 'saliency':
            print('Saliency Map...')
            bkd_nids, bkd_nid_groups = select_poisoned_nodes_2(self.args, bkd_gids, self.benign_dr.data['adj_list'],
                                                               self.benign_dr, self.benign_model,
                                                               self.benign_dr.data['nlabels'])

        assert len(bkd_gids) == len(bkd_nids) == len(bkd_nid_groups)
        # 返回的是：被注入后门的图，图中的中毒节点id，和中毒节点id的分组
        return bkd_gids, bkd_nids, bkd_nid_groups

    @staticmethod
    def init_trigger(args, dr: DataReader, bkd_gids: list, bkd_nid_groups: list, init_edge: float, init_feat: float):
        '''
        这个方法用于在选定的图和节点中注入触发器。这包括修改图的结构（邻接矩阵）和节点的特征，以及将图的标签改为攻击者指定的目标类别。
        args（一般包含全局配置和攻击参数）
        dr（一个 DataReader 实例，包含图数据）
        bkd_gids（被选为后门攻击目标的图ID列表）
        bkd_nid_groups（每个目标图中，被选为后门触发器节点的节点组列表）
        init_edge 和 init_feat（分别指定要注入的边和特征的初始值）
        '''
        if init_feat == None:
            init_feat = - 1
            print('init feat == None, transferred into -1')

        # (in place) datareader trigger injection
        # 使用 tqdm 生成一个进度条，遍历 bkd_gids 列表中的每个图ID gid
        for i in tqdm(range(len(bkd_gids)), desc="initializing trigger..."):
            gid = bkd_gids[i]  # 对于每个目标图 gid
            for group in bkd_nid_groups[i]:
                # change adj in-place
                # 首先，遍历该图的每个节点组 group。对于组内的每对不同节点（v1, v2），将它们的连接（在邻接矩阵中）设置为 init_edge。
                # 这通过修改 dr.data['adj_list'][gid] 实现，即直接在原始数据上修改
                src, dst = [], []
                for v1 in group:
                    for v2 in group:
                        if v1 != v2:
                            src.append(v1)
                            dst.append(v2)
                a = np.array(dr.data['adj_list'][gid])
                a[src, dst] = init_edge
                dr.data['adj_list'][gid] = a.tolist()

                # change features in-place
                # 计算特征维度 featdim，然后将选定节点组 group 的特征全部设置为 init_feat。这同样直接修改了 dr.data['features'][gid]
                featdim = len(dr.data['features'][0][0])
                a = np.array(dr.data['features'][gid])
                a[group] = np.ones((len(group), featdim)) * init_feat
                dr.data['features'][gid] = a.tolist()

            # change graph labels
            # 确保 args.target_class 不为 None，然后将目标图的标签改为攻击者指定的目标类别 args.target_class
            assert args.target_class is not None
            dr.data['labels'][gid] = args.target_class

        return dr


if __name__ == '__main__':
    args = parse_args()
    attack = GraphBackdoor(args)
    attack.run()