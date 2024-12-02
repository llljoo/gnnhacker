import argparse

def add_data_group(group):
    group.add_argument('--device',type=int,default=0, help='cuda:0')
    group.add_argument('--seed', type=int, default=123)
    group.add_argument('--dataset', type=str, default='MUTAG', help="used dataset") # politifact gossipcop ENZYMES AIDS DHFR COX2 BZR PROTEINS_full // FIRSTMM_DB
    group.add_argument('--data_path', type=str, default='../dataset', help="the directory used to save dataset")
    group.add_argument('--use_nlabel_asfeat', action='store_true', default=True, help="use node labels as (part of) node features")
    group.add_argument('--use_org_node_attr', action='store_true', default=True, help="use node attributes as (part of) node features")
    group.add_argument('--use_degree_asfeat', action='store_true', default=True, help="use node degrees as (part of) node features")
    group.add_argument('--data_verbose', action='store_true', help="print detailed dataset info")
    group.add_argument('--save_data', action='store_true')


def add_model_group(group):
    group.add_argument('--model', type=str, default='sage', help="used model") # gat gcn sage gin
    group.add_argument('--train_ratio', type=float, default=0.8, help="ratio of trainset from whole dataset")
    group.add_argument('--hidden_dim', nargs='+', default=[64, 16], type=int, help='constrain how much products a vendor can have')
    group.add_argument('--num_head', type=int, default=3, help="GAT head number")

    group.add_argument('--batch_size', type=int, default=16)
    group.add_argument('--train_epochs', type=int, default=2)
    group.add_argument('--lr', type=float, default=0.01)
    group.add_argument('--lr_decay_steps', nargs='+', default=[5, 15], type=int) # [25, 35]
    group.add_argument('--weight_decay', type=float, default=5e-4)
    group.add_argument('--dropout', type=float, default=0.5)
    group.add_argument('--train_verbose', action='store_true', default=True, help="print training details")
    group.add_argument('--log_every', type=int, default=1, help='print every x epoch')
    group.add_argument('--eval_every', type=int, default=5, help='evaluate every x epoch')

    group.add_argument('--clean_model_save_path', type=str, default='../save/model/clean')
    group.add_argument('--save_clean_model', action='store_true', default=True)

def add_atk_group(group):
    group.add_argument('--bkd_gratio_train', type=float, default=0.1, help="backdoor graph ratio in trainset")
    group.add_argument('--bkd_gratio_test', type=float, default=0.5, help="backdoor graph ratio in testset")
    group.add_argument('--bkd_num_pergraph', type=int, default=1, help="number of backdoor triggers per graph") # 触发器数量
    group.add_argument('--bkd_size', type=int, default=15, help="number of nodes for each trigger") # 触发器大小
    group.add_argument('--target_class', type=int, default=0, help="the targeted node/graph label")
     
    group.add_argument('--gtn_layernum', type=int, default=3, help="layer number of GraphTrojanNet")
    group.add_argument('--pn_rate', type=float, default=1, help="ratio between trigger-embedded graphs (positive) and benign ones (negative)")
    group.add_argument('--gtn_input_type', type=str, default='2hop', help="how to process org graphs before inputting to GTN")

    group.add_argument('--resample_steps', type=int, default=1, help="# iterations to re-select graph samples") # 选不同的图
    group.add_argument('--bilevel_steps', type=int, default=1, help="# bi-level optimization iterations")
    group.add_argument('--gtn_lr', type=float, default=0.01)
    group.add_argument('--gtn_epochs', type=int, default=20, help="# attack epochs")
    group.add_argument('--topo_activation', type=str, default='sigmoid', help="activation function for topology generator")
    group.add_argument('--feat_activation', type=str, default='relu', help="activation function for feature generator")
    group.add_argument('--topo_thrd', type=float, default=0.5, help="threshold for topology generator")
    group.add_argument('--feat_thrd', type=float, default=0, help="threshold for feature generator (only useful for binary feature)")

    group.add_argument('--lambd', type=float, default=0.6, help="a hyperparameter to balance attack loss components")
    # group.add_argument('--atk_verbose', action='store_true', help="print attack details")
    group.add_argument('--save_bkd_model', action='store_true', default=True)
    group.add_argument('--bkd_model_save_path', type=str, default='../save/model/bkd')


    group.add_argument('--saliency_weight', type=float, default=0.6)
    group.add_argument('--dis_weight', type=float, default=0.5)
    group.add_argument('--randomly_preserve', type=float, default=0.1)
    group.add_argument('--WKT', type=int, default=5,help='random walk times')
    group.add_argument('--WKL', type=int, default=4,help='random walk length')
    group.add_argument('--randomly_smooth', action='store_true')
    group.add_argument('--select_method', type=str, default='random', help="methods of select trigger nodes") # cluster_degree/random/saliency
    group.add_argument('--real_model', type=str, default='gat')
    group.add_argument('--collaborative', action='store_true')
    group.add_argument('--iter_per_epoch', type=int, default=50)
    group.add_argument('--sample_method', type=str, default='edge') # edge node

    group.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')

def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    atk_group = parser.add_argument_group(title="Attack-related configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_atk_group(atk_group)

    return parser.parse_args()
