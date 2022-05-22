import torch
from utils.cmi import CMI
import argparse
import numpy as np
import copy
import os
from utils.total_dataset_gen_forward import Total_dataset_gen_forward

def parse_args():
    parser = argparse.ArgumentParser(description='code for information-theoretic bounds')
    parser.add_argument('--num_u_samples', type=int, default=30000, help='number of maximum different true U (hyperparameter) to average over U')
    parser.add_argument('--num_samples', type=int, default=100, help='number of maximum samples that we will use for Z (m)')
    parser.add_argument('--num_meta_tr_tasks', type=int, default=100, help='number of maximum tasks that we will use for meta-training (N)')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')
    parser.add_argument('--path_for_total_dataset', type=str, default= '../../bnn_dataset_obs_noise/', help='path for Z_{1:N} (meta-training dataset), Z (meta-test dataset for training), (X,Y) (meta-test dataset for test)')
    parser.add_argument('--std_W', type=float, default=0.1, help='standard deviation for W that determines how useful the meta-learning can be (Fig. 2 vs. Fig. 3)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    random_seed = args.random_seed
    if_fix_random_seed = False
    if if_fix_random_seed:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
    else:
        pass
    # setting
    # hyperprior for U: mean: N(0,1)
    mean_U = torch.tensor(0)
    std_U = torch.tensor(1)
    # prior for W
    #mean_W is U
    std_W = torch.tensor(args.std_W) # if args.std_W = 0.1: Fig. 2 (small_std_W in /runs), if args.std_W = 1: Fig. 3 (large_std_W in /runs)
    mean_X = torch.tensor(0.0) 
    std_X = torch.tensor(1.0)
    num_u_samples = args.num_u_samples # this is actually number of total datapoints for MI computation
    num_samples = args.num_samples
    num_meta_tr_tasks = args.num_meta_tr_tasks

    if not os.path.isdir(args.path_for_total_dataset):
        os.makedirs(args.path_for_total_dataset)

    path_for_dataset_joint = args.path_for_total_dataset + 'joint.pt'
    path_for_dataset_marg = args.path_for_total_dataset + 'marg.pt'

    # joint dataset
    print('generate new dataset from scratch')
    # joint and marginal data here is in fact both joint -- we will use this 'marginal' dataset along with 'joint' dataset here to make actual marginal dataset (by substituting desired rvs in 'joint' data to 'marginal' dataset)
    total_dataset_forward_joint = Total_dataset_gen_forward(num_u_samples, num_meta_tr_tasks, num_samples, mean_U, std_U, std_W, mean_X, std_X)
    dataset_joint_original = total_dataset_forward_joint.gen_total()
    total_dataset_forward_marg = Total_dataset_gen_forward(num_u_samples, num_meta_tr_tasks, num_samples, mean_U, std_U, std_W, mean_X, std_X)
    dataset_marg_tmp_original = total_dataset_forward_marg.gen_total()
    # save data to the predefined path
    torch.save(dataset_joint_original, path_for_dataset_joint)
    torch.save(dataset_marg_tmp_original, path_for_dataset_marg)
    