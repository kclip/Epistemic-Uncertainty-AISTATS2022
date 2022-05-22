import torch
from utils.cmi import CMI
import argparse
import numpy as np
import copy
import os
from utils.total_dataset_gen_forward import Total_dataset_gen_forward, trim_to_mN, selecting_essentials
import time
import scipy.io as sio

def parse_args():
    parser = argparse.ArgumentParser(description='code for information-theoretic bounds')
    parser.add_argument('--num_u_samples', type=int, default=30000, help='number of different true U (hyperparameter) to average over U')
    parser.add_argument('--num_samples', type=int, default=None, help='number of samples for Z (m)')
    parser.add_argument('--num_meta_tr_tasks', type=int, default=None, help='number of tasks for meta-training (N)')
    parser.add_argument('--classifier_lr_MI', type=float, default=0.001, help='lr for train in C-MINE')
    parser.add_argument('--mb_size_MI', type=int, default=64, help='mb size for train in C-MINE')
    parser.add_argument('--classifier_tr_epoch_MI', type=int, default=200, help='num training epochs for training classifier used in C-MINE') 
    parser.add_argument('--weight_decay', type=float, default=0.001, help='l2 reg for training C-MINE')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')
    parser.add_argument('--num_indep_exp_total', type=int, default=10, help='number of total independent experiments')
    parser.add_argument('--path_for_total_dataset', type=str, default= '../../bnn_dataset_obs_noise/', help='path for Z_{1:N} (meta-training dataset), Z (meta-test dataset for training), (X,Y) (meta-test dataset for test), U (hyperparameter), W_{1:N} (parameter for meta-training dataset), W (parameter for meta-testing dataset)')
    parser.add_argument('--results_path_name', type=str, default= 'setting_of_exp', help='dir name for saving results')
    parser.add_argument('--exp_mode', type=int, default=0, help='0: per m (number of samples per task), 1: per N (number of meta-training tasks)')
    parser.add_argument('--smile_tau', type=float, default=None, help='clipping of E_Q term in SMILE, if None no clipping') # UNDERSTANDING THE LIMITATIONS OF VARIATIONAL MUTUAL INFORMATION ESTIMATORS ICLR 2020
    parser.add_argument('--if_conven', dest='if_conven', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    random_seed = args.random_seed
    if_fix_random_seed = True
    if if_fix_random_seed:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
    else:
        pass

    start_time = time.time()
    results_dir = '../../results/' + args.results_path_name + '/smile_' + str(int(args.smile_tau)) + '/'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if args.if_conven:
        assert args.exp_mode == 0 # per N exp. has no meaning for conven!
        num_task_fixed = 0
        results_path = results_dir + '/memr_conven_exp_mode_' + str(args.exp_mode) + '.mat'
    else:
        num_task_fixed = 1
        results_path = results_dir +'/memr_exp_mode_' + str(args.exp_mode) + '.mat'
    if args.num_samples is None:
        if args.exp_mode == 0: # per m
            num_samples_iter = [1, 2, 5, 50]
        elif args.exp_mode == 1: # per N
            num_samples_iter = [1]
        else: 
            raise NotImplementedError
    else:
        num_samples_iter = [args.num_samples]
    if args.num_meta_tr_tasks is None:
        if args.exp_mode == 0: # per m
            num_tasks_iter = [num_task_fixed]
        elif args.exp_mode == 1: # per N
            num_tasks_iter = [1, 2, 5, 10, 15]
        else:
            raise NotImplementedError
    else:
        num_tasks_iter = [args.num_meta_tr_tasks]
    results_dict = {}
    memr_eval = torch.zeros(args.num_indep_exp_total, len(num_samples_iter), len(num_tasks_iter))
    m_for_save = torch.zeros(len(num_samples_iter))
    N_for_save = torch.zeros(len(num_tasks_iter))
    for ind_exp in range(args.num_indep_exp_total):
        print('ind indep experiment: ', ind_exp)
        for ind_num_samples_iterative in range(len(num_samples_iter)):
            for ind_num_meta_tr_tasks_iterative in range(len(num_tasks_iter)): 
                num_samples_iterative = num_samples_iter[ind_num_samples_iterative]
                num_meta_tr_tasks_iterative = num_tasks_iter[ind_num_meta_tr_tasks_iterative]
                m_for_save[ind_num_samples_iterative] = num_samples_iterative
                N_for_save[ind_num_meta_tr_tasks_iterative] = num_meta_tr_tasks_iterative
                args.num_samples = num_samples_iterative
                args.num_meta_tr_tasks = num_meta_tr_tasks_iterative
                # setting
                # hyperprior for U: mean: N(0,1)
                mean_U = torch.tensor(0)
                std_U = torch.tensor(1)
                # prior for W
                #mean_W is U
                std_W = torch.tensor(0.1)
                mean_X = torch.tensor(0.0) 
                std_X = torch.tensor(1.0)
                num_u_samples = args.num_u_samples # this is actually number of total datapoints for MI computation
                num_samples = args.num_samples
                num_meta_tr_tasks = args.num_meta_tr_tasks
                num_indep_exp_total = args.num_indep_exp_total
                clamping_MI = torch.finfo(torch.float32).eps # clamping value for MI estimation (http://proceedings.mlr.press/v115/mukherjee20a/mukherjee20a.pdf)
                classifier_lr_MI = args.classifier_lr_MI
                mb_size_MI = args.mb_size_MI
                classifier_tr_epoch_MI = args.classifier_tr_epoch_MI
                # load two joint data set (as described in datagen.py)
                path_for_dataset_joint = args.path_for_total_dataset + 'joint.pt'
                path_for_dataset_marg = args.path_for_total_dataset + 'marg.pt'
                curr_key_for_saved_dataset = 'num_u_' + str(num_u_samples) + '_m_' + str(num_samples) + '_N_' + str(num_meta_tr_tasks)
                if os.path.exists(path_for_dataset_joint) and os.path.exists(path_for_dataset_marg): # 0 for joint, 1 for marg
                    dataset_joint_original_max = torch.load(path_for_dataset_joint)
                    dataset_marg_tmp_original_max = torch.load(path_for_dataset_marg)
                    # since saved data contains maximum m and maximum N, select only desired m and N
                    dataset_joint_original = trim_to_mN(dataset_joint_original_max, num_u_samples, num_meta_tr_tasks, num_samples)
                    dataset_marg_tmp_original = trim_to_mN(dataset_marg_tmp_original_max, num_u_samples, num_meta_tr_tasks, num_samples)
                else:
                    print('run datagen.py (total_data_gen.sh) first!')
                    raise NotImplementedError
                dataset_joint = dataset_joint_original
                dataset_marg_tmp = dataset_marg_tmp_original
                dataset_marg = copy.deepcopy(dataset_joint)
                dataset_marg['y_test'] = dataset_marg_tmp['y_test'] # this makes 'y_test' follows marginal distribution not joint distribution
                variable_list_mi_1_1 = ['y_test', 'x_test', 'w_mte', 'z_mtr', 'z_mte']
                flatten_dataset_mi_1_1_joint = selecting_essentials(dataset_joint, variable_list_mi_1_1)
                flatten_dataset_mi_1_1_marg = selecting_essentials(dataset_marg, variable_list_mi_1_1)
                MI_1_1 = CMI(joint_dataset=flatten_dataset_mi_1_1_joint, marginal_dataset=flatten_dataset_mi_1_1_marg, smile_tau=args.smile_tau)
                mi_1_1 = MI_1_1.MI_compute(clamping=clamping_MI, classifier_lr=classifier_lr_MI, mb_size=int(mb_size_MI), tr_epoch=int(classifier_tr_epoch_MI), weight_decay=args.weight_decay)
                variable_list_mi_1_2 = ['y_test', 'x_test', 'z_mtr', 'z_mte']
                flatten_dataset_mi_1_2_joint = selecting_essentials(dataset_joint, variable_list_mi_1_2)
                flatten_dataset_mi_1_2_marg = selecting_essentials(dataset_marg, variable_list_mi_1_2)
                MI_1_2 = CMI(joint_dataset=flatten_dataset_mi_1_2_joint, marginal_dataset=flatten_dataset_mi_1_2_marg, smile_tau=args.smile_tau)
                mi_1_2 = MI_1_2.MI_compute(clamping=clamping_MI, classifier_lr=classifier_lr_MI, mb_size=int(mb_size_MI), tr_epoch=int(classifier_tr_epoch_MI), weight_decay=args.weight_decay)
                mi_1 = mi_1_1 - mi_1_2
                print('------------------------------------------------------------------')
                print('num_samples: ', num_samples, 'num_meta_tr_tasks: ', num_meta_tr_tasks, 'memr: ', mi_1)
                memr_eval[ind_exp, ind_num_samples_iterative, ind_num_meta_tr_tasks_iterative] = mi_1
                results_dict = {}
                results_dict['memr'] = memr_eval.data.numpy()
                results_dict['m'] = m_for_save.data.numpy()
                results_dict['N'] = N_for_save.data.numpy()
                sio.savemat(results_path, results_dict)
            results_dict = {}
            results_dict['memr'] = memr_eval.data.numpy()
            results_dict['m'] = m_for_save.data.numpy()
            results_dict['N'] = N_for_save.data.numpy()
            sio.savemat(results_path, results_dict)
        results_dict = {}
        results_dict['memr'] = memr_eval.data.numpy()
        results_dict['m'] = m_for_save.data.numpy()
        results_dict['N'] = N_for_save.data.numpy()
        sio.savemat(results_path, results_dict)
    results_dict = {}
    results_dict['memr'] = memr_eval.data.numpy()
    results_dict['m'] = m_for_save.data.numpy()
    results_dict['N'] = N_for_save.data.numpy()
    sio.savemat(results_path, results_dict)





