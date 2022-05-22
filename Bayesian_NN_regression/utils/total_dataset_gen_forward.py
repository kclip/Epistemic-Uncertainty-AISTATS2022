import torch
import torch.nn as nn
from torch.nn import functional as F

FIRST_HIDDEN = 3 # number of neurons of the hidden layer (currenly only having one hidden layer)
STD_Y = 0.1 # standard deviation for observation noise

class Total_dataset_gen_forward:
    def __init__(self, num_u_samples, num_meta_tr_tasks, num_samples, mean_U, std_U, std_W, mean_X, std_X):
        # data points that are interested
        self.dataset = {}
        self.num_neurons = FIRST_HIDDEN + FIRST_HIDDEN + FIRST_HIDDEN + 1 # fc1 weight, fc1 bias, fc2 weight, fc2 bias
        self.dataset['u'] = torch.zeros(num_u_samples, self.num_neurons)
        self.dataset['w_mtr'] = torch.zeros(num_u_samples, num_meta_tr_tasks*self.num_neurons)   
        self.dataset['w_mte'] = torch.zeros(num_u_samples, self.num_neurons)   
        self.dataset['z_mtr'] = torch.zeros(num_u_samples, num_meta_tr_tasks*num_samples*2) # Nm for each u # 2 for (x,y)
        self.dataset['z_mte'] = torch.zeros(num_u_samples, num_samples*2) 
        self.dataset['x_test'] = torch.zeros(num_u_samples, 1) # single x
        self.dataset['y_test'] = torch.zeros(num_u_samples, 1) # single y
        self.std_W = std_W
        self.mean_X = mean_X
        self.std_X = std_X
        self.num_u_samples = num_u_samples
        self.mean_U = mean_U
        self.std_U = std_U
        self.num_samples = num_samples
        self.num_meta_tr_tasks = num_meta_tr_tasks
        print('generating dataset...', 'm', self.num_samples, 'N', self.num_meta_tr_tasks)
    def gen_total(self):
        # generate data
        for ind_u in range(self.num_u_samples):
            if ind_u % 100 == 0:
                print('ind u', ind_u)
            u_fc1 = nn.Linear(1, FIRST_HIDDEN)
            u_fc2 = nn.Linear(FIRST_HIDDEN, 1)
            torch.nn.init.normal_(u_fc1.weight, mean=self.mean_U, std=self.std_U)
            torch.nn.init.normal_(u_fc2.weight, mean=self.mean_U, std=self.std_U)
            torch.nn.init.normal_(u_fc1.bias, mean=self.mean_U, std=self.std_U)
            torch.nn.init.normal_(u_fc2.bias, mean=self.mean_U, std=self.std_U)
            para_list_u_fc1 = list(map(lambda p: p[0], zip(u_fc1.parameters())))
            para_list_u_fc2 = list(map(lambda p: p[0], zip(u_fc2.parameters())))
            u_fc1_weight_flatten = torch.flatten(para_list_u_fc1[0])
            u_fc1_bias_flatten = torch.flatten(para_list_u_fc1[1])
            u_fc2_weight_flatten = torch.flatten(para_list_u_fc2[0])
            u_fc2_bias_flatten = torch.flatten(para_list_u_fc2[1])
            curr_u = torch.cat((u_fc1_weight_flatten, u_fc1_bias_flatten, u_fc2_weight_flatten, u_fc2_bias_flatten))
            self.dataset['u'][ind_u] = curr_u.detach()
            mean_W = [para_list_u_fc1, para_list_u_fc2]
            for ind_w in range(self.num_meta_tr_tasks): # N
                dnn_curr_w = DNN(mean_W, self.std_W)
                para_list_from_curr_w = list(map(lambda p: p[0], zip(dnn_curr_w.parameters())))
                curr_w = para_list_to_flatdata(para_list_from_curr_w)
                self.dataset['w_mtr'][ind_u, ind_w*self.num_neurons:(ind_w+1)*self.num_neurons] = curr_w.detach()
                # generate x (m samples)
                x = torch.normal(mean=self.mean_X * torch.ones(self.num_samples), std=self.std_X*torch.ones(self.num_samples))
                x = x.unsqueeze(dim=1)
                y = dnn_curr_w(x, para_list_from_curr_w)
                z_curr_task = torch.flatten(torch.cat((x,y), dim=1))
                self.dataset['z_mtr'][ind_u, ind_w*self.num_samples*2:(ind_w+1)*self.num_samples*2] = z_curr_task.detach().squeeze()
            for ind_w in range(1):
                dnn_curr_w = DNN(mean_W, self.std_W)
                para_list_from_curr_w = list(map(lambda p: p[0], zip(dnn_curr_w.parameters())))
                curr_w = para_list_to_flatdata(para_list_from_curr_w)
                self.dataset['w_mte'][ind_u, ind_w*self.num_neurons:(ind_w+1)*self.num_neurons] = curr_w.detach()
                # generate dataset 
                x = torch.normal(mean=self.mean_X * torch.ones(self.num_samples), std=self.std_X*torch.ones(self.num_samples))
                x = x.unsqueeze(dim=1)
                y = dnn_curr_w(x, para_list_from_curr_w)
                z_curr_task = torch.flatten(torch.cat((x,y), dim=1))
                self.dataset['z_mte'][ind_u, ind_w*self.num_samples*2:(ind_w+1)*self.num_samples*2] = z_curr_task.detach()
                # test set
                x = torch.normal(mean=self.mean_X * torch.ones(1), std=self.std_X*torch.ones(1))
                x = x.unsqueeze(dim=1)
                y = dnn_curr_w(x, para_list_from_curr_w)
                self.dataset['x_test'][ind_u, :] = x.detach()
                self.dataset['y_test'][ind_u, :] = y.detach()
        return self.dataset


def selecting_essentials(dataset, variable_list): # only select desired rvs for MI estimation
    len_data = 0
    for variable in variable_list:
        len_data += dataset[variable].shape[1]
    num_tot_datapoints = dataset[variable].shape[0]
    flatten_dataset = torch.zeros(num_tot_datapoints, len_data)
    start_ind = 0
    for ind_variable in range(len(variable_list)):
        variable = variable_list[ind_variable]
        flatten_dataset[:, start_ind:start_ind+dataset[variable].shape[1]] = dataset[variable]
        start_ind += dataset[variable].shape[1]
    return flatten_dataset

def selecting_essentials_for_mi_para_via_avg(ind_u, dataset, variable_list, num_tot_datapoints_per_u, len_data, m): # only select desired rvs for MI estimation (especially for parameter-level MI since we are actually using dataset in meta-training tasks as meta-testing task)
    num_tot_datapoints = num_tot_datapoints_per_u
    flatten_dataset = torch.zeros(num_tot_datapoints, len_data) # N_max, shape(z_mte) + shape(w_mte)
    num_neurons = FIRST_HIDDEN + FIRST_HIDDEN + FIRST_HIDDEN + 1
    for ind_task in range(num_tot_datapoints):
        curr_task_w = dataset['w_mtr'][ind_u][ind_task*num_neurons:(ind_task+1)*num_neurons]
        curr_task_z = dataset['z_mtr'][ind_u][ind_task*m*2:(ind_task+1)*m*2]
        curr_task_w_z = torch.cat((curr_task_w, curr_task_z))
        flatten_dataset[ind_task, :] = curr_task_w_z
    return flatten_dataset


class DNN(nn.Module):
    def __init__(self, mean_W, std_W):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(1, FIRST_HIDDEN)
        self.fc2 = nn.Linear(FIRST_HIDDEN, 1)
        para_list_u_fc1 = list(map(lambda p: p[0], zip(self.fc1.parameters())))
        para_list_u_fc2 = list(map(lambda p: p[0], zip(self.fc2.parameters())))
        N_fc1_weight = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(torch.flatten(mean_W[0][0]).shape), torch.eye(torch.flatten(mean_W[0][0]).shape[0]))
        para_list_u_fc1[0].data = mean_W[0][0] + std_W * N_fc1_weight.sample().reshape(mean_W[0][0].shape)
        N_fc1_bias = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(torch.flatten(mean_W[0][1]).shape), torch.eye(torch.flatten(mean_W[0][1]).shape[0]))
        para_list_u_fc1[1].data = mean_W[0][1] + std_W * N_fc1_bias.sample().reshape(mean_W[0][1].shape)
        N_fc2_weight = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(torch.flatten(mean_W[1][0]).shape), torch.eye(torch.flatten(mean_W[1][0]).shape[0]))
        para_list_u_fc2[0].data = mean_W[1][0] + std_W * N_fc2_weight.sample().reshape(mean_W[1][0].shape)
        N_fc2_bias = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(torch.flatten(mean_W[1][1]).shape), torch.eye(torch.flatten(mean_W[1][1]).shape[0]))
        para_list_u_fc2[1].data = mean_W[1][1] + std_W * N_fc2_bias.sample().reshape(mean_W[1][1].shape)
        self.activ = nn.ReLU()
    def forward(self, input, var):
        x = self.activ(F.linear(input, var[0], var[1]))
        x = F.linear(x, var[2], var[3])
        # add observation noise
        noise = torch.normal(mean=torch.zeros(x.shape[0]), std = torch.ones(x.shape[0]))
        x = x + noise.unsqueeze(dim=1)*STD_Y
        return x

def para_list_to_flatdata(para_list):
    flatten_list = []
    for para in para_list:
        flatten_list.append(torch.flatten(para))
    return torch.cat(flatten_list)

def trim_to_mN(dataset, num_u_samples, N, m): # as dataset contains max. number of m and N, trim to desired m and N
    trimmed_dataset = {}
    num_neurons = FIRST_HIDDEN + FIRST_HIDDEN + FIRST_HIDDEN + 1 # fc1 weight, fc1 bias, fc2 weight, fc2 bias
    trimmed_dataset['u'] = torch.zeros(num_u_samples, num_neurons)
    trimmed_dataset['w_mtr'] = torch.zeros(num_u_samples, N*num_neurons)   
    trimmed_dataset['w_mte'] = torch.zeros(num_u_samples, num_neurons)   
    trimmed_dataset['z_mtr'] = torch.zeros(num_u_samples, N*m*2) # Nm for each u # 2 for (x,y)
    trimmed_dataset['z_mte'] = torch.zeros(num_u_samples, m*2) 
    trimmed_dataset['x_test'] = torch.zeros(num_u_samples, 1)
    trimmed_dataset['y_test'] = torch.zeros(num_u_samples, 1) 
    trimmed_dataset['u'] = dataset['u'][:num_u_samples]
    trimmed_dataset['w_mtr'] = dataset['w_mtr'][:num_u_samples, :N*num_neurons]
    trimmed_dataset['w_mte'] = dataset['w_mte'][:num_u_samples]
    max_num_samples = dataset['z_mte'][0].shape[0]//2
    assert max_num_samples == 100
    for ind_w in range(N):
        original_start_ind = ind_w*max_num_samples*2
        trimmed_dataset['z_mtr'][:, ind_w*m*2:(ind_w+1)*m*2] = dataset['z_mtr'][:num_u_samples, original_start_ind:original_start_ind+m*2]
    for ind_w in range(1):
        original_start_ind = ind_w*max_num_samples*2
        trimmed_dataset['z_mte'][:, ind_w*m*2:(ind_w+1)*m*2] = dataset['z_mte'][:num_u_samples, original_start_ind:original_start_ind+m*2]
    trimmed_dataset['x_test'] = dataset['x_test'][:num_u_samples]
    trimmed_dataset['y_test'] = dataset['y_test'][:num_u_samples]
    return trimmed_dataset