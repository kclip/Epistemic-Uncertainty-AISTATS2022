import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# code for C-MINE (http://proceedings.mlr.press/v115/mukherjee20a/mukherjee20a.pdf)
class CMI:
    def __init__(self, joint_dataset, marginal_dataset, smile_tau):
        # joint_dataset: # data, length of each data
        self.joint_dataset = joint_dataset
        self.joint_label = torch.ones(len(joint_dataset)).unsqueeze(dim=1)
        self.marginal_dataset = marginal_dataset
        self.marginal_label = torch.zeros(len(marginal_dataset)).unsqueeze(dim=1)
        self.smile_tau = smile_tau
        assert len(self.joint_dataset) == len(self.marginal_dataset) # surely since we are making marg. via permutation
    def split_tr_te(self):
        self.training_data = {}
        self.test_data = {}
        total_data_len = len(self.joint_dataset)
        rand_perm = torch.arange(total_data_len)
        # use same for tr te (no split)
        tr_idx = rand_perm[:total_data_len]
        te_idx = tr_idx
        # label unsqueeze
        joint_tr_data = self.joint_dataset[tr_idx]
        joint_tr_label = self.joint_label[tr_idx] # always one...
        marg_tr_data = self.marginal_dataset[tr_idx]
        marg_tr_label = self.marginal_label[tr_idx]
        self.training_data['data'] = torch.cat((joint_tr_data, marg_tr_data), dim=0)
        self.training_data['label'] = torch.cat((joint_tr_label, marg_tr_label), dim=0)
        self.training_data['joint_data'] = joint_tr_data
        self.training_data['joint_label'] = joint_tr_label
        self.training_data['marginal_data'] = marg_tr_data
        self.training_data['marginal_label'] = marg_tr_label
        joint_te_data = self.joint_dataset[te_idx]
        joint_te_label = self.joint_label[te_idx] # always one...
        marg_te_data = self.marginal_dataset[te_idx]
        marg_te_label = self.marginal_label[te_idx]
        self.test_data['joint_data'] = joint_te_data
        self.test_data['joint_label'] = joint_te_label
        self.test_data['marginal_data'] = marg_te_data
        self.test_data['marginal_label'] = marg_te_label

    def train_classifier(self, tr_epoch, mb_size, classifier_lr, weight_decay):
        bce_loss = torch.nn.BCELoss()
        classifier = Classifier(self.training_data['data'].shape[1])
        optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr, weight_decay=weight_decay)
        best_acc = -100
        assert mb_size % 2 == 0
        for epoch in range(tr_epoch):
            perm_joint = torch.randperm(self.training_data['joint_data'].shape[0]) 
            perm_marg = torch.randperm(self.training_data['marginal_data'].shape[0]) 
            curr_shuffle_data_joint = self.training_data['joint_data'][perm_joint]
            curr_shuffle_label_joint = self.training_data['joint_label'][perm_joint]
            curr_shuffle_data_marg = self.training_data['marginal_data'][perm_marg]
            curr_shuffle_label_marg = self.training_data['marginal_label'][perm_marg]
            for iter in range(self.training_data['data'].shape[0]//mb_size):
                optimizer.zero_grad()
                mb_size_joint = mb_size // 2
                mb_size_marg = mb_size // 2 
                mb_tr_data_joint = curr_shuffle_data_joint[iter*mb_size_joint:(iter+1)*mb_size_joint]
                mb_tr_label_joint = curr_shuffle_label_joint[iter*mb_size_joint:(iter+1)*mb_size_joint]
                mb_tr_data_marg = curr_shuffle_data_marg[iter*mb_size_marg:(iter+1)*mb_size_marg]
                mb_tr_label_marg = curr_shuffle_label_marg[iter*mb_size_marg:(iter+1)*mb_size_marg]
                mb_tr_data = torch.cat((mb_tr_data_joint, mb_tr_data_marg), dim=0)
                mb_tr_label = torch.cat((mb_tr_label_joint, mb_tr_label_marg), dim=0)
                out = classifier(mb_tr_data)
                loss = bce_loss(out, mb_tr_label)
                loss.backward()
                optimizer.step()
            if epoch % 1 == 0: # use the classifier that has the best average 'accuracy' for the total training dataset
                out_joint_tr = classifier(self.training_data['joint_data'])
                out_marginal_tr = classifier(self.training_data['marginal_data'])
                joint_acc_tr = self.getting_accuracy(out_joint_tr, self.training_data['joint_label'])
                marg_acc_tr = self.getting_accuracy(out_marginal_tr, self.training_data['marginal_label'])
                avg_acc_tr = joint_acc_tr*0.5 + marg_acc_tr*0.5
                if avg_acc_tr > best_acc:
                    best_acc = avg_acc_tr
                    classifier_best_acc = copy.deepcopy(classifier) # choose best training accuracy for C-MINE
                else:
                    pass
        return classifier_best_acc
    
    def test_classifier(self, trained_classifer, clamping):
        # train accuracy
        out_joint_tr = trained_classifer(self.training_data['joint_data'])
        out_marginal_tr = trained_classifer(self.training_data['marginal_data'])
        joint_acc_tr = self.getting_accuracy(out_joint_tr, self.training_data['joint_label'])
        marg_acc_tr = self.getting_accuracy(out_marginal_tr, self.training_data['marginal_label'])
        # joint test
        out_joint = trained_classifer(self.test_data['joint_data'])
        out_marginal = trained_classifer(self.test_data['marginal_data'])
        # getting accuracy just for check
        joint_acc = self.getting_accuracy(out_joint, self.test_data['joint_label'])
        marg_acc = self.getting_accuracy(out_marginal, self.test_data['marginal_label'])
        # actual computation of MI
        out_joint = torch.clamp(out_joint, min=clamping, max=1-clamping)
        out_marginal = torch.clamp(out_marginal, min=clamping, max=1-clamping)
        joint_log_prob = 0
        for ind_te in range(out_joint.shape[0]):
            joint_log_prob += torch.log(out_joint[ind_te]/(1-out_joint[ind_te]))
        joint_log_prob /= out_joint.shape[0]
        marg_prob = 0
        for ind_te in range(out_marginal.shape[0]):
            curr_exp_term = (out_marginal[ind_te]/(1-out_marginal[ind_te]))
            if self.smile_tau is not None:
                # SMILE (https://openreview.net/pdf?id=B1x62TNtDS)
                curr_exp_term = torch.clamp(curr_exp_term, min= torch.exp(torch.tensor(-self.smile_tau)), max = torch.exp(torch.tensor(self.smile_tau)))
            else:
                pass
            marg_prob += curr_exp_term
        marg_prob /= out_marginal.shape[0]
        marg_log_prob = torch.log(marg_prob)
        MI = joint_log_prob - marg_log_prob
        return MI

    @staticmethod
    def getting_accuracy(est_out, label):
        hard_out = est_out.clone().detach()
        hard_out[hard_out>0.5] = 1
        hard_out[hard_out<=0.5] = 0
        acc = (torch.eq(hard_out, label).sum())/hard_out.shape[0]
        return float(acc)

    def compute_one_mc_MI(self, clamping, classifier_lr, mb_size, tr_epoch, weight_decay):
        self.split_tr_te()
        trained_classifer = self.train_classifier(tr_epoch = tr_epoch, mb_size = mb_size, classifier_lr = classifier_lr, weight_decay=weight_decay)
        MI_est = self.test_classifier(trained_classifer=trained_classifer, clamping =clamping)
        return MI_est
    
    def MI_compute(self, clamping, classifier_lr, mb_size, tr_epoch, weight_decay):
        # since no split, one mc is sufficient
        avg_MI = 0
        one_mc = self.compute_one_mc_MI(clamping, classifier_lr, mb_size, tr_epoch, weight_decay)
        avg_MI += float(one_mc)
        avg_MI 
        return avg_MI

class Classifier(nn.Module):
    # from (w,y) -> real number
    # input (w,y) as (B, 2)
    def __init__(self, input_len):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.activ = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        y = self.activ(x)
        return y




    




