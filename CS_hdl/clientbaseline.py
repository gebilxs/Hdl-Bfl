
from clientbase import Client
import torch.nn as nn
import time
import torch
from collections import defaultdict
import copy
import torch.nn.functional as F
import random
import numpy as np
import os
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
class clientHbase(Client):
    def __init__(self, args, id, train_samples, test_samples,pubilc_samples,**kwargs):
        super().__init__(args, id, train_samples, test_samples,pubilc_samples,**kwargs)
        self.logits = None
        self.global_logits = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        # ofa 参数
        self.temperature = args.temperature
        self.eps = args.ofa_eps
        self.stage_count = args.ofa_stage
        self.ofa_loss_weight = args.ofa_loss_weight
        self.ofa_temperature = args.ofa_temperature
        self.gt_loss_weight = args.loss_gt_weight
        self.kd_loss_weight = args.loss_kd_weight

        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.clip_grad = args.clip_grad

    def train(self):
        trainloadere = self.load_train_data()
        start_time = time.time()

        self.model.train()
        # print(self.modelname)
        max_local_epochs = self.local_epochs
        logits = defaultdict(list)

        loss_gt = 0
        loss_kd = 0
        total_loss_gt = 0
        total_loss_kd = 0
        for epoch in range(max_local_epochs):
            for i,(x,y) in enumerate(trainloadere):
                self.optimizer.zero_grad()
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss_gt = self.loss(output,y) # 计算交叉熵损失 loss_gt
                total_loss_gt += loss_gt * self.gt_loss_weight


                if self.global_logits != None:
                    
                    logit_new = copy.deepcopy(output.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        logit_new[i,:] = self.global_logits[y_c].data.to(self.device)
                        
                    loss_kd = self.loss_mse(logit_new,output) *self.kd_loss_weight
                    total_loss_kd += loss_kd

                for i,yy in enumerate(y):
                    yc = yy.item()
                    logits[yc].append(output[i,:].detach().data)
                (loss_kd  + loss_gt).backward()
                self.optimizer.step()
            self.logits = agg_func(logits)
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
            torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.learning_rate_decay_gamma)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
    def set_logits(self, global_logits):
        self.global_logits = copy.deepcopy(global_logits)
        
    def train_metrics(self):
        trainloader = self.load_train_data()
        # publicloadere = self.load_public_data()
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        total_loss_gt = 0
        total_loss_ofa = 0
        total_loss_kd = 0
        train_public_num = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                # LOSS GT
                loss_gt = self.loss(output, y) * self.gt_loss_weight
                total_loss_gt += loss_gt.item() * y.size(0)
                train_num += y.size(0)
                
                if self.global_logits is not None:
                    logit_new = copy.deepcopy(output.detach())
                    for i,yy in enumerate(y):    
                        y_c = yy.item()
                        logit_new[i,:] = self.global_logits[y_c].data.to(self.device)
                    loss_kd = self.loss_mse(logit_new,output) *self.kd_loss_weight
                    total_loss_kd += loss_kd.item() * y.size(0)
                

        average_loss_gt = total_loss_gt / train_num
        average_loss_kd = total_loss_kd / train_num

        print(f"client{self.id}")
        print(f"Ground Truth Loss (loss_gt): {average_loss_gt}")
        print(f"Knowledge Distillation Loss (loss_kd): {average_loss_kd}")

        total_average_loss = (average_loss_gt  + average_loss_kd) * train_num

    
        return total_average_loss, train_num
    

def agg_func(logits):
    """
    Returns the average of the weights.
    """

    for [label, logit_list] in logits.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            logits[label] = logit / len(logit_list)
        else:
            logits[label] = logit_list[0]

    return logits