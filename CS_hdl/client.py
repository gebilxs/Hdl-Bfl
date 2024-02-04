
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
class clientHdl(Client):
    def __init__(self, args, id, train_samples, test_samples,pubilc_samples,**kwargs):
        super().__init__(args, id, train_samples, test_samples,pubilc_samples,**kwargs)

        self.logits = None
        self.global_logits = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

    def train(self):
        trainloadere = self.load_train_data()
        publicloadere = self.load_public_data()
        start_time = time.time()

        self.model.train()

        max_local_epochs = self.local_epochs
        logits = defaultdict(list)
        for epoch in range(max_local_epochs):
            for i,(x,y) in enumerate(trainloadere):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                loss = self.loss(output,y)

                # if self.global_logits!=None:
                #     logit_new = copy.deepcopy(output.detach())
                #     for i, yy in enumerate(y):
                #         y_c = yy.item()
                #         if type(self.global_logits[y_c]) != type([]):
                #             logit_new[i, :] = self.global_logits[y_c].data
                #     loss += self.loss_mse(logit_new, output) * self.lamda
            # 知识蒸馏损失
                if self.global_logits is not None:
                    temperature = 1.0  # 温度参数，可调整
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if not isinstance(self.global_logits[y_c], list):
                            teacher_logits = self.global_logits[y_c].to(self.device)
                            # 将输出和教师logits都除以温度
                            soft_target = F.softmax(teacher_logits / temperature, dim=0)
                            soft_output = F.softmax(output[i, :] / temperature, dim=0)
                            # 使用KL散度作为蒸馏损失
                            distillation_loss = F.kl_div(soft_output.log(), soft_target, reduction='batchmean')
                            # 综合两部分损失
                            loss += distillation_loss * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
    # use public_data to predict
        self.model.eval()
        with torch.no_grad():
            logits = defaultdict(list)
            for x, _ in publicloadere:
                # if x.shape[1] == 3:  # 假设x的形状为[N, C, H, W]
                #     x = x.mean(dim=1, keepdim=True)
                x = x.to(self.device)
                output = self.model(x)
                for i in range(output.shape[0]):
                    logits[i].append(output[i,:].detach().data)
        self.logits = agg_func(logits)


    def set_logits(self, global_logits):
        self.global_logits = copy.deepcopy(global_logits)

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_logits is not None:
                    temperature = 10.0  # 温度参数，可调整
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if not isinstance(self.global_logits[y_c], list):
                            teacher_logits = self.global_logits[y_c].to(self.device)
                            # 将输出和教师logits都除以温度
                            soft_target = F.softmax(teacher_logits / temperature, dim=0)
                            soft_output = F.softmax(output[i, :] / temperature, dim=0)
                            # 使用KL散度作为蒸馏损失
                            distillation_loss = F.kl_div(soft_output.log(), soft_target, reduction='batchmean')
                            # 综合两部分损失
                            loss += distillation_loss * self.lamda
                    
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
# https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L205
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