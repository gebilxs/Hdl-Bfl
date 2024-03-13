
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
        self.temperature = args.temperature
        self.loss_gt_weight = args.loss_gt_weight
        self.loss_kd_weight = args.loss_kd_weight

    def train(self):
        trainloadere = self.load_train_data()
        publicloadere = self.load_public_data()
        start_time = time.time()

        self.model.train()

        max_local_epochs = self.local_epochs
        logits = defaultdict(list)
        loss_kd =0
        for epoch in range(max_local_epochs):
            for i,(x,y) in enumerate(trainloadere):
                self.optimizer.zero_grad()
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss_gt = self.loss(output,y) * self.loss_gt_weight
                loss_gt.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.global_logits !=None:
                for i,(xp,yp) in enumerate(publicloadere): 
                    xp = xp.to(self.device)
                    yp = yp.to(self.device)
                    output = self.model(xp)
                    self.optimizer.zero_grad()
                    loss_kd = 0
                    for i, yy in enumerate(yp):
                        y_c = yy.item()
                        if not isinstance(self.global_logits[y_c], list):
                            teacher_logits = self.global_logits[y_c].to(self.device)
                                # 将输出和教师logits都除以温度
                            max_val, _ = torch.max(output, dim=1, keepdim=True)
                            min_val, _ = torch.min(output, dim=1, keepdim=True)

                                # 计算缩放因子和平移量，以将logits线性缩放到[-1, 1]
                            scale = 2 / (max_val - min_val)
                            shift = -1 - min_val * scale

                                # 应用缩放和平移
                            output_adjusted = output * scale + shift

                            soft_target = F.softmax(teacher_logits / self.temperature, dim=0)
                            soft_output = F.softmax(output_adjusted[i, :] / self.temperature, dim=0)

                            # print(f"teacher_logits:{soft_target}")
                            # print(f"student_logits:{soft_output}")
                                # 使用KL散度作为蒸馏损失
                            loss_kd += F.kl_div(soft_output.log(), soft_target, reduction='batchmean') * self.loss_kd_weight
                            
                    loss_kd.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                max_val, _ = torch.max(output, dim=1, keepdim=True)
                min_val, _ = torch.min(output, dim=1, keepdim=True)

                # 计算缩放因子和平移量，以将logits线性缩放到[-1, 1]
                scale = 2 / (max_val - min_val)
                shift = -1 - min_val * scale

                # 应用缩放和平移
                output_adjusted = output * scale + shift

                for i in range(output.shape[0]):
                    logits[i].append(output_adjusted[i,:].detach().data)
        self.logits = agg_func(logits)


    def set_logits(self, global_logits):
        self.global_logits = copy.deepcopy(global_logits)

    def train_metrics(self):
        trainloader = self.load_train_data()
        publicloadere = self.load_public_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        pub_train_num = 0
        losses = 0 
        total_loss =0
        total_loss_gt = 0
        total_loss_kd = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss_gt = self.loss(output, y) * self.loss_gt_weight
                train_num += y.shape[0]
                total_loss_gt += loss_gt.item() * y.shape[0]
            print(f"Client is {self.id}")
            print(f"total_loss_gt:{total_loss_gt/train_num}")

            if self.global_logits != None:
                # print(self.global_logits) 
                for i,(xp,yp) in enumerate(publicloadere):
                    xp = xp.to(self.device)
                    yp = yp.to(self.device)
                    output = self.model(xp)
                    loss_kd =0 
                    for i, yy in enumerate(yp):
                        y_c = yy.item()
                        
                        # 逐一处理所以dim=0
                        if not isinstance(self.global_logits[y_c], list):
                            teacher_logits1 = self.global_logits[y_c].to(self.device)
                            # 将输出和教师logits都除以温度
                            soft_target = F.softmax(teacher_logits1 / self.temperature , dim=0)
                            # soft_output = F.softmax(output[i, :] / temperature, dim=-1)
                            # 对在output上预测的结果进行数学投影
                            max_val, _ = torch.max(output, dim=1, keepdim=True)
                            min_val, _ = torch.min(output, dim=1, keepdim=True)

                            # 计算缩放因子和平移量，以将logits线性缩放到[-1, 1]
                            scale = 2 / (max_val - min_val)
                            shift = -1 - min_val * scale

                            # 应用缩放和平移
                            output_adjusted = output * scale + shift

                            log_soft_output = F.log_softmax(output_adjusted[i, :] / self.temperature, dim=0)
                            
                            # 使用KL散度作为蒸馏损失
                            loss_kd+=(F.kl_div(log_soft_output, soft_target, reduction='batchmean')) * self.loss_kd_weight
                    pub_train_num += yp.shape[0]
                    total_loss_kd += loss_kd.item() * pub_train_num

                print(f"total_loss_kd:{total_loss_kd/pub_train_num}")
            losses = total_loss_gt + total_loss_kd
            print(f"total_loss:{losses/(train_num)}")
            # TODO train_num can be more accurate
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