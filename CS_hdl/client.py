
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
from timm.models.layers import _assert, trunc_normal_
SEED = 2024
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

        # ofa 参数
        self.temperature = args.temperature
        self.eps = args.ofa_eps
        self.stage_count = args.ofa_stage
        self.ofa_loss_weight = args.ofa_loss_weight
        self.ofa_temperature = args.ofa_temperature
        self.gt_loss_weight = args.loss_gt_weight
        self.kd_loss_weight = args.loss_kd_weight
        self.projectors = nn.ModuleDict({
                    str(i): self._create_projector(i,self.modelname)
                    for i in range(1, 5)  # 假设有4个阶段
                })
        self.projectors.apply(self.init_weights)
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.clip_grad = args.clip_grad
    def train(self):
        trainloadere = self.load_train_data()
        start_time = time.time()

        self.model.train()
        # print(self.modelname)
        max_local_epochs = self.local_epochs
        logits = defaultdict(list)

        # 根据stage 进行分析
        # stage_list = self.stage_info(self.modelname, self.stage)
        loss_ofa = 0
        loss_gt = 0
        loss_kd = 0
        total_loss_gt = 0
        total_loss_kd = 0
        total_loss_ofa = 0
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
                    if len(y.shape) != 1:
                        target_mask = F.one_hot(y.argmax(-1).long(), self.num_classes)
                    else:
                        target_mask = F.one_hot(y.long(), self.num_classes)

                    ofa_losses_recent = []
                    for stage, eps in zip(self.stage_count, self.eps):
                            model_n = self.stage_info(self.modelname, stage).to(self.device)
                            feat = model_n(x) # 获取特征
                            logits_student = self.projectors[str(stage)].to(self.device)(feat) # 获取学生模型输出

                            teacher_logits = torch.stack([self.global_logits[yi.item()] for yi in y])
                            ofa_losses_recent.append(self.ofa_loss(logits_student,teacher_logits,target_mask,eps,self.ofa_temperature))

                    logit_new = copy.deepcopy(output.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        logit_new[i,:] = self.global_logits[y_c].data.to(self.device)
                        
                    loss_kd = self.loss_mse(logit_new,output) *self.kd_loss_weight
                    loss_ofa = sum(ofa_losses_recent) * self.ofa_loss_weight
                    total_loss_kd += loss_kd
                    total_loss_ofa += loss_ofa
                
                    
                for i,yy in enumerate(y):
                    yc = yy.item()
                    logits[yc].append(output[i,:].detach().data)
                (loss_kd + loss_ofa + loss_gt).backward()
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
                    
                    if len(y.shape) != 1:
                        target_mask = F.one_hot(y.argmax(-1).long(), self.num_classes)
                    else:
                        target_mask = F.one_hot(y.long(), self.num_classes)

                    # LOSS OFA   
                    loss_ofa_recent = []
                    for stage, eps in zip(self.stage_count, self.eps):
                        model_n = self.stage_info(self.modelname, stage).to(self.device)
                        feat = model_n(x)  # 获取特征
                        logits_student = self.projectors[str(stage)].to(self.device)(feat)  # 获取学生模型输出
                            # 注意：这里需要确保global_logits已经转换为适合每个阶段的格式
                            # teacher_logits = torch.tensor(np.array(self.global_logits))
                            # print(self.global_logits.shape)
                        teacher_logits = torch.stack([self.global_logits[yi.item()] for yi in y])
                        loss_ofa_recent.append(self.ofa_loss(logits_student, teacher_logits, target_mask, eps, self.ofa_temperature))
                 

                    logit_new = copy.deepcopy(output.detach())
                    for i,yy in enumerate(y):    
                        y_c = yy.item()
                        logit_new[i,:] = self.global_logits[y_c].data.to(self.device)
                    loss_kd = self.loss_mse(logit_new,output) *self.kd_loss_weight
                          


                    loss_ofa = sum(loss_ofa_recent) * self.ofa_loss_weight
                        
                    train_public_num += y.size(0)
                    total_loss_ofa += loss_ofa.item() * y.size(0)
                    total_loss_kd += loss_kd.item() * y.size(0)
                
                


        average_loss_gt = total_loss_gt / train_num if train_num > 0 else 0
        average_loss_ofa = total_loss_ofa / train_public_num if train_public_num > 0 else 0
        average_loss_kd = total_loss_kd / train_public_num if train_public_num > 0 else 0

        print(f"client{self.id}")
        print(f"Ground Truth Loss (loss_gt): {average_loss_gt}")
        print(f"OFA Loss (loss_ofa): {average_loss_ofa}")
        print(f"Knowledge Distillation Loss (loss_kd): {average_loss_kd}")

        total_average_loss = (average_loss_gt + average_loss_ofa + average_loss_kd) * train_num

    
        return total_average_loss, train_num


    def stage_info(self, modelname, stage):
        # Retrieve the model

        # Define stages for each model
        if modelname == 'resnet':
            stages = {
                1: nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool),
                2: nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1),
                3: nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1, self.model.layer2),
                4: nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1, self.model.layer2, self.model.layer3),
            }
        elif modelname == 'shufflenet':
            # ShuffleNet stages might be defined differently as they have a different block structure.
            stages = {
                1: nn.Sequential(self.model.conv1, self.model.maxpool),
                2: nn.Sequential(self.model.conv1, self.model.maxpool, self.model.stage2),
                3: nn.Sequential(self.model.conv1, self.model.maxpool, self.model.stage2, self.model.stage3),
                4: nn.Sequential(self.model.conv1, self.model.maxpool, self.model.stage2, self.model.stage3, self.model.stage4),
            }
        elif modelname == 'googlenet':
            stages = {
                1: nn.Sequential(
                    self.model.conv1,
                    self.model.maxpool1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.maxpool2
                ),  # 输出特征维度: 192
                2: nn.Sequential(
                    self.model.conv1,
                    self.model.maxpool1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.maxpool2,
                    self.model.inception3a,
                    self.model.inception3b,
                    self.model.maxpool3
                ),  # 需要计算输出特征维度
                3: nn.Sequential(
                    self.model.conv1,
                    self.model.maxpool1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.maxpool2,
                    self.model.inception3a,
                    self.model.inception3b,
                    self.model.maxpool3,
                    self.model.inception4a,
                    self.model.inception4b,
                    self.model.inception4c,
                    self.model.inception4d,
                    self.model.inception4e,
                    self.model.maxpool4
                ),  # 需要计算输出特征维度
                4: nn.Sequential(
                    self.model.conv1,
                    self.model.maxpool1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.maxpool2,
                    self.model.inception3a,
                    self.model.inception3b,
                    self.model.maxpool3,
                    self.model.inception4a,
                    self.model.inception4b,
                    self.model.inception4c,
                    self.model.inception4d,
                    self.model.inception4e,
                    self.model.maxpool4,
                    self.model.inception5a,
                    self.model.inception5b,
                    self.model.avgpool
                )  # 输出特征维度: 1024
}


        elif modelname == 'alexnet':
            stages = {
                1: nn.Sequential(
                    self.model.conv1,
                ),
                2: nn.Sequential(
                    self.model.conv1,
                    self.model.conv2,
                ),
                3: nn.Sequential(
                    self.model.conv1,
                    self.model.conv2,
                    self.model.conv3,
                ),
                4: nn.Sequential(
                    self.model.conv1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.avgpool,
                    # nn.Flatten(),  # 添加Flatten层，因为之后接的是全连接层
                    # self.model.fc,
                ),
            }
        # maybe for cifar10
        # elif modelname == 'alexnet':
        #     stages = {
        #         1: nn.Sequential(self.model.features[:3]),
        #         2: nn.Sequential(self.model.features[:6]),
        #         3: nn.Sequential(self.model.features[:8]),
        #         4: nn.Sequential(self.model.features[:], self.model.avgpool),
        #     }
        else:
            raise ValueError(f"Model {modelname} not supported")

        # Return the requested stage
        if stage in stages:
            return stages[stage]
        else:
            raise ValueError(f"Stage {stage} not defined for model {modelname}")
        
    def ofa_loss(self,logits_student, logits_teacher, target_mask, eps, temperature):

        logits_teacher -= torch.max(logits_teacher, dim=1, keepdim=True).values
        logits_student -= torch.max(logits_student, dim=1, keepdim=True).values


        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        prod = (pred_teacher + target_mask) ** eps
        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
        return loss.mean()

    def _get_feature_dim(self, model_n, modelname, stage):
        if modelname == 'googlenet':
            stage_dims = {
                1: 192,  # After conv3, before maxpool2
                2: 480,  # Output of inception3a
                3: 832,  # Output of inception4a
                4: 1024, # Output before avgpool, after inception5b
            }
            return stage_dims[stage]
        elif modelname == 'resnet':
            stage_dims = {
                1: 64,   # After maxpool
                2: 64,   # Output of layer1
                3: 128,  # Output of layer2
                4: 256,  # Output of layer3
                5: 512,  # Output of layer4, before avgpool
            }
            return stage_dims[stage]
        elif modelname == 'shufflenet':
            stage_dims = {
                1: 24,  # After maxpool
                2: 116, # Output of stage2
                3: 232, # Output of stage3
                4: 464, # Output of stage4, before conv5
            }
            return stage_dims[stage]
        elif modelname == 'alexnet':
            stage_dims = {
                1: 64,   # After conv1
                2: 128,  # After conv2
                3: 256,  # After conv3
                4: 256,  # After avgpool, before entering fc (assuming fc input feature size)
            }
            return stage_dims[stage]
        else:
            raise NotImplementedError(f"Model {modelname} not supported for feature dimension extraction.")
    
    def _create_projector(self, stage,modelname):
        model_n = self.stage_info(self.modelname, stage)
        # print(model_n)
        # 需要根据model_n的具体输出维度来定制projector，这里只是一个示例
        feature_dim = self._get_feature_dim(model_n,modelname,stage)
        intermediate_dim = feature_dim // 2  # 示例：中间层维度设置为输出维度的一半

        projector = nn.Sequential(
            nn.Conv2d(feature_dim, intermediate_dim, kernel_size=3, stride=1, padding=1),  # 3x3卷积层
            nn.BatchNorm2d(intermediate_dim),  # 2D批量归一化
            nn.ReLU(inplace=True),      
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # nn.Dropout(0.5),  # 添加Dropout，丢弃率设置为0.5
            nn.Linear(intermediate_dim, intermediate_dim // 2),  
            nn.ReLU(),  # 再次使用ReLU激活函数
            nn.Linear(intermediate_dim // 2, self.num_classes)  # 最终输出层
    )
        return projector
    
    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
            
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