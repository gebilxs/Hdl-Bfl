
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



    def train(self):
        trainloadere = self.load_train_data()
        publicloadere = self.load_public_data()
        start_time = time.time()

        self.model.train()
        # print(self.modelname)
        max_local_epochs = self.local_epochs
        logits = defaultdict(list)
        self.projectors = nn.ModuleDict({
                    str(i): self._create_projector(i,self.modelname)
                    for i in range(1, 5)  # 假设有4个阶段
                })
        # 根据stage 进行分析
        # stage_list = self.stage_info(self.modelname, self.stage)
        loss_ofa = 0
        loss_gt = 0
        loss_kd = 0
        
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
                loss_gt = loss_gt * self.gt_loss_weight
                # 普通知识蒸馏
                # if self.global_logits!=None:
                #     logit_new = copy.deepcopy(output.detach())
                #     for i, yy in enumerate(y):
                #         y_c = yy.item()
                #         if type(self.global_logits[y_c]) != type([]):
                #             logit_new[i, :] = self.global_logits[y_c].data
                #     loss += self.loss_mse(logit_new, output) * self.lamda
                # 利用公共数据集进行知识蒸馏损失
                
                # if self.global_logits is not None:
                #     temperature = 1.0  # 温度参数，可调整
                #     for i, yy in enumerate(y):
                #         y_c = yy.item()
                #         if not isinstance(self.global_logits[y_c], list):
                #             teacher_logits = self.global_logits[y_c].to(self.device)
                #             # 将输出和教师logits都除以温度
                #             soft_target = F.softmax(teacher_logits / temperature, dim=0)
                #             soft_output = F.softmax(output[i, :] / temperature, dim=0)
                #             # 使用KL散度作为蒸馏损失
                #             distillation_loss = F.kl_div(soft_output.log(), soft_target, reduction='batchmean')
                #             # 综合两部分损失
                #             loss += distillation_loss * self.lamda

                if self.global_logits is not None:
                    ofa_losses = []
                    if len(y.shape) != 1:
                        target_mask = F.one_hot(y.argmax(-1),self.num_classes)
                    else:
                        target_mask = F.one_hot(y, self.num_classes)

                    for stage, eps in zip(self.stage_count, self.eps):
                        model_n = self.stage_info(self.modelname, stage).to(self.device)
                        feat = model_n(x) # 获取特征
                        logits_student = self.projectors[str(stage)].to(self.device)(feat) # 获取学生模型输出
                        teacher_logits = torch.stack([self.global_logits[yi.item()] for yi in y])
                        ofa_losses.append(self.ofa_loss(logits_student,teacher_logits,target_mask,eps,self.ofa_temperature))
                    
                    loss_ofa = sum(ofa_losses) * self.ofa_loss_weight
                    # teacher_logits1 = torch.stack([self.global_logits[yi.item()] for yi in y])
                    # loss_kd =  F.kl_div(F.log_softmax(output, dim=1), F.softmax(teacher_logits1, dim=1), reduction='batchmean') * self.kd_loss_weight
                    loss_kd1 = []
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        temperature = self.temperature
                        if not isinstance(self.global_logits[y_c], list):
                            teacher_logits1 = self.global_logits[y_c].to(self.device)
                            # 将输出和教师logits都除以温度
                            soft_target = F.softmax(teacher_logits1 / temperature, dim=-1)
                            soft_output = F.softmax(output[i, :] / temperature, dim=-1)
                            # 使用KL散度作为蒸馏损失
                            loss_kd1.append(F.kl_div(soft_output.log(), soft_target, reduction='batchmean'))
                    loss_kd = sum(loss_kd1) * self.kd_loss_weight
                    
                loss = loss_ofa +loss_gt + loss_kd
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

    # def train_metrics(self):
    #     trainloader = self.load_train_data()
    #     # self.model = self.load_model('model')
    #     self.model.to(self.device)
    #     self.model.eval()

    #     train_num = 0
    #     losses = 0
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             loss = self.loss(output, y)

    #             if self.global_logits is not None:
    #                 temperature = 10.0  # 温度参数，可调整
    #                 for i, yy in enumerate(y):
    #                     y_c = yy.item()
    #                     if not isinstance(self.global_logits[y_c], list):
    #                         teacher_logits = self.global_logits[y_c].to(self.device)
    #                         # 将输出和教师logits都除以温度
    #                         soft_target = F.softmax(teacher_logits / temperature, dim=0)
    #                         soft_output = F.softmax(output[i, :] / temperature, dim=0)
    #                         # 使用KL散度作为蒸馏损失
    #                         distillation_loss = F.kl_div(soft_output.log(), soft_target, reduction='batchmean')
    #                         # 综合两部分损失
    #                         loss += distillation_loss * self.lamda
                    
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]

    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')

    #     return losses, train_num
        
    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        total_loss_gt = 0
        total_loss_ofa = 0
        total_loss_kd = 0
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

                
                if self.global_logits is not None:
                    if len(y.shape) != 1:
                        target_mask = F.one_hot(y.argmax(-1), self.num_classes)
                    else:
                        target_mask = F.one_hot(y, self.num_classes)
                # LOSS OFA
                    ofa_losses = []
                    for stage, eps in zip(self.stage_count, self.eps):
                        model_n = self.stage_info(self.modelname, stage).to(self.device)
                        feat = model_n(x)  # 获取特征
                        # print(self.modelname)
                        # print(stage)
                        logits_student = self.projectors[str(stage)].to(self.device)(feat)  # 获取学生模型输出
                        # 注意：这里需要确保global_logits已经转换为适合每个阶段的格式
                        # teacher_logits = torch.tensor(np.array(self.global_logits))
                        # print(self.global_logits.shape)
                        teacher_logits = torch.stack([self.global_logits[yi.item()] for yi in y])
                        ofa_losses.append(self.ofa_loss(logits_student, teacher_logits, target_mask, eps, self.ofa_temperature))
                # LOSS KD
                    loss_kd1 = []
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        temperature = self.temperature
                        if not isinstance(self.global_logits[y_c], list):
                            teacher_logits1 = self.global_logits[y_c].to(self.device)
                            # 将输出和教师logits都除以温度
                            soft_target = F.softmax(teacher_logits1 / temperature, dim=-1)
                            soft_output = F.softmax(output[i, :] / temperature, dim=-1)
                            # 使用KL散度作为蒸馏损失
                            loss_kd1.append(F.kl_div(soft_output.log(), soft_target, reduction='batchmean'))

                    loss_kd = sum(loss_kd1) * self.kd_loss_weight
                    loss_ofa = sum(ofa_losses) * self.ofa_loss_weight
                else:
                    loss_ofa = torch.tensor(0.0).to(self.device)
                    loss_kd = torch.tensor(0.0).to(self.device)

                
                train_num += y.size(0)
                total_loss_gt += loss_gt.item() * y.size(0)
                total_loss_ofa += loss_ofa.item() * y.size(0)
                total_loss_kd += loss_kd.item() * y.size(0)
                loss = total_loss_gt + total_loss_ofa + total_loss_kd

        average_loss_gt = total_loss_gt / train_num
        average_loss_ofa = total_loss_ofa / train_num
        average_loss_kd = total_loss_kd / train_num
        print(f"client{self.id}")
        print(f"Ground Truth Loss (loss_gt): {average_loss_gt}")
        print(f"OFA Loss (loss_ofa): {average_loss_ofa}")
        print(f"Knowledge Distillation Loss (loss_kd): {average_loss_kd}")

        total_average_loss = (average_loss_gt + average_loss_ofa + average_loss_kd)


        return loss, train_num


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
        
    def ofa_loss(self,logits_student, logits_teacher, target_mask, eps, temperature=1.):
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


    # 通用识别比较困难
    # def _get_feature_dim(self, model_n, modelname, stage):
    #     if modelname == 'googlenet':
    #         # Manually define the output feature dimension for each stage of GoogLeNet.
    #         # These dimensions are determined based on the structure of the GoogLeNet model.
    #         # Note: The actual values here might need to be adjusted based on the specific implementation of GoogLeNet you are using.
    #         stage_dims = {
    #             1: 192,  # After conv3, before maxpool2
    #             2: 256,  # Output of inception3a
    #             3: 512,  # Output of inception4a
    #             4: 1024, # Output before avgpool, after inception5b
    #         }
    #         return stage_dims[stage]
    #     else:
    #         # For other models, use the general approach to determine feature dimension.
    #         def get_dim(layer):
    #             if isinstance(layer, nn.Conv2d):
    #                 return layer.out_channels
    #             elif isinstance(layer, nn.Linear):
    #                 return layer.out_features
    #             elif isinstance(layer, nn.Sequential):
    #                 for sub_layer in reversed(layer):
    #                     result = get_dim(sub_layer)
    #                     if result is not None:
    #                         return result
    #             elif isinstance(layer, nn.AdaptiveAvgPool2d):
    #                 # If we encounter an AdaptiveAvgPool2d, we assume the feature dimension
    #                 # is equal to the number of output channels of the last Conv2d layer.
    #                 for sub_layer in reversed(layer._modules.values()):
    #                     result = get_dim(sub_layer)
    #                     if result is not None:
    #                         return result
    #             return None

    #         feature_dim = get_dim(model_n)
    #         if feature_dim is not None:
    #             return feature_dim

    #     raise NotImplementedError("Could not determine feature dimension from the model stage.")
    

    def _create_projector(self, stage,modelname):
        # 这个函数基于每个阶段的模型输出维度创建一个适合的projector
        model_n = self.stage_info(self.modelname, stage)
        # print(model_n)
        # 需要根据model_n的具体输出维度来定制projector，这里只是一个示例
        feature_dim = self._get_feature_dim(model_n,modelname,stage)
        intermediate_dim = feature_dim // 2  # 示例：中间层维度设置为输出维度的一半

        projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),  # 添加Batch Normalization层
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Dropout(0.5),  # 添加Dropout，丢弃率设置为0.5
            nn.Linear(intermediate_dim, intermediate_dim // 2),  
            nn.ReLU(),  # 再次使用ReLU激活函数
            nn.Linear(intermediate_dim // 2, self.num_classes)  # 最终输出层
    )
        return projector
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