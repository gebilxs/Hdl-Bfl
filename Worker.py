from Devicebase import Device
import random
from Miner import Miner
from Validator import Validator
import time
import torch
import torch.nn as nn
import numpy as np
import copy
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn import metrics
class Worker(Device):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.model = copy.deepcopy(args.model_name)
        self.local_updates_rewards_per_transaction = 0
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0        
        ''' For malicious node '''
        self.variance_of_noises = None or []
        if args.even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        self.rs_test_acc = []
        self.rs_train_loss = []
        self.loss = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.logits = None
        self.global_logits = None

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.id = id
        self.check_signature = args.check_signature
        self.learning_rate_decay = args.learning_rate_decay
        self.train_samples = train_samples
        self.lamda = args.lamda
    def worker_reset_vars_for_new_round(self):
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.local_updates_rewards_per_transaction = 0
        self.has_added_block = False
        self.the_added_block = None
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        self.variance_of_noises.clear()
        self.round_end_time = 0

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def return_mined_block(self):
        return self.mined_block

    def associate_with_miner(self, to_associate_device_role):
        to_associate_device = vars(self)[f'{self.role}_associated_{to_associate_device_role}']
        # to_associate_device = to_associate_device_role
        shuffled_peer_list = list(self.peer_list)
        random.shuffle(shuffled_peer_list)
        for peer in shuffled_peer_list:
            # select the first found eligible device from a shuffled order
            if peer.return_role() == to_associate_device_role and peer.is_online():
                if not peer.return_id() in self.black_list:
                    to_associate_device = Miner(peer.args,peer.id,peer.train_samples,peer.test_samples)
                    to_associate_device.role = peer.role
        if not to_associate_device:
            # there is no device matching the required associated role in this device's peer list
            return False
        print(f"{self.role} {self.id} associated with {to_associate_device.return_role()} {to_associate_device.return_id()}")
        return to_associate_device
    
    def associate_with_validator(self,to_associate_device_role):
        to_associate_device = vars(self)[f'{self.role}_associated_{to_associate_device_role}']
        # to_associate_device = to_associate_device_role
        shuffled_peer_list = list(self.peer_list)
        random.shuffle(shuffled_peer_list)
        for peer in shuffled_peer_list:
            # select the first found eligible device from a shuffled order
            if peer.return_role() == to_associate_device_role and peer.is_online():
                if not peer.return_id() in self.black_list:
                    to_associate_device = Validator(peer.args,peer.id,peer.train_samples,peer.test_samples)
                    to_associate_device.role = peer.role
        if not to_associate_device:
            # there is no device matching the required associated role in this device's peer list
            return False
        print(f"{self.role} {self.id} associated with {to_associate_device.return_role()} {to_associate_device.return_id()}")
        print()
        return to_associate_device 
    
    def worker_local_update(self, rewards, log_files_folder_path_comm_round, comm_round, local_epochs=1):
        print(f"Worker {self.id} is doing local_update with computation power {self.computation_power} and link speed {round(self.link_speed,3)} bytes/s")
        # begin train
        trainloader = self.load_train_data()
        self.model.train()
        self.local_update_time = time.time()
        # local worker update by specified epochs
        # usually, if validator acception time is specified, local_epochs should be 1
        # logging maliciousness
        is_malicious_node = "M" if self.return_is_malicious() else "B"
        self.local_updates_rewards_per_transaction = 0

        for step in range(local_epochs):
            for i, (x, y) in enumerate(trainloader):
                # Data to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                output = self.model(x)
                loss = self.loss(output, y)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update rewards
                self.local_updates_rewards_per_transaction += rewards * (y.shape[0])
        
        # for epoch in range(local_epochs):
        #     for data, label in self.train_dl:
        #         data, label = data.to(self.dev), label.to(self.dev)
        #         preds = self.net(data)
        #         loss = self.loss_func(preds, label)
        #         loss.backward()
        #         self.opti.step()
        #         self.opti.zero_grad()
        #         self.local_updates_rewards_per_transaction += rewards * (label.shape[0])

        # record accuracies to find good -vh

            # with open(f"{log_files_folder_path_comm_round}/worker_{self.idx}_{is_malicious_node}_local_updating_accuracies_comm_{comm_round}.txt", "a") as file:
            #     file.write(f"{self.return_id()} epoch_{local_epochs+1} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
            self.local_total_epoch += 1
        try:
            self.local_update_time = (time.time() - self.local_update_time)/self.computation_power
        except:
            self.local_update_time = float('inf')

        # 如果是恶意节点增加噪声
        if self.is_malicious:
            self.net.apply(self.malicious_worker_add_noise_to_weights)
            print(f"malicious worker {self.idx} has added noise to its local updated weights before transmitting")
            with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
                file.write(f"{self.return_id()} {self.return_role()} {is_malicious_node} noise variances: {self.variance_of_noises}\n")
        # record accuracies to find good -vh

        # TODO 记录日志
        # with open(f"{log_files_folder_path_comm_round}/worker_final_local_accuracies_comm_{comm_round}.txt", "a") as file:
        #     file.write(f"{self.return_id()} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
        
        print(f"Done {local_epochs} epoch(s) and total {self.local_total_epoch} epochs")
        # self.local_train_parameters = self.net.state_dict()
        return self.local_update_time
    
    def FedDistill_worker_local_update(self, rewards, log_files_folder_path_comm_round, comm_round, local_epochs=1):
        print(f"Worker {self.id} is doing FedDistill_local_update with computation power {self.computation_power} and link speed {round(self.link_speed,3)} bytes/s")
        # begin train
        trainloader = self.load_train_data()
        self.model.train()
        self.local_update_time = time.time()
        # local worker update by specified epochs
        # usually, if validator acception time is specified, local_epochs should be 1
        # logging maliciousness
        is_malicious_node = "M" if self.return_is_malicious() else "B"
        self.local_updates_rewards_per_transaction = 0
        logits = {}

        for step in range(local_epochs):
            for i, (x, y) in enumerate(trainloader):
                # Data to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_logits != None:
                    logit_new = copy.deepcopy(output.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_logits[y_c]) != type([]):
                            logit_new[i, :] = self.global_logits[y_c].data
                    loss += self.loss_mse(logit_new, output) * self.lamda

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    if y_c not in logits:
                        logits[y_c] = []
                    logits[y_c].append(output[i, :].detach().data)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update rewards
                self.local_updates_rewards_per_transaction += rewards * (y.shape[0])

        # record accuracies to find good -vh
            # with open(f"{log_files_folder_path_comm_round}/worker_{self.idx}_{is_malicious_node}_local_updating_accuracies_comm_{comm_round}.txt", "a") as file:
            #     file.write(f"{self.return_id()} epoch_{local_epochs+1} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")

            self.local_total_epoch += 1
            self.evaluate_one_epoch()
        
        self.logits = agg_func(logits)
        try:
            self.local_update_time = (time.time() - self.local_update_time)/self.computation_power
        except:
            self.local_update_time = float('inf')

        # 如果是恶意节点增加噪声
        if self.is_malicious:
            self.net.apply(self.malicious_worker_add_noise_to_weights)
            print(f"malicious worker {self.idx} has added noise to its local updated weights before transmitting")
            with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
                file.write(f"{self.return_id()} {self.return_role()} {is_malicious_node} noise variances: {self.variance_of_noises}\n")
        # record accuracies to find good -vh

        # TODO 记录日志
        # with open(f"{log_files_folder_path_comm_round}/worker_final_local_accuracies_comm_{comm_round}.txt", "a") as file:
        #     file.write(f"{self.return_id()} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
        
        print(f"Done {local_epochs} epoch(s) and total {self.local_total_epoch} epochs")
        # self.local_train_parameters = self.net.state_dict()
        return self.local_update_time

    def return_local_updates_and_signature(self, comm_round):
        # local_total_accumulated_epochs_this_round also stands for the lastest_epoch_seq for this transaction(local params are calculated after this amount of local epochs in this round)
        # last_local_iteration(s)_spent_time may be recorded to determine calculating time? But what if nodes do not wish to disclose its computation power
        local_updates_dict = {'worker_device_idx': self.id, 'in_round_number': comm_round, 
                              "local_updates_params": copy.deepcopy(self.model), "local_updates_rewards": self.local_updates_rewards_per_transaction, 
                              "local_iteration(s)_spent_time": self.local_update_time, "local_total_accumulated_epochs_this_round": self.local_total_epoch, 
                              "worker_rsa_pub_key": self.return_rsa_pub_key(),"train_samples":self.train_samples,"logits":self.logits}
    
        local_updates_dict["worker_signature"] = self.sign_msg(sorted(local_updates_dict.items()))
        # add train sample message
        return local_updates_dict
    def set_deviceL_to_worker(self,logits):
        self.global_logits = logits
    def evaluate_one_epoch(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        # return test_acc, test_num, auc
        print(f"worker {self.id} acc is {test_acc/test_num}")

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