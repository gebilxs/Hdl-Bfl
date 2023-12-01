from Devicebase import Device
import random
from Miner import Miner
import copy
import time
import torch
import torch.nn as nn
from torch import optim
from hashlib import sha256

class Validator(Device):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.model = args.model_name
        self.validator_associated_worker_set = set()
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.validator_associated_miner = None
        # when validator directly accepts workers' updates
        self.unordered_arrival_time_accepted_worker_transactions = {}
        self.validator_accepted_broadcasted_worker_transactions = None or []
        self.final_transactions_queue_to_validate = {}
        self.post_validation_transactions_queue = None or []
        self.validator_threshold = args.validator_threshold
        self.validator_local_accuracy = None

        self.id = id
        if args.even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        # black_list stores device index rather than the object
        self.black_list = set()
        self.knock_out_rounds = args.knock_out_rounds
        self.lazy_worker_knock_out_rounds = args.lazy_worker_knock_out_rounds
        self.worker_accuracy_accross_records = {}
        self.has_added_block = False
        self.the_added_block = None
        self.is_malicious = args.is_malicious
        self.noise_variance = args.noise_variance
        self.check_signature = args.check_signature
        self.not_resync_chain = args.destroy_tx_in_block
        self.malicious_updates_discount = args.malicious_updates_discount
        self.validator_model = copy.deepcopy(self.model).to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.validator_optimizer = torch.optim.SGD(self.validator_model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.validator_optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def validator_reset_vars_for_new_round(self):
        self.validation_rewards_this_round = 0
        # self.accuracies_this_round = {}
        self.has_added_block = False
        self.the_added_block = None
        self.validator_associated_miner = None
        self.validator_local_accuracy = None
        self.validator_associated_worker_set.clear()
        #self.post_validation_transactions.clear()
        #self.broadcasted_post_validation_transactions.clear()
        self.unordered_arrival_time_accepted_worker_transactions.clear()
        self.final_transactions_queue_to_validate.clear()
        self.validator_accepted_broadcasted_worker_transactions.clear()
        self.post_validation_transactions_queue.clear()
        self.round_end_time = 0

    def return_associated_workers(self):
        return vars(self)[f'{self.role}_associated_worker_set']

    def add_device_to_association(self, to_add_device):
            role_key = f'{self.role}_associated_{to_add_device.return_role()}_set'
            if not hasattr(self, role_key):
                setattr(self, role_key, set())
            
            if not to_add_device.return_id() in self.black_list:
                getattr(self, role_key).add(to_add_device)
            else:
                print(f"WARNING: {to_add_device.return_id()} in {self.role} {self.id}'s black list. Not added by the {self.role}.")
                # if not to_add_device.return_id() in self.black_list:
                #     vars(self)[f'{self.role}_associated_{to_add_device.return_role()}_set'].add(to_add_device)
                # else:
                #     print(f"WARNING: {to_add_device.return_id()} in {self.role} {self.id}'s black list. Not added by the {self.role}.")
    
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
    
    def set_unordered_arrival_time_accepted_worker_transactions(self, unordered_transaction_arrival_queue):
        self.unordered_arrival_time_accepted_worker_transactions = unordered_transaction_arrival_queue

    def set_transaction_for_final_validating_queue(self, final_transactions_arrival_queue):
        self.final_transactions_queue_to_validate = final_transactions_arrival_queue
        # for i in self.final_transactions_queue_to_validate:
        #     if i.worker_device_idx = 

    def set_transaction_for_final_validating_queue(self, final_transactions_arrival_queue):
        self.final_transactions_queue_to_validate = []
        seen_worker_device_idx = set()
        for arrival_time, transaction in final_transactions_arrival_queue:
            worker_device_idx = transaction['worker_device_idx']
            
            # 如果这个worker_device_idx还没有被处理过，则添加到队列中
            if worker_device_idx not in seen_worker_device_idx:
                self.final_transactions_queue_to_validate.append((arrival_time, transaction))
                seen_worker_device_idx.add(worker_device_idx)
            else:
                # 可以在这里处理或记录重复的worker_device_idx，如果需要
                pass

    # 此时 self.final_transactions_queue_to_validate 包含了去重后的交易


    def validator_broadcast_worker_transactions(self,validators_this_round):
            # for peer in self.peer_list:
            #     if peer.is_online():
            #         if peer.return_role() == "validator":
            #             if not peer.return_id() in self.black_list:
            #                 print(f"validator {self.id} is broadcasting received validator transactions to validator {peer.return_id()}.")
            #                 final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator = copy.copy(self.unordered_arrival_time_accepted_worker_transactions)
            #                 # if offline, it's like the broadcasted transaction was not received, so skip a transaction
            #                 for arrival_time, tx in self.unordered_arrival_time_accepted_worker_transactions.items():
            #                     if not (self.online_switcher() and peer.online_switcher()):
            #                         del final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator[arrival_time]
            #                 # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received transactions(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the transactions itself received
            #                 peer.accept_validator_broadcasted_worker_transactions(self, final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)
                            
            #                 print(f"validator {self.id} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)} worker transactions to validator {peer.return_idx()}.")
            #             else:
            #                 print(f"Destination validator {peer.return_id()} is in this validator {self.id}'s black_list. broadcasting skipped for this dest validator.")
            for validator in validators_this_round:
                if not validator.return_id() in self.black_list:
                    print(f"validator {self.id} is broadcasting received validator transactions to validator {validator.return_id()}.")
                    final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator = copy.copy(self.unordered_arrival_time_accepted_worker_transactions)
                            # if offline, it's like the broadcasted transaction was not received, so skip a transaction
                    for arrival_time, tx in self.unordered_arrival_time_accepted_worker_transactions.items():
                        if not (self.online_switcher() and validator.online_switcher()):
                            del final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator[arrival_time]
                            # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received transactions(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the transactions itself received
                    validator.accept_validator_broadcasted_worker_transactions(self, final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)
                            
                    print(f"validator {self.id} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)} worker transactions to validator {validator.return_id()}.")
                else:
                    print(f"Destination validator {validator.return_id()} is in this validator {self.id}'s black_list. broadcasting skipped for this dest validator.")
    def accept_validator_broadcasted_worker_transactions(self, source_validator, unordered_transaction_arrival_queue_from_source_validator):
        if not source_validator.return_id() in self.black_list:
            self.validator_accepted_broadcasted_worker_transactions.append({'source_validator_link_speed': source_validator.return_link_speed(),'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_validator)})
            print(f"validator {self.id} has accepted worker transactions from validator {source_validator.return_id()}")
        else:
            print(f"Source validator {source_validator.return_id()} is in validator {self.id}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_worker_transactions(self):
        return self.validator_accepted_broadcasted_worker_transactions
    
    def return_unordered_arrival_time_accepted_worker_transactions(self):
        return self.unordered_arrival_time_accepted_worker_transactions
    
    def return_final_transactions_validating_queue(self):
        return self.final_transactions_queue_to_validate
    
    def validator_update_model_by_one_epoch_and_validate_local_accuracy(self):
        # return time spent
        print(f"validator {self.id} is performing one epoch of local update and validation")
        if self.computation_power == 0:
            print(f"validator {self.id} has computation power 0 and will not be able to complete this validation")
            return float('inf')
        else:
            trainloader = self.load_train_data()


            self.validator_model.train()
            # currently_used_lr = 0.005
            # for param_group in self.opti.param_groups:
            #     currently_used_lr = param_group['lr']
            # # by default use SGD. Did not implement others
            # if opti == 'SGD':
            #     validation_opti = optim.SGD(updated_model.parameters(), lr=currently_used_lr)
            local_validation_time = time.time()
            for i,(x,y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.validator_model(x)
                loss = self.loss(output,y)

                self.validator_optimizer.zero_grad()
                loss.backward()
                self.validator_optimizer.step()

            # validate by local test set
            self.evaluate_test_data()

            return (time.time() - local_validation_time)/self.computation_power
        
    def evaluate_test_data(self):
        testloader = self.load_test_data()  # Assuming you have a method to load test data
        self.validator_model.eval()  # Set the model to evaluation mode

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # Disable gradient computation during evaluation
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.validator_model(x)
                loss = self.loss(output, y)

                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()

        # Calculate average loss and accuracy
        average_loss = total_loss / total_predictions
        accuracy = 100 * correct_predictions / total_predictions
        self.validator_local_accuracy = accuracy
        
        print(f"Validator - {self.id} Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    def validate_worker_transaction(self, transaction_to_validate, rewards, log_files_folder_path, comm_round, malicious_validator_on):
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if self.computation_power == 0:
            print(f"validator {self.id} has computation power 0 and will not be able to validate this transaction in time")
            return False, False
        else:
            worker_transaction_device_idx = transaction_to_validate['worker_device_idx']
            if worker_transaction_device_idx in self.black_list:
                print(f"{worker_transaction_device_idx} is in validator's blacklist. Trasaction won't get validated.")
                return False, False
            validation_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_validate)
                del transaction_before_signed["worker_signature"]
                modulus = transaction_to_validate['worker_rsa_pub_key']["modulus"]
                pub_key = transaction_to_validate['worker_rsa_pub_key']["pub_key"]
                signature = transaction_to_validate["worker_signature"]
                # begin validation
                # 1 - verify signature
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.id}!")
                    transaction_to_validate['worker_signature_valid'] = True
                else:
                    print(f"Signature invalid. Transaction from worker {worker_transaction_device_idx} does NOT pass verification.")
                    # will also add sig not verified transaction due to the validator's verification effort and its rewards needs to be recorded in the block
                    transaction_to_validate['worker_signature_valid'] = False
            else:
                print(f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.id}!")
                transaction_to_validate['worker_signature_valid'] = True
            # 2 - validate worker's local_updates_params if worker's signature is valid
            if transaction_to_validate['worker_signature_valid']:
                # accuracy validated by worker's update
                accuracy_by_worker_update_using_own_data = self.validate_model_weights(transaction_to_validate["local_updates_params"])
                # if worker's accuracy larger, or lower but the difference falls within the validator threshold value, meaning worker's updated model favors validator's dataset,
                # so their updates are in the same direction - True, otherwise False. We do not consider the accuracy gap so far, meaning if worker's update is way too good, it is still fine
                
                # print(f'validator updated model accuracy - {self.validator_local_accuracy}')
                # print(f"After applying worker's update, model accuracy becomes - {accuracy_by_worker_update_using_own_data}")
                # record their accuracies and difference for choosing a good validator threshold
                is_malicious_validator = "M" if self.is_malicious else "B"
                with open(f"{log_files_folder_path_comm_round}/validator_{self.id}_{is_malicious_validator}_validation_records_comm_{comm_round}.txt", "a") as file:
                    is_malicious_node = "M" if self.devices_dict[worker_transaction_device_idx].return_is_malicious() else "B"
                    file.write(f"{accuracy_by_worker_update_using_own_data - self.validator_local_accuracy}: validator {self.return_id()} {is_malicious_validator} in round {comm_round} evluating worker {worker_transaction_device_idx}, diff = v_acc:{self.validator_local_accuracy} - w_acc:{accuracy_by_worker_update_using_own_data} {worker_transaction_device_idx}_maliciousness: {is_malicious_node}\n")
                
                print(f"两者差为：{accuracy_by_worker_update_using_own_data - self.validator_local_accuracy}")

                if accuracy_by_worker_update_using_own_data - self.validator_local_accuracy < (self.validator_threshold * -70):
                    transaction_to_validate['update_direction'] = False
                    print(f"NOTE: worker {worker_transaction_device_idx}'s updates is deemed as suspiciously malicious by validator {self.id}")
                    # is it right?
                    if not self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(f"Warning - {worker_transaction_device_idx} is benign and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.id} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_negative_malicious_nodes_inside_caught.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.id} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                else:
                    transaction_to_validate['update_direction'] = True
                    print(f"worker {worker_transaction_device_idx}'s' updates is deemed as GOOD by validator {self.id}")
                    # is it right?
                    if self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(f"Warning - {worker_transaction_device_idx} is malicious and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.id} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_positive_good_nodes_inside_correct.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.id} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                if self.is_malicious and malicious_validator_on:
                    old_voting = transaction_to_validate['update_direction']
                    transaction_to_validate['update_direction'] = not transaction_to_validate['update_direction']
                    with open(f"{log_files_folder_path_comm_round}/malicious_validator_log.txt", 'a') as file:
                        file.write(f"malicious validator {self.id} has flipped the voting of worker {worker_transaction_device_idx} from {old_voting} to {transaction_to_validate['update_direction']} in round {comm_round}\n")
                transaction_to_validate['validation_rewards'] = rewards
            else:
                transaction_to_validate['update_direction'] = 'N/A'
                transaction_to_validate['validation_rewards'] = 0
            transaction_to_validate['validation_done_by'] = self.id
            validation_time = (time.time() - validation_time)/self.computation_power
            transaction_to_validate['validation_time'] = validation_time
            transaction_to_validate['validator_rsa_pub_key'] = self.return_rsa_pub_key()
            # assume signing done in negligible time
            transaction_to_validate["validator_signature"] = self.sign_msg(sorted(transaction_to_validate.items()))
            return validation_time, transaction_to_validate

# 通过worker的模型参数在validator上面进行验证
    def validate_model_weights(self, weights_to_eval=None):
        # 不一定一定是这个model
        validator_worker_model = self.model
        testloader = self.load_test_data()
        validator_worker_model.eval()
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            if weights_to_eval:
                state_dict = weights_to_eval.state_dict()  # 提取状态字典
                validator_worker_model.load_state_dict(state_dict, strict=True)
                # validate_model.load_state_dict(weights_to_eval, strict=True)
            else:
                validator_worker_model.load_state_dict(self.global_parameters, strict=True)

            for x,y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = validator_worker_model(x)
                # loss = self.loss(output,y)

                # total_loss +=loss.item()
                # output = torch.argmax(output.data, dim=1)
                # sum_accu += (output == y).float().mean()
                # num += 1

                _, predicted = torch.max(output.data, 1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()
            
            # average_loss = total_loss / len(testloader)
            accuracy = 100 * correct_predictions / total_predictions

        print(f"通过worker的模型参数在validator Accuracy: {accuracy:.2f}%")
        # print (f"通过worker的模型参数在validator上面进行验证准确率是：{100*sum_accu/num}%")
        return accuracy
    
    def add_post_validation_transaction_to_queue(self, transaction_to_add):
        self.post_validation_transactions_queue.append(transaction_to_add)