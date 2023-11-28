from Devicebase import Device
import random
from Miner import Miner
import copy
class Validator(Device):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
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
        # when validator directly accepts workers' updates
        self.unordered_arrival_time_accepted_worker_transactions = {}
        self.validator_accepted_broadcasted_worker_transactions = None or []
        self.final_transactions_queue_to_validate = {}
        self.post_validation_transactions_queue = None or []
        self.validator_threshold = args.validator_threshold
        self.validator_local_accuracy = None
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
