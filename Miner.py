from Devicebase import Device
import copy




class Miner(Device):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.miner_associated_worker_set = set()
        self.miner_associated_validator_set = set()
        # dict cannot be added to set()
        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.miner_acception_wait_time = args.miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = args.miner_accepted_transactions_size_limit
        # when miner directly accepts validators' updates
        self.unordered_arrival_time_accepted_validator_transactions = {}
        self.miner_accepted_broadcasted_validator_transactions = None or []
        self.final_candidate_transactions_queue_to_mine = {}
        self.block_generation_time_point = None
        self.unordered_propagated_block_processing_queue = {} # pure simulation queue and does not exist in real distributed system

    def miner_reset_vars_for_new_round(self):
        self.miner_associated_worker_set.clear()
        self.miner_associated_validator_set.clear()
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()
        # self.unconfirmmed_validator_transactions.clear()
        # self.validator_accepted_broadcasted_worker_transactions.clear()
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.has_added_block = False
        self.the_added_block = None
        self.unordered_arrival_time_accepted_validator_transactions.clear()
        self.miner_accepted_broadcasted_validator_transactions.clear()
        self.block_generation_time_point = None
#		self.block_to_add = None
        self.unordered_propagated_block_processing_queue.clear()
        self.round_end_time = 0

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

    def return_associated_validators(self):
        return self.miner_associated_validator_set

    def set_unordered_arrival_time_accepted_validator_transactions(self, unordered_arrival_time_accepted_validator_transactions):
        self.unordered_arrival_time_accepted_validator_transactions = unordered_arrival_time_accepted_validator_transactions


    def miner_broadcast_validator_transactions(self,miner_this_round):
        # for peer in self.peer_list:
        #     if peer.is_online():
        #         if peer.return_role() == "miner":
        #             if not peer.return_idx() in self.black_list:
        #                 print(f"miner {self.idx} is broadcasting received validator transactions to miner {peer.return_idx()}.")
        #                 final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner = copy.copy(self.unordered_arrival_time_accepted_validator_transactions)
        #                 # offline situation similar in validator_broadcast_worker_transactions()
        #                 for arrival_time, tx in self.unordered_arrival_time_accepted_validator_transactions.items():
        #                     if not (self.online_switcher() and peer.online_switcher()):
        #                         del final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner[arrival_time]
        #                 peer.accept_miner_broadcasted_validator_transactions(self, final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)
        #                 print(f"miner {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)} validator transactions to miner {peer.return_idx()}.")
        #             else:
        #                 print(f"Destination miner {peer.return_idx()} is in miner {self.idx}'s black_list. broadcasting skipped for this dest miner.")
        for miner in miner_this_round:
                if not miner.return_id() in self.black_list and miner.return_id()!=self.id:
                    print(f"miner {self.id} is broadcasting received validator transactions to miner {miner.return_id()}.")
                    final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner = copy.copy(self.unordered_arrival_time_accepted_validator_transactions)
                            # if offline, it's like the broadcasted transaction was not received, so skip a transaction
                    for arrival_time, tx in self.unordered_arrival_time_accepted_validator_transactions.items():
                        if not (self.online_switcher() and miner.online_switcher()):
                            del final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner[arrival_time]
                            # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received transactions(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the transactions itself received
                    miner.accept_miner_broadcasted_validator_transactions(self, final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)
                            
                    print(f"miner {self.id} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)} validator transactions to miner {miner.return_id()}.")
                else:
                    print(f"Destination miner {miner.return_id()} is in  miner {self.id}'s black_list. broadcasting skipped for this dest miner.")
        

    def accept_miner_broadcasted_validator_transactions(self, source_device, unordered_transaction_arrival_queue_from_source_miner):
        # discard malicious node
        if not source_device.return_id() in self.black_list:
            self.miner_accepted_broadcasted_validator_transactions.append({'source_device_link_speed': source_device.return_link_speed(),'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_miner)})
            print(f"{self.role} {self.id} has accepted validator transactions from {source_device.return_role()} {source_device.return_id()}")
        else:
            print(f"Source miner {source_device.return_role()} {source_device.return_id()} is in {self.role} {self.id}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_validator_transactions(self):
        return self.miner_accepted_broadcasted_validator_transactions
    
    def return_unordered_arrival_time_accepted_validator_transactions(self):
        return self.unordered_arrival_time_accepted_validator_transactions

    def set_candidate_transactions_for_final_mining_queue(self, final_transactions_arrival_queue):
        self.final_candidate_transactions_queue_to_mine = final_transactions_arrival_queue