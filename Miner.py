from Devicebase import Device
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