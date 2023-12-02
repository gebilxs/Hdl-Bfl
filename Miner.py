from Devicebase import Device
import copy
import time
from hashlib import sha256
import random
from sys import getsizeof
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
        if args.even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        self.check_signature = args.check_signature
        
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

    def return_final_candidate_transactions_mining_queue(self):
        return self.final_candidate_transactions_queue_to_mine
    
    def return_miner_acception_wait_time(self):
        return self.miner_acception_wait_time
    
    def return_miner_accepted_transactions_size_limit(self):
        return self.miner_accepted_transactions_size_limit
    
    def verify_validator_transaction(self, transaction_to_verify):
        if self.computation_power == 0:
            print(f"miner {self.id} has computation power 0 and will not be able to verify this transaction in time")
            return False, None
        else:
            transaction_validator_idx = transaction_to_verify['validation_done_by']

            if transaction_validator_idx in self.black_list:
                print(f"{transaction_validator_idx} is in miner's blacklist. Trasaction won't get verified.")
                return False, None
            
            verification_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_verify)
                del transaction_before_signed["validator_signature"]
                modulus = transaction_to_verify['validator_rsa_pub_key']["modulus"]
                pub_key = transaction_to_verify['validator_rsa_pub_key']["pub_key"]
                signature = transaction_to_verify["validator_signature"]
                # begin verification
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.id}!")
                    verification_time = (time.time() - verification_time)/self.computation_power
                    return verification_time, True
                else:
                    print(f"Signature invalid. Transaction from validator {transaction_validator_idx} is NOT verified.")
                    return (time.time() - verification_time)/self.computation_power, False
            else:
                print(f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.id}!")
                verification_time = (time.time() - verification_time)/self.computation_power
                return verification_time, True
    def sign_candidate_transaction(self, candidate_transaction):
        signing_time = time.time()
        candidate_transaction['miner_rsa_pub_key'] = self.return_rsa_pub_key()
        if 'miner_signature' in candidate_transaction.keys():
            del candidate_transaction['miner_signature']
        candidate_transaction["miner_signature"] = self.sign_msg(sorted(candidate_transaction.items()))
        signing_time = (time.time() - signing_time)/self.computation_power
        return signing_time
    def set_block_generation_time_point(self, block_generation_time_point):
        self.block_generation_time_point = block_generation_time_point

    def mine_block(self, candidate_block, rewards, starting_nonce=0):

        candidate_block.set_mined_by(self.id)
        pow_mined_block = self.proof_of_work(candidate_block)
        # pow_mined_block.set_mined_by(self.idx)

        # TODO 修改rewards机制
        pow_mined_block.set_mining_rewards(rewards)
        return pow_mined_block
    
    def proof_of_work(self, candidate_block, starting_nonce=0):
        candidate_block.set_mined_by(self.id)
        ''' Brute Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        # candidate_block.set_pow_difficulty(self.pow_difficulty)
        while not current_hash.startswith('0' * self.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
        # also set its hash as well. block_hash is the same as pow proof
        candidate_block.set_pow_proof(current_hash)
        return candidate_block

    def sign_block(self, block_to_sign):
        block_to_sign.set_signature(self.sign_msg(block_to_sign.__dict__))

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def propagated_the_block(self, propagating_time_point, block_to_propagate,miners_this_round):
        for miner in miners_this_round:
            if miner.is_online():
                if miner.return_id() != self.id and not miner.return_id() in self.black_list:
                    print(f"{self.role} {self.id} is propagating its mined block to {miner.return_role()} {miner.return_id()}.")
                    if miner.online_switcher():
                        miner.accept_the_propagated_block(self, self.block_generation_time_point, block_to_propagate)
                    else:
                        print(f"Destination miner {miner.return_id()} is in {self.role} {self.id}'s black_list. Propagating skipped for this dest miner.")

        # for peer in self.peer_list:
        #     if peer.is_online():
        #         if peer.return_role() == "miner":
        #             if not peer.return_idx() in self.black_list:
        #                 print(f"{self.role} {self.idx} is propagating its mined block to {peer.return_role()} {peer.return_idx()}.")
        #                 if peer.online_switcher():
        #                     peer.accept_the_propagated_block(self, self.block_generation_time_point, block_to_propagate)
        #             else:
        #                 print(f"Destination miner {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Propagating skipped for this dest miner.")

    def accept_the_propagated_block(self, source_miner, source_miner_propagating_time_point, propagated_block):
        if not source_miner.return_id() in self.black_list:
            source_miner_link_speed = source_miner.return_link_speed()
            this_miner_link_speed = self.link_speed
            lower_link_speed = this_miner_link_speed if this_miner_link_speed < source_miner_link_speed else source_miner_link_speed
            transmission_delay = getsizeof(str(propagated_block.__dict__))/lower_link_speed
            self.unordered_propagated_block_processing_queue[source_miner_propagating_time_point + transmission_delay] = propagated_block
            print(f"{self.role} {self.id} has accepted accepted a propagated block from miner {source_miner.return_id()}")
        else:
            print(f"Source miner {source_miner.return_role()} {source_miner.return_id()} is in {self.role} {self.id}'s black list. Propagated block not accepted.")

    def return_block_generation_time_point(self):
        return self.block_generation_time_point
    
    def return_unordered_propagated_block_processing_queue(self):
        return self.unordered_propagated_block_processing_queue
    
    def return_mined_block(self):
        return self.mined_block
    

    

    def request_to_download(self, block_to_download, requesting_time_point):
        # 所有的区块都进行下载
        print(f"miner {self.id} is requesting its associated devices to download the block it just added to its chain")
        devices_in_association = self.miner_associated_validator_set.union(self.miner_associated_worker_set)
        for device in devices_in_association:
            # theoratically, one device is associated to a specific miner, so we don't have a miner_block_arrival_queue here
            if self.online_switcher() and device.online_switcher():
                miner_link_speed = self.return_link_speed()
                device_link_speed = device.return_link_speed()
                lower_link_speed = device_link_speed if device_link_speed < miner_link_speed else miner_link_speed
                transmission_delay = getsizeof(str(block_to_download.__dict__))/lower_link_speed
                verified_block, verification_time = device.verify_block(block_to_download, block_to_download.return_mined_by())
                if verified_block:
                    # forgot to check for maliciousness of the block miner
                    device.add_block(verified_block)
                device.add_to_round_end_time(requesting_time_point + transmission_delay + verification_time)
            else:
                print(f"Unfortunately, either miner {self.id} or {device.return_id()} goes offline while processing this request-to-download block.")