import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import random
from BlockChain import Blockchain 
from Crypto.PublicKey import RSA
from hashlib import sha256
import time
from collections import defaultdict
# ChainCode logic 
# every node got ability to connect with blockChain
class Device(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model_name)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.args = args
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batchsize
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.is_malicious = args.is_malicious

        # if IID is 1 -> NonIID 
        # if IID is 0 -> normal dataset
        self.opti = args.optimizer
        self.num_devices = args.num_devices
        self.round_end_time = 0

        self.devices_after_load_data = {}
        self.malicious_nodes_set = []
        self.num_malicious = args.num_malicious
        self.devices_dict = None
        self.aio = False
        self.pow_difficulty = args.pow_difficulty
        self.peer_list = set()
        self.online = True
        if self.num_malicious:
            self.malicious_nodes_set = random.sample(range(self.num_devices), self.num_malicious)
        self.network_stability = args.network_stability
        # print(self.is_malicious) 

        # check BatchNorm
        self.has_BatchNorm = False
        self.role = ""
        # black_list stores device index rather than the object
        self.black_list = set()
        self.blockchain = Blockchain()
        self.not_resync_chain = args.destroy_tx_in_block
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        # self.train_slow = kwargs['train_slow']
        # self.send_slow = kwargs['send_slow']
        # self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        # self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        if args.even_link_speed_strength:
            self.link_speed = args.base_data_transmission_speed

        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()

        self.the_added_block = None
        self.rewards = 0
        self.block_generation_time_point = None

        # used to identify slow or lazy workers
        self.active_worker_record_by_round = {}
        self.untrustworthy_workers_record_by_comm_round = {}
        self.untrustworthy_validators_record_by_comm_round = {}
        # for picking PoS legitimate blockd;bs
        # self.stake_tracker = {} # used some tricks in main.py for ease of programming
        # used to determine the slowest device round end time to compare PoW with PoS round end time. If simulate under computation_power = 0, this may end up equaling infinity
        self.round_end_time = 0
        self.check_signature = args.check_signature
        self.malicious_updates_discount = args.malicious_updates_discount
        self.knock_out_rounds = args.knock_out_rounds

        if args.even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        self.check_signature = args.check_signature
        self.lazy_worker_knock_out_rounds = args.lazy_worker_knock_out_rounds

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        '''logits set'''
        self.global_logits = [None for _ in range(args.num_classes)]
        self.used_global_logits = None
        self.lamda = args.lamda
    # TODO malicious_node_load_train_data ! add noise

    ''' getter '''
    def return_id(self):
        return self.id
    
    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_peers(self):
        return self.peer_list

    def return_role(self):
        return self.role

    def is_online(self):
        return self.online

    def return_is_malicious(self):
        return self.is_malicious

    def return_black_list(self):
        return self.black_list

    def return_blockchain_object(self):
        return self.blockchain

    def return_stake(self):
        return self.rewards

    def return_computation_power(self):
        return self.computation_power

    def return_the_added_block(self):
        return self.the_added_block

    def return_round_end_time(self):
        return self.round_end_time
    
    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy

    def set_auc_this_round(self,test_auc):
        self.test_auc = test_auc

    def set_loss_this_round(self,train_loss):
        self.train_loss_this_round =  train_loss

    def return_test_auc(self):
        return self.test_auc
    
    def return_accuracy_this_round(self):
        return self.accuracy_this_round

    def return_train_loss_this_round(self):
        return self.train_loss_this_round
    
    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e

    '''load message'''
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    '''test part'''
    '''for global model'''
    def test_metrics(self):
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
        
        return test_acc, test_num, auc

    # def train_metrics(self):
    #     trainloader = self.load_train_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
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
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]

    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')

    #     return losses, train_num
    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
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

                if self.global_logits != None:
                    logit_new = copy.deepcopy(output.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_logits[y_c]) != type([]):
                            logit_new[i, :] = self.global_logits[y_c].data
                    loss += self.loss_mse(logit_new, output) * self.lamda
                    
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def set_devices_dict_and_aio(self,devices_dict,aio):
        self.devices_dict = devices_dict
        self.aio = aio

    # TODO add func add_peer
    def add_peers(self,new_peers):
        if isinstance(new_peers, Device):
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)

    def register_in_the_network(self, check_online=False):
            if self.aio:
                self.add_peers(set(self.devices_dict.values()))
            else:
                potential_registrars = set(self.devices_dict.values())
                # it cannot register with itself
                potential_registrars.discard(self)		
                # pick a registrar
                registrar = random.sample(potential_registrars, 1)[0]
                if check_online:
                    if not registrar.is_online():
                        online_registrars = set()
                        for registrar in potential_registrars:
                            if registrar.is_online():
                                online_registrars.add(registrar)
                        if not online_registrars:
                            return False
                        registrar = random.sample(online_registrars, 1)[0]
                # registrant add registrar to its peer list
                self.add_peers(registrar)
                # this device sucks in registrar's peer list
                self.add_peers(registrar.return_peers())
                # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
                registrar.add_peers(self)
                return True
    def is_online(self):
        return self.online
    
    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):
            self.peer_list.discard(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)

    '''resync blockchain'''
    def pow_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a longer chain in the network...")
        longest_chain = None
        updated_from_peer = None
        curr_chain_len = self.return_blockchain_object().return_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer_chain):
                        print(f"A longer chain from {peer.return_id()} with chain length {peer_chain.return_chain_length()} has been found (> currently compared chain length {curr_chain_len}) and verified.")
                        # Longer valid chain found!
                        curr_chain_len = peer_chain.return_chain_length()
                        longest_chain = peer_chain
                        updated_from_peer = peer.return_id()
                    else:
                        print(f"A longer chain from {peer.return_id()} has been found BUT NOT verified. Skipped this chain for syncing.")
        if longest_chain:
            # compare chain difference
            longest_chain_structure = longest_chain.return_chain_structure()
            # need more efficient machenism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(longest_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            #return block_iter
            return True 
        print("Chain not resynced.")
        return False
    
    def pos_resync_chain(self):
        print(f"{self.role} {self.id} is looking for a chain with the highest accumulated miner's stake in the network...")
        highest_stake_chain = None
        updated_from_peer = None
        curr_chain_stake = self.accumulate_chain_stake(self.return_blockchain_object())
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                peer_chain_stake = self.accumulate_chain_stake(peer_chain)
                if peer_chain_stake > curr_chain_stake:
                    if self.check_chain_validity(peer_chain):
                        print(f"A chain from {peer.return_id()} with total stake {peer_chain_stake} has been found (> currently compared chain stake {curr_chain_stake}) and verified.")
                        # Higher stake valid chain found!
                        curr_chain_stake = peer_chain_stake
                        highest_stake_chain = peer_chain
                        updated_from_peer = peer.return_id()
                    else:
                        print(f"A chain from {peer.return_id()} with higher stake has been found BUT NOT verified. Skipped this chain for syncing.")
        if highest_stake_chain:
            # compare chain difference
            highest_stake_chain_structure = highest_stake_chain.return_chain_structure()
            # need more efficient machenism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(highest_stake_chain_structure)
            print(f"{self.id} chain resynced from peer {updated_from_peer}.")
            #return block_iter
            return True 
        print("Chain not resynced.")
        return False
    
    def accumulate_chain_stake(self, chain_to_accumulate):
        accumulated_stake = 0
        chain_to_accumulate = chain_to_accumulate.return_chain_structure()
        for block in chain_to_accumulate:
            accumulated_stake += self.devices_dict[block.return_mined_by()].return_stake()
        return accumulated_stake
    
    def update_model_after_chain_resync(self, log_files_folder_path, conn, conn_cursor):
        # reset global params to the initial weights of the net
        self.global_parameters = self.model
        # in future version, develop efficient updating algorithm based on chain difference
        for block in self.return_blockchain_object().return_chain_structure():
            self.process_block(block,log_files_folder_path, conn, conn_cursor, when_resync=True)
            print("process BlockChain")

    def online_switcher(self):
        old_status = self.online
        online_indicator = random.random()
        # print("in online_switcher")
        if online_indicator < self.network_stability:
            self.online = True
            # if back online, update peer and resync chain
            if old_status == False:
                print(f"{self.id} goes back online.")
                # update peer list
                self.update_peer_list()
                # resync chain
                if self.pow_resync_chain():
                    self.update_model_after_chain_resync()
        else:
            self.online = False
            print(f"{self.id} goes offline.")
        return self.online
    
    def update_peer_list(self):
        print(f"\n{self.id} - {self.role} is updating peer list...")
        old_peer_list = copy.copy(self.peer_list)
        online_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)
        # for online peers, suck in their peer list
        for online_peer in online_peers:
            self.add_peers(online_peer.return_peers())
        # remove itself from the peer_list if there is
        self.remove_peers(self)
        # remove malicious peers
        removed_peers = []
        potential_malicious_peer_set = set()
        for peer in self.peer_list:
            if peer.return_id() in self.black_list:
                potential_malicious_peer_set.add(peer)
        self.remove_peers(potential_malicious_peer_set)
        removed_peers.extend(potential_malicious_peer_set)
        # print updated peer result
        if old_peer_list == self.peer_list:
            print("Peer list NOT changed.")
        else:
            print("Peer list has been changed.")
            added_peers = self.peer_list.difference(old_peer_list)
            if potential_malicious_peer_set:
                print("These malicious peers are removed")
                for peer in removed_peers:
                    print(f"d_{peer.return_id().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            if added_peers:
                print("These peers are added")
                for peer in added_peers:
                    print(f"d_{peer.return_id().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print("Final peer list:")
            for peer in self.peer_list:
                print(f"d_{peer.return_id().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
            print()
        # WILL ALWAYS RETURN TRUE AS OFFLINE PEERS WON'T BE REMOVED ANY MORE, UNLESS ALL PEERS ARE MALICIOUS...but then it should not register with any other peer. Original purpose - if peer_list ends up empty, randomly register with another device
        return False if not self.peer_list else True

    def check_pow_proof(self, block_to_check):
        # remove its block hash(compute_hash() by default) to verify pow_proof as block hash was set after pow
        pow_proof = block_to_check.return_pow_proof()
        # print("pow_proof", pow_proof)
        # print("compute_hash", block_to_check.compute_hash())
        return pow_proof.startswith('0' * self.pow_difficulty) and pow_proof == block_to_check.compute_hash()

    def check_chain_validity(self, chain_to_check):
        chain_len = chain_to_check.return_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            chain_to_check = chain_to_check.return_chain_structure()
            for block in chain_to_check[1:]:
                if self.check_pow_proof(block) and block.return_previous_block_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True):
                    pass
                else:
                    return False
        return True
    
    def resync_chain(self, mining_consensus):
        if self.not_resync_chain:
            return # temporary workaround to save GPU memory
        if mining_consensus == 'PoW':
            self.pow_resync_chain()
        else:
            self.pos_resync_chain()

    def receive_rewards(self, rewards):
        self.rewards += rewards

    def return_link_speed(self):
        return self.link_speed

    def sign_msg(self, msg):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return signature
    
    def add_to_round_end_time(self, time_to_add):
        self.round_end_time += time_to_add

    def return_the_added_block(self):
        return self.the_added_block
    
    def verify_block(self, block_to_verify, sending_miner):
        if not self.online_switcher():
            print(f"{self.id} goes offline when verifying a block")
            return False, False
        verification_time = time.time()
        mined_by = block_to_verify.return_mined_by()
        if sending_miner in self.black_list:
            print(f"The miner propagating/sending this block {sending_miner} is in {self.id}'s black list. Block will not be verified.")
            return False, False
        if mined_by in self.black_list:
            print(f"The miner {mined_by} mined this block is in {self.id}'s black list. Block will not be verified.")
            return False, False
        # check if the proof is valid(verify _block_hash).
        if not self.check_pow_proof(block_to_verify):
            print(f"PoW proof of the block from miner {self.id} is not verified.")
            return False, False
        # # check if miner's signature is valid
        if self.check_signature:
            signature_dict = block_to_verify.return_miner_rsa_pub_key()
            modulus = signature_dict["modulus"]
            pub_key = signature_dict["pub_key"]
            signature = block_to_verify.return_signature()
            # verify signature
            block_to_verify_before_sign = copy.deepcopy(block_to_verify)
            block_to_verify_before_sign.remove_signature_for_verification()
            hash = int.from_bytes(sha256(str(block_to_verify_before_sign.__dict__).encode('utf-8')).digest(), byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash != hashFromSignature:
                print(f"Signature of the block sent by miner {sending_miner} mined by miner {mined_by} is not verified by {self.role} {self.id}.")
                return False, False
            # check previous hash based on own chain
            last_block = self.return_blockchain_object().return_last_block()
            if last_block is not None:
                # check if the previous_hash referred in the block and the hash of latest block in the chain match.
                last_block_hash = last_block.compute_hash(hash_entire_block=True)
                if block_to_verify.return_previous_block_hash() != last_block_hash:
                    print(f"Block sent by miner {sending_miner} mined by miner {mined_by} has the previous hash recorded as {block_to_verify.return_previous_block_hash()}, but the last block's hash in chain is {last_block_hash}. This is possibly due to a forking event from last round. Block not verified and won't be added. Device needs to resync chain next round.")
                    return False, False
        # All verifications done.
        print(f"Block accepted from miner {sending_miner} mined by {mined_by} has been verified by {self.id}!")
        verification_time = (time.time() - verification_time)/self.computation_power
        return block_to_verify, verification_time
    
    def add_block(self, block_to_add):
        self.return_blockchain_object().append_block(block_to_add)
        print(f"d_{self.id} - {self.role[0]} has appened a block to its chain. Chain length now - {self.return_blockchain_object().return_chain_length()}")
        # TODO delete has_added_block
        # self.has_added_block = True
        self.the_added_block = block_to_add
        return True

    def return_block_generation_time_point(self):
        return self.block_generation_time_point
    
    def process_block(self,block_to_process, log_files_folder_path, conn, conn_cursor, when_resync=False):
        # collect usable updated params, malicious nodes identification, get rewards and do local udpates
        processing_time = time.time()
        if not self.online_switcher():
            print(f"{self.role} {self.id} goes offline when processing the added block. Model not updated and rewards information not upgraded. Outdated information may be obtained by this node if it never resyncs to a different chain.") # may need to set up a flag indicating if a block has been processed
        if block_to_process:
            mined_by = block_to_process.return_mined_by()
            if mined_by in self.black_list:
                # in this system black list is also consistent across devices as it is calculated based on the information on chain, but individual device can decide its own validation/verification mechanisms and has its own 
                print(f"The added block is mined by miner {block_to_process.return_mined_by()}, which is in this device's black list. Block will not be processed.")
            else:
                # process validator sig valid transactions
                # used to count positive and negative transactions worker by worker, select the transaction to do global update and identify potential malicious worker
                self_rewards_accumulator = 0
                valid_transactions_records_by_worker = {}
                valid_validator_sig_worker_transacitons_in_block = block_to_process.return_transactions()['valid_validator_sig_transacitons']
                comm_round = block_to_process.return_block_idx()
                self.active_worker_record_by_round[comm_round] = set()
                for valid_validator_sig_worker_transaciton in valid_validator_sig_worker_transacitons_in_block:
                    # verify miner's signature(miner does not get reward for receiving and aggregating)
                    if self.verify_miner_transaction_by_signature(valid_validator_sig_worker_transaciton, mined_by):
                        worker_device_idx = valid_validator_sig_worker_transaciton['worker_device_idx']
                        self.active_worker_record_by_round[comm_round].add(worker_device_idx)
                        if not worker_device_idx in valid_transactions_records_by_worker.keys():
                            valid_transactions_records_by_worker[worker_device_idx] = {}
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = None
                            # -------------------------------------------------------------
                            valid_transactions_records_by_worker[worker_device_idx]['train_samples'] = int
                            valid_transactions_records_by_worker[worker_device_idx]['logits'] = {}
                        # epoch of this worker's local update
                        local_epoch_seq = valid_validator_sig_worker_transaciton['local_total_accumulated_epochs_this_round']
                        positive_direction_validators = valid_validator_sig_worker_transaciton['positive_direction_validators']
                        negative_direction_validators = valid_validator_sig_worker_transaciton['negative_direction_validators']
                        worker_train_sample = valid_validator_sig_worker_transaciton['train_samples']
                        worker_logits = valid_validator_sig_worker_transaciton['logits']
                        if len(positive_direction_validators) >= len(negative_direction_validators):
                            # worker transaction can be used
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'].add(local_epoch_seq)
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(local_epoch_seq)
                            # -----------------------------------------------------------------
                            valid_transactions_records_by_worker[worker_device_idx]['train_samples']=worker_train_sample
                            valid_transactions_records_by_worker[worker_device_idx]['logits'].update(worker_logits)
                            # see if this is the latest epoch from this worker
                            if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = valid_validator_sig_worker_transaciton['local_updates_params']
                            # give rewards to this worker
                            if self.id == worker_device_idx:
                                self_rewards_accumulator += valid_validator_sig_worker_transaciton['local_updates_rewards']
                        else:
                            if self.malicious_updates_discount:
                                # worker transaction voted negative and has to be applied for a discount
                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(local_epoch_seq)
                                valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(local_epoch_seq)
                                # see if this is the latest epoch from this worker
                                if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                    # apply discount
                                    discounted_valid_validator_sig_worker_transaciton_local_updates_params = copy.deepcopy(valid_validator_sig_worker_transaciton['local_updates_params'])
                                    for var in discounted_valid_validator_sig_worker_transaciton_local_updates_params:
                                        discounted_valid_validator_sig_worker_transaciton_local_updates_params[var] *= self.malicious_updates_discount
                                    valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = discounted_valid_validator_sig_worker_transaciton_local_updates_params
                                # worker receive discounted rewards for negative update
                                if self.id == worker_device_idx:
                                    self_rewards_accumulator += valid_validator_sig_worker_transaciton['local_updates_rewards'] * self.malicious_updates_discount
                            else:
                                # discount specified as 0, worker transaction voted negative and cannot be used
                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(local_epoch_seq)
                                # worker does not receive rewards for negative update
                        # give rewards to validators and the miner in this transaction
                        for validator_record in positive_direction_validators + negative_direction_validators:
                            if self.id == validator_record['validator']:
                                self_rewards_accumulator += validator_record['validation_rewards']
                            if self.id == validator_record['miner_device_idx']:
                                self_rewards_accumulator += validator_record['miner_rewards_for_this_tx']
                    else:
                        print(f"one validator transaction miner sig found invalid in this block. {self.id} will drop this block and roll back rewards information")
                        return
                # identify potentially malicious worker
                self.untrustworthy_workers_record_by_comm_round[comm_round] = set()
                for worker_idx, local_updates_direction_records in valid_transactions_records_by_worker.items():
                    if len(local_updates_direction_records['negative_epochs']) >  len(local_updates_direction_records['positive_epochs']):
                        self.untrustworthy_workers_record_by_comm_round[comm_round].add(worker_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_workers_record_by_comm_round.keys():
                                if worker_idx in self.untrustworthy_workers_record_by_comm_round[comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(worker_idx)
                            # is it right?
                            if when_resync:
                                msg_end = " when resyncing!\n"
                            else:
                                msg_end = "!\n"
                            if self.devices_dict[worker_idx].return_is_malicious():
                                msg = f"{self.id} has successfully identified a malicious worker device {worker_idx} in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)", (worker_idx, 1, self.idx, "", comm_round, when_resync))
                                conn.commit()
                            else:
                                msg = f"WARNING: {self.idx} has mistakenly regard {worker_idx} as a malicious worker device in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)", (worker_idx, 0, "", self.idx, comm_round, when_resync))
                                conn.commit()
                            print(msg)
                           
                            # cont = print("Press ENTER to continue")
                
                # identify potentially compromised validator
                self.untrustworthy_validators_record_by_comm_round[comm_round] = set()
                invalid_validator_sig_worker_transacitons_in_block = block_to_process.return_transactions()['invalid_validator_sig_transacitons']
                for invalid_validator_sig_worker_transaciton in invalid_validator_sig_worker_transacitons_in_block:
                    if self.verify_miner_transaction_by_signature(invalid_validator_sig_worker_transaciton, mined_by):
                        validator_device_idx = invalid_validator_sig_worker_transaciton['validator']
                        self.untrustworthy_validators_record_by_comm_round[comm_round].add(validator_device_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_validators_record_by_comm_round.keys():
                                if validator_device_idx in self.untrustworthy_validators_record_by_comm_round[comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(validator_device_idx)
                            print(f"{validator_device_idx} has been regarded as a compromised validator by {self.id} in {comm_round}.")
                            # actually, we did not let validator do malicious thing if is_malicious=1 is set to this device. In the submission of 2020/10, we only focus on catching malicious worker
                            # is it right?
                            # if when_resync:
                            #	 msg_end = " when resyncing!\n"
                            # else:
                            #	 msg_end = "!\n"
                            # if self.devices_dict[validator_device_idx].return_is_malicious():
                            #	 msg = f"{self.idx} has successfully identified a compromised validator device {validator_device_idx} in comm_round {comm_round}{msg_end}"
                            #	 with open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'a') as file:
                            #		 file.write(msg)
                            # else:
                            #	 msg = f"WARNING: {self.idx} has mistakenly regard {validator_device_idx} as a compromised validator device in comm_round {comm_round}{msg_end}"
                            #	 with open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'a') as file:
                            #		 file.write(msg)
                            # print(msg)
                            # cont = print("Press ENTER to continue")
                    else:
                        print(f"one validator transaction miner sig found invalid in this block. {self.id} will drop this block and roll back rewards information")
                        return
                    # give rewards to the miner in this transaction
                    if self.id == invalid_validator_sig_worker_transaciton['miner_device_idx']:
                        self_rewards_accumulator += invalid_validator_sig_worker_transaciton['miner_rewards_for_this_tx']
                # miner gets mining rewards
                if self.id == mined_by:
                    self_rewards_accumulator += block_to_process.return_mining_rewards()
                # set received rewards this round based on info from this block
                self.receive_rewards(self_rewards_accumulator)
                print(f"{self.role} {self.id} has received total {self_rewards_accumulator} rewards for this comm round.")
                # collect usable worker updates and do global updates /maybe it's model
                finally_used_local_params = []
                # record True Positive, False Positive, True Negative and False Negative for identified workers
                TP, FP, TN, FN = 0, 0, 0, 0
                for worker_device_idx, local_params_record in valid_transactions_records_by_worker.items():
                    is_worker_malicious = self.devices_dict[worker_device_idx].return_is_malicious()
                    if local_params_record['finally_used_params']:
                        # identified as benigh worker
                        finally_used_local_params.append((worker_device_idx, local_params_record['finally_used_params'],local_params_record['train_samples'],local_params_record['logits'])) # could be None
                        if not is_worker_malicious:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        # identified as malicious worker
                        if is_worker_malicious:
                            TN += 1
                        else:
                            FN += 1


                if self.online_switcher():
                        # self.global_update(finally_used_local_params)
                        self.global_logits=self.logits_update(finally_used_local_params)
                        self.set_logits(self.global_logits)
                else:
                    print(f"Unfortunately, {self.role} {self.id} goes offline when it's doing global_updates.")
        
        malicious_worker_validation_log_path = f"{log_files_folder_path}/comm_{comm_round}/malicious_worker_validation_log.txt"
        if not os.path.exists(malicious_worker_validation_log_path):
            with open(malicious_worker_validation_log_path, 'w') as file:
                accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN) else 0
                precision = TP / (TP + FP) if TP else 0
                recall = TP / (TP + FN) if TP else 0
                f1 = precision * recall / (precision + recall) if precision * recall else 0
                file.write(f"In comm_{comm_round} of validating workers, TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}. \
                        \nAccuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1 Score = {f1}")
        processing_time = (time.time() - processing_time)/self.computation_power
        return processing_time
    
    def verify_miner_transaction_by_signature(self, transaction_to_verify, miner_device_idx):
        if miner_device_idx in self.black_list:
            print(f"{miner_device_idx} is in miner's blacklist. Trasaction won't get verified.")
            return False
        if self.check_signature:
            transaction_before_signed = copy.deepcopy(transaction_to_verify)
            del transaction_before_signed["miner_signature"]
            modulus = transaction_to_verify['miner_rsa_pub_key']["modulus"]
            pub_key = transaction_to_verify['miner_rsa_pub_key']["pub_key"]
            signature = transaction_to_verify["miner_signature"]
            # verify
            hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash == hashFromSignature:
                print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
                return True
            else:
                print(f"Signature invalid. Transaction recorded by {miner_device_idx} is NOT verified.")
                return False
        else:
            print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
            return True

    def other_tasks_at_the_end_of_comm_round(self, this_comm_round, log_files_folder_path):
        self.kick_out_slow_or_lazy_workers(this_comm_round, log_files_folder_path)

    def kick_out_slow_or_lazy_workers(self, this_comm_round, log_files_folder_path):
        for device in self.peer_list:
            if device.return_role() == 'worker':
                if this_comm_round in self.active_worker_record_by_round.keys():
                    if not device.return_id() in self.active_worker_record_by_round[this_comm_round]:
                        not_active_accumulator = 1
                        # check if not active for the past (lazy_worker_knock_out_rounds - 1) rounds
                        for comm_round_to_check in range(this_comm_round - self.lazy_worker_knock_out_rounds + 1, this_comm_round):
                            if comm_round_to_check in self.active_worker_record_by_round.keys():
                                if not device.return_id() in self.active_worker_record_by_round[comm_round_to_check]:
                                    not_active_accumulator += 1
                        if not_active_accumulator == self.lazy_worker_knock_out_rounds:
                            # kick out
                            # TODO 没有同步，之后black_list需要同步给每个节点 
                            self.black_list.add(device.return_id())
                            msg = f"worker {device.return_id()} has been regarded as a lazy worker by {self.id} in comm_round {this_comm_round}.\n"
                            with open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'a') as file:
                                file.write(msg)
                else:
                    # this may happen when a device is put into black list by every worker in a certain comm round
                    pass

    # TODO 联邦学习中聚合model实例的参数
    def global_update(self, local_update_model_to_be_used):
        # filter local_params
        local_params_by_benign_workers = []
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_sample = 0
        if local_update_model_to_be_used:
        # receive model part
            for (worker_device_idx,worker_model,train_sample) in local_update_model_to_be_used:
                if not worker_device_idx in self.black_list:
                    tot_sample += train_sample
                    self.uploaded_ids.append(worker_device_idx)
                    self.uploaded_weights.append(train_sample)
                    self.uploaded_models.append(worker_model)

            for i,w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_sample
            
            # aggregate_paramerters
            assert (len(self.uploaded_models)>0)

            self.model = copy.deepcopy(self.uploaded_models[0])
            for param in self.model.parameters():
                param.data.zero_()

            for w,worker_model in zip(self.uploaded_weights,self.uploaded_models):
                self.add_parameter(w,worker_model)
            print(f"global updates done by {self.id}")
        else:
            print(f"There are no available local params for {self.id} to perform global updates in this comm round.")

    def logits_update(self,used_message):
        agg_logits_label = defaultdict(list)
        local_logits_list =[]
        worker_device_list =[]
        for(worker_device_idx,worker_model,train_sample,upload_logits) in used_message:
            local_logits_list.append(upload_logits)
            worker_device_list.append(worker_device_idx)
        for local_logits in local_logits_list:
            for label in local_logits.keys():
                agg_logits_label[label].append(local_logits[label])

        for [label, logit_list] in agg_logits_label.items():
            if len(logit_list) > 1:
                logit = 0 * logit_list[0].data
                for i in logit_list:
                    logit += i.data
                agg_logits_label[label] = logit / len(logit_list)
            else:
                agg_logits_label[label] = logit_list[0].data

        return dict(agg_logits_label)
    
    def set_logits(self,global_logits):
        self.used_global_logits = copy.deepcopy(global_logits)

    def use_logit_train(self):
        trainloader = self.load_train_data()
        self.model.train()
        local_epoch =1
        for step in range(local_epoch):
            for i,(x,y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output,y)

                if self.used_global_logits !=None:
                    logit_new = copy.deepcopy(output.detach())
                    for i,yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.used_global_logits[y_c])!=type([]):
                            logit_new[i,:] = self.global_logits[y_c].data
                    loss += self.loss_mse(logit_new,output) * self.lamda
        
        # here put new logits to v.but miner and validator don't need update 
        # logtis = defaultdict(list)
        # for i,yy in enumerate(y):
        #     y_c = yy.item()
        #     logits[y_c].append(output[i,:].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def add_parameter(self,w,worker_model):
        for server_param, client_param in zip(self.model.parameters(), worker_model.parameters()):
            server_param.data += client_param.data.clone() * w


        # for (worker_device_idx, local_params) in local_update_params_potentially_to_be_used.item():
        #     if not worker_device_idx in self.black_list:
        #         local_params_by_benign_workers.append(local_params)
        #     else:
        #         print(f"global update skipped for a worker {worker_device_idx} in {self.id}'s black list")


        # if local_params_by_benign_workers:
        #     # avg the gradients
        #     sum_parameters = None
        #     for local_updates_params in local_params_by_benign_workers:
        #         if sum_parameters is None:
        #             sum_parameters = copy.deepcopy(local_updates_params)
        #         else:
        #             for var in sum_parameters:
        #                 sum_parameters[var] += local_updates_params[var]
        #     # number of finally filtered workers' updates
        #     num_participants = len(local_params_by_benign_workers)
        #     for var in self.global_parameters:
        #         self.global_parameters[var] = (sum_parameters[var] / num_participants)
        #     print(f"global updates done by {self.id}")
        # else:
        #     print(f"There are no available local params for {self.id} to perform global updates in this comm round.")
        


    def evaluate(self):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        test_acc = stats[0]/stats[1]
        test_auc = stats[2]/stats[1]
        train_loss = stats_train[0]/stats_train[1]

        self.set_accuracy_this_round(test_acc)
        self.set_loss_this_round(train_loss)
        self.set_auc_this_round(test_auc)

