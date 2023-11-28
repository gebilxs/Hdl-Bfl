import os
import torch
import torch.nn.functional as F
import pickle
from datetime import datetime
import sys
from Model import model_judge
from sys import getsizeof
from Block import Block
from BlockChain import Blockchain
from Device import DevicesInNetwork
import shutil
import sqlite3
import time
import random
from Worker import Worker
from Miner import Miner
from Validator import Validator
import copy
# Add project runtime
def run(args):
    NETWORK_SNAPSHOTS_BASE_FOLDER = "snapshots"
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_files_folder_path = f"logs/{date_time}"
    # create log / if not exist
    if not os.path.exists("logs"):
        os.makedirs('logs')

    # select CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    dev = args.device
    # pre-define system variables
    latest_round_num = 0

    # 网络快照
    ''' If network_snapshot is specified, continue from left '''
    if args.resume_path:
        if not args.save_network_snapshots:
            print("NOTE: save_network_snapshots is set to 0. New network_snapshots won't be saved by conituing.")
        network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{args.resume_path}"
        latest_network_snapshot_file_name = sorted([f for f in os.listdir(network_snapshot_save_path) if not f.startswith('.')], key = lambda fn: int(fn.split('_')[-1]) , reverse=True)[0]
        print(f"Loading network snapshot from {args.resume_path}/{latest_network_snapshot_file_name}")
        print("BE CAREFUL - loaded dev env must be the same as the current dev env, namely, cpu, gpu or gpu parallel")
        latest_round_num = int(latest_network_snapshot_file_name.split('_')[-1])
        devices_in_network = pickle.load(open(f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", "rb"))
        devices_list = list(devices_in_network.devices_set.values())

        log_files_folder_path = f"logs/{args.resume_path}"

        # for colab
		# log_files_folder_path = f"/content/drive/MyDrive/BFA/logs/{args['resume_path']}"
		# original arguments file
        args_used_file = f"{log_files_folder_path}/args_used.txt"
        file = open(args_used_file,"r") 
        log_whole_text = file.read()
        lines_list = log_whole_text.split("\n")
        for line in lines_list:
			# abide by the original specified rewards
            if line.startswith('--unit_reward'):
                rewards = int(line.split(" ")[-1])
			# get number of roles
            if line.startswith('--hard_assign'):
                roles_requirement = line.split(" ")[-1].split(',')
			# get mining consensus
            if line.startswith('--pow_difficulty'):
                mining_consensus = 'PoW' if int(line.split(" ")[-1]) else 'PoS'
		# determine roles to assign
        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1
    else:
        ''' SETTING UP FROM SCRATCH'''
        # real logic

        # 0. create log_files_folder_path if not resume
        os.mkdir(log_files_folder_path)

       # 1. save arguments used
        with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
            f.write("Command line arguments used -\n")
            f.write(' '.join(sys.argv[1:]))
            f.write("\n\nAll arguments used -\n")
               # 遍历args对象并写入每个参数及其值
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")

        # 2. create network_snapshot folder default:0
        if args.save_network_snapshots:
            network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{date_time}"
            os.mkdir(network_snapshot_save_path)

        # 3. assign system variables
        # for demonstration purposes, this reward is for every rewarded action
        rewards = args.unit_reward

        # 4. get number of roles needed in the network
        roles_requirement = args.hard_assign.split(',')
        # print(f"role:{rewards}")
        # determine roles to assign
        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1

        # 5. check arguments eligibility
        num_devices = args.num_devices
        num_malicious = args.num_malicious

        if num_devices < workers_needed + miners_needed + validators_needed:
            sys.exit("ERROR: Roles assigned to the devices exceed the maximum number of allowed devices in the network.")
        
        if num_devices < 3:
            sys.exit("ERROR: There are not enough devices in the network.\n The system needs at least one miner, one worker and/or one validator to start the operation.\nSystem aborted.")

        if num_malicious:
            if num_malicious > num_devices:
                sys.exit("ERROR: The number of malicious nodes cannot exceed the total number of devices set in this network")
            else:
                print(f"Malicious nodes vs total devices set to {num_malicious}/{num_devices} = {(num_malicious/num_devices)*100:.2f}%")

        # 6. create neural net based on the input model name
        # model in model.py
        model_judge(args)
        # 7. assign GPU(s) if available to the net, otherwise CPU
		# os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

        if torch.cuda.device_count() > 1:
            args.model_name = torch.nn.DataParallel(args.model_name)
        print(f"{torch.cuda.device_count()} GPUs are available to use!")

        # 8.set base loss_function (different loss can be set in Class'_init_')
        # loss_func set in class will be more flexible
        # loss_func = F.cross_entropy
        

        # 9.create devices in the network pass 
        devices_in_network = DevicesInNetwork(args)
        devices_list = list(devices_in_network.devices_after_load_data.values())
        
        # create ChainCode 

        for device in devices_list:
            device.set_parameters(args.model_name)
            # helper function for registration simulation - set devices_list and aio
            device.set_devices_dict_and_aio(devices_in_network.devices_after_load_data,args.all_in_one)
            # simulate peer registration, with respect to device idx order
            device.register_in_the_network()
            
        # print(devices_list)
        # if need data then load_data
        for device in devices_list:
            device.remove_peers(device)

		# 11. build logging files/database path
		# create log files
        open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'w').close()
        open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'w').close()
        open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'w').close()
        open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'w').close()
		# open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'w').close()
		# open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'w').close()
        open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'w').close()

		# 12. setup the mining consensus
        mining_consensus = 'PoW' if args.pow_difficulty else 'PoS'
        print("Finish")

	# create malicious worker identification database
    conn = sqlite3.connect(f'{log_files_folder_path}/malicious_wokrer_identifying_log.db')
    conn_cursor = conn.cursor()
    conn_cursor.execute("""CREATE TABLE if not exists  malicious_workers_log (
	device_seq text,
	if_malicious integer,
	correctly_identified_by text,
	incorrectly_identified_by text,
	in_round integer,
	when_resyncing text
	)""")
################Start####################
    for comm_round in range(latest_round_num +1 ,args.max_num_comm + 1):
 # create round specific log folder
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if os.path.exists(log_files_folder_path_comm_round):
            print(f"Deleting {log_files_folder_path_comm_round} and create a new one.")
            shutil.rmtree(log_files_folder_path_comm_round)
        os.mkdir(log_files_folder_path_comm_round)
		# free cuda memory
        if dev == torch.device("cuda"):
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
        print(f"\nCommunication round {comm_round}")
        comm_round_start_time = time.time()
		# (RE)ASSIGN ROLES       
        # 10 show the final list like who is w v m
        # devices_list = list(devices_in_network.devices_set.values())
        # print(devices_list)
        workers_to_assign = workers_needed
        miners_to_assign = miners_needed
        validators_to_assign = validators_needed
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []
        # set random
        random.shuffle(devices_list)
        for device in (devices_list):
            if workers_to_assign:
                # put device in workers_this_round
                device.role = "worker"
                worker = Worker(device.args,device.id,device.train_samples,device.test_samples)
                worker.role = "worker"
                worker.peer_list = device.peer_list
                workers_this_round.append(worker)
                workers_to_assign-=1
            elif miners_to_assign:
                device.role = "miner"
                miner = Miner(device.args,device.id,device.train_samples,device.test_samples)
                miner.role = "miner"
                miner.peer_list = device.peer_list
                miners_this_round.append(miner)
                miners_to_assign-=1
            elif validators_to_assign:
                device.role = "validator"
                validator = Validator(device.args,device.id,device.train_samples,device.test_samples)
                validator.role = "validator"
                validator.peer_list = device.peer_list
                validators_this_round.append(validator)
                validators_to_assign-=1
# reset
            # print("begin_online_switcher")
            device.online_switcher()

        ''' DEBUGGING CODE '''
        if args.verbose:
        # show devices initial chain length and if online
            for device in devices_list:
                if device.is_online():
                    print(f'{device.return_id()} {device.return_role()} online - ', end='')
                else:
                    print(f'{device.return_id()} {device.return_role()} offline - ', end='')
				# debug chain length
                print(f"chain length {device.return_blockchain_object().return_chain_length()}")
			# show device roles
            print(f"\nThere are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")
            print("\nworkers this round are")
            for worker in workers_this_round:
                print(f"d_{worker.return_id()} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}")
            print("\nminers this round are")
            for miner in miners_this_round:
                print(f"d_{miner.return_id()} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}")
            print("\nvalidators this round are")
            for validator in validators_this_round:
                print(f"d_{validator.return_id()} online - {validator.is_online()} with chain len {validator.return_blockchain_object().return_chain_length()}")
            print()

			# show peers with round number
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
            for device_seq, device in devices_in_network.devices_after_load_data.items():
                peers = device.return_peers()
                print(f"d_{device_seq} - {device.return_role()[0]} has peer list ", end='')
                for peer in peers:
                    print(f"d_{peer.return_id()} - {peer.return_role()[0]}", end=', ')
                print()
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")

        ''' DEBUGGING CODE ENDS '''
        global_model = copy.deepcopy(args.model_name)
		# re-init round vars - in real distributed system, they could still fall behind in comm round, but here we assume they will all go into the next round together, thought device may go offline somewhere in the previous round and their variables were not therefore reset
        for miner in miners_this_round:
            miner.miner_reset_vars_for_new_round()
            miner.set_parameters(global_model)
        for worker in workers_this_round:
            worker.worker_reset_vars_for_new_round()
            worker.set_parameters(global_model)
        for validator in validators_this_round:
            validator.validator_reset_vars_for_new_round()
            validator.set_parameters(global_model)

		# DOESN'T MATTER ANY MORE AFTER TRACKING TIME, but let's keep it - orginal purpose: shuffle the list(for worker, this will affect the order of dataset portions to be trained)
        random.shuffle(workers_this_round) 
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)

        ''' workers, validators and miners take turns to perform jobs '''

        print(''' Step 1 - workers assign associated miner and validator (and do local updates, but it is implemented in code block of step 2) \n''')
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
			# resync chain(block could be dropped due to fork from last round)
            if worker.resync_chain(mining_consensus):
                worker.update_model_after_chain_resync(log_files_folder_path_comm_round, conn, conn_cursor)
			# worker (should) perform local update and associate
            print(f"{worker.return_id()} - worker {worker_iter+1}/{len(workers_this_round)} will associate with a validator and a miner, if online...")
			# worker associates with a miner to accept finally mined block
            # online_switcher() means updata data include peer_list
            if worker.online_switcher():
                # print(worker.return_peers()) why the set is empty
                associated_miner = worker.associate_with_miner("miner")
                judge = 0
                for miner in miners_this_round:
                    if miner.id == associated_miner.id:
                        judge = 1
                        miner.add_device_to_association(worker)
                        break
                if judge == 0:
                    print(f"Cannot find a qualified miner in {worker.return_id()} peer list.")
                # if associated_miner:
                #     associated_miner.add_device_to_association(worker)
                # else:
                #     print(f"Cannot find a qualified miner in {worker.return_id()} peer list.")
			# worker associates with a validator to send worker transactions
            if worker.online_switcher():
                associated_validator = worker.associate_with_validator("validator")
                judge = 0
                for validator in validators_this_round:
                    if validator.id == associated_validator.id:
                        judge = 1
                        validator.add_device_to_association(worker)
                        break
                if judge == 0:
                    print(f"Cannot find a qualified validator in {worker.return_id()} peer list.")
                # if associated_validator:
                #     associated_validator.add_device_to_association(worker)
                # else:
                #     print(f"Cannot find a qualified validator in {worker.return_id()} peer list.")

        print(''' Step 2 - validators accept local updates and broadcast to other validators in their respective peer lists (workers local_updates() are called in this step.\n''')
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
			# resync chain
            if validator.resync_chain(mining_consensus):
                validator.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
			# associate with a miner to send post validation transactions
            if validator.online_switcher():
                associated_miner = validator.associate_with_miner("miner")
                judge = 0
                for miner in miners_this_round:
                    if miner.id == associated_miner.id:
                        judge = 1
                        miner.add_device_to_association(worker)
                        break
                if judge == 0:
                    print(f"Cannot find a qualified miner in {worker.return_id()} peer list.")
                # if associated_miner:
                #     associated_miner.add_device_to_association(validator)
                # else:
                #     print(f"Cannot find a qualified miner in validator {validator.return_id()} peer list.")
			# validator accepts local updates from its workers association

            associated_workers = list(validator.return_associated_workers())
            if not associated_workers:
                print(f"No workers are associated with validator {validator.return_id()} {validator_iter+1}/{len(validators_this_round)} for this communication round.")
                continue
            validator_link_speed = validator.return_link_speed()
            print(f"{validator.return_id()} - validator {validator_iter+1}/{len(validators_this_round)} is accepting workers' updates with link speed {validator_link_speed} bytes/s, if online...")
		    # records_dict used to record transmission delay for each epoch to determine the next epoch updates arrival time
            records_dict = dict.fromkeys(associated_workers, None)
            for worker, _ in records_dict.items():
                records_dict[worker] = {}
			# used for arrival time easy sorting for later validator broadcasting (and miners' acception order)
            transaction_arrival_queue = {}
			# workers local_updates() called here as their updates transmission may be restrained by miners' acception time and/or size
            # TODO if there is time limit 
            if args.miner_acception_wait_time:
                print(f"miner waiting time is specified as {args.miner_acception_wait_time} seconds. let each worker do local_updates till time limit")
                for worker_iter in range(len(associated_workers)):
                    worker = associated_workers[worker_iter]
                    if not worker.return_id() in validator.return_black_list():
						# TODO here, also add print() for below miner's validators
                        print(f'worker {worker_iter+1}/{len(associated_workers)} of validator {validator.return_id()} is doing local updates')	 
                        total_time_tracker = 0
                        update_iter = 1
                        worker_link_speed = worker.return_link_speed()
                        lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
                        while total_time_tracker < validator.return_miner_acception_wait_time():
							# simulate the situation that worker may go offline during model updates transmission to the validator, based on per transaction
                            if worker.online_switcher():
                                local_update_spent_time = worker.worker_local_update(rewards, log_files_folder_path_comm_round, comm_round)
                                unverified_transaction = worker.return_local_updates_and_signature(comm_round)
								# size in bytes, usually around 35000 bytes per transaction
                                unverified_transactions_size = getsizeof(str(unverified_transaction))
                                transmission_delay = unverified_transactions_size/lower_link_speed
                                if local_update_spent_time + transmission_delay > validator.return_miner_acception_wait_time():
									# last transaction sent passes the acception time window
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter]['local_update_time'] = local_update_spent_time
                                records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
                                records_dict[worker][update_iter]['local_update_unverified_transaction'] = unverified_transaction
                                records_dict[worker][update_iter]['local_update_unverified_transaction_size'] = unverified_transactions_size
                                if update_iter == 1:
                                    total_time_tracker = local_update_spent_time + transmission_delay
                                else:
                                    total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1]['transmission_delay'] + local_update_spent_time + transmission_delay
                                records_dict[worker][update_iter]['arrival_time'] = total_time_tracker
                                if validator.online_switcher():
									# accept this transaction only if the validator is online
                                    print(f"validator {validator.return_idx()} has accepted this transaction.")
                                    transaction_arrival_queue[total_time_tracker] = unverified_transaction
                                else:
                                    print(f"validator {validator.return_idx()} offline and unable to accept this transaction")
                            else:
								# worker goes offline and skip updating for one transaction, wasted the time of one update and transmission
                                wasted_update_time, wasted_update_params = worker.waste_one_epoch_local_update_time(args['optimizer'])
                                wasted_update_params_size = getsizeof(str(wasted_update_params))
                                wasted_transmission_delay = wasted_update_params_size/lower_link_speed
                                if wasted_update_time + wasted_transmission_delay > validator.return_miner_acception_wait_time():
									# wasted transaction "arrival" passes the acception time window
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
                                if update_iter == 1:
                                    total_time_tracker = wasted_update_time + wasted_transmission_delay
                                    print(f"worker goes offline and wasted {total_time_tracker} seconds for a transaction")
                                else:
                                    total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1]['transmission_delay'] + wasted_update_time + wasted_transmission_delay
                            update_iter += 1
            else:
				 # did not specify wait time. every associated worker perform specified number of local epochs
                for worker_iter in range(len(associated_workers)):
                    worker = associated_workers[worker_iter]
                    if not worker.return_id() in validator.return_black_list():
                        print(f'worker {worker_iter+1}/{len(associated_workers)} of validator {validator.return_id()} is doing local updates')	 
                        if worker.online_switcher():
                            local_update_spent_time = worker.worker_local_update(rewards, log_files_folder_path_comm_round, comm_round, local_epochs=args.local_epochs)
                            worker_link_speed = worker.return_link_speed()
                            lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
                            unverified_transaction = worker.return_local_updates_and_signature(comm_round)
                            unverified_transactions_size = getsizeof(str(unverified_transaction))
                            transmission_delay = unverified_transactions_size/lower_link_speed
                            # put worker's message(unverified_transaction) into transaction_arrival_queue
                            if validator.online_switcher():
                                transaction_arrival_queue[local_update_spent_time + transmission_delay] = unverified_transaction
                                print(f"validator {validator.return_id()} has accepted this transaction.")
                            else:
                                print(f"validator {validator.return_id()} offline and unable to accept this transaction")
                        else:
                            print(f"worker {worker.return_id()} offline and unable do local updates")
                    else:
                        print(f"worker {worker.return_id()} in validator {validator.return_id()}'s black list. This worker's transactions won't be accpeted.")
            validator.set_unordered_arrival_time_accepted_worker_transactions(transaction_arrival_queue)
			# in case validator off line for accepting broadcasted transactions but can later back online to validate the transactions itself receives
            validator.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))
			
			# broadcast to other validators
            if transaction_arrival_queue:
                validator.validator_broadcast_worker_transactions(validators_this_round)
            else:
                print("No transactions have been received by this validator, probably due to workers and/or validators offline or timeout while doing local updates or transmitting updates, or all workers are in validator's black list.")
        print("Finish")
        # TODO 检查每个worker模型是不是自己在用