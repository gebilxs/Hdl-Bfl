from Devicebase import Device
from utils.data_utils import read_client_data
import random
class DevicesInNetwork(Device):
    # include send data like server load_data to every client
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.id = id
        # if IID is 1 -> NonIID 
        # if IID is 0 -> normal dataset
        self.batch_size = args.batchsize
        self.learning_rate = args.local_learning_rate
        self.opti = args.optimizer
        self.num_devices = args.num_devices
        self.model = args.model_name

        self.devices_after_load_data = {}
        self.malicious_nodes_set = []
        self.num_malicious = args.num_malicious
        self.devices_dict = None
        self.aio = False
        self.pow_difficulty = args.pow_difficulty
        self.peer_list = set()
        self.online = True
        self.devices_set = {}
        if self.num_malicious:
            self.malicious_nodes_set = random.sample(range(self.num_devices), self.num_malicious)
    # TODO every client get train_data and test_data
        self.data_set_dir_allocation(args,Device)

    
    
    def data_set_dir_allocation(self,args,deviceObj):
        # TODO add malicious_node set
        for i in range (self.num_devices):
            args.is_malicious = False
            train_data = read_client_data(self.dataset_name, i,is_train=True)
            test_data = read_client_data(self.dataset_name,i,is_train=False)
            if i in self.malicious_nodes_set:
                args.is_malicious = True
            device = deviceObj(self.args,
                               id = i,
                               train_samples = len(train_data),
                               test_samples = len(test_data),
                               )
            

            self.devices_after_load_data[i] = device


