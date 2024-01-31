import torch

from client import clientHdl
from serverbase import Server
class Hdl(Server):
    def __init__(self,args,times):
        super().__init__(args,times)
        self.set_clients(clientHdl)
        
        print("\nevaluate global model on all clients")
        self.global_logits = [None for _ in range(args.num_classes)]

    def train(self):
        for i in range(self.global_round+1):
            self.selected_clients = self.select_clients()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
            for client in self.selected_clients:
                client.train()
        
        
        