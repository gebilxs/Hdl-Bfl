import torch

from clientbaseline import clientHbase
from serverbase import Server
from collections import defaultdict
class Hbase(Server):
    def __init__(self,args,times):
        super().__init__(args,times)
        self.set_clients(clientHbase)
        
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
            
            self.receive_logits()
            self.global_logits = logit_aggregation(self.uploaded_logits)

            self.send_logits()
        print(max(self.rs_test_acc))

    def send_logits(self):
        assert (len(self.clients) > 0)

        for client in self.clients:

            client.set_logits(self.global_logits)

    def receive_logits(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_logits = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_logits.append(client.logits)
        
# https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L221
def logit_aggregation(local_logits_list):
    agg_logits_label = defaultdict(list)
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

    return agg_logits_label