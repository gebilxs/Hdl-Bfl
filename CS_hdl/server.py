import torch

from client import clientHdl
from serverbase import Server
from collections import defaultdict
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
    # agg_logits_label = []
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
            

# def logit_aggregation(local_logits_list):
#     # Assuming each local_logits entry has the same labels, 
#     # and each label's tensor is of the same size across entries.
#     # First, collect all logits for each label across all local models.
#     agg_logits_label = defaultdict(list)
#     for local_logits in local_logits_list:
#         for label, logit in local_logits.items():
#             agg_logits_label[label].append(logit)
    
#     # Now, average the collected logits for each label.
#     # Initialize a placeholder for the aggregated logits.
#     aggregated_logits = {}
#     for label, logit_list in agg_logits_label.items():
#         # Stack all logits for the current label into a new dimension and then average across this dimension.
#         stacked_logits = torch.stack(logit_list, dim=0)
#         average_logits = torch.mean(stacked_logits, dim=0)
#         aggregated_logits[label] = average_logits
    
#     # Convert aggregated_logits to a single tensor if required.
#     # Note: The following step assumes you want to concatenate all label logits into a single tensor.
#     # This step might need adjustment based on how you intend to use the aggregated logits.
#     labels_sorted = sorted(aggregated_logits.keys())  # Ensure consistent order
#     final_aggregated_tensor = torch.cat([aggregated_logits[label] for label in labels_sorted], dim=0)

#     return final_aggregated_tensor
