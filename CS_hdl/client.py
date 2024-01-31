
from clientbase import Client
import torch.nn as nn
class clientHdl(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.logits = None
        self.global_logits = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

    def train(self):
        print()