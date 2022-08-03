import json
import multiprocessing
import os
import torch
import util
from torch import nn
from d2l import torch as d2l
import pandas as pd

ctc_loss = nn.CTCLoss()
log_probs = torch.randn(5, 8, 10).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 10, (8, 30), dtype=torch.long)
input_lengths = torch.full((8,), 5, dtype=torch.long)
target_lengths = torch.full((8,), 5, dtype=torch.long)
# target_lengths = torch.randint(10,30,(8,), dtype=torch.long)

for i in range(100):
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

    # print('log_probs',log_probs.size())
    # print('targets',targets.size())
    # print('input_lengths',input_lengths.size())
    # print('target_lengths',target_lengths.size())

    # print('log_probs',log_probs)
    # print('targets',targets)
    # print('input_lengths',input_lengths)
    # print('target_lengths',target_lengths)
    print('loss',loss)

    loss.backward()