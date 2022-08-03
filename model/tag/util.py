import json
import os
import torch
from d2l import torch as d2l

# 获取 segments_token
def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['[SEP]']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


# 加载英文bert模型
def load_pretrained_model_english(data_dir, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    # bert.small.torch
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    bert.load_state_dict(torch.load(os.path.join(data_dir,'pretrained.params')))
    return bert, vocab


# 加载中文bert模型
def load_pretrained_model_chinese(data_dir, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):         
    # chinese_L-12_H-768_A-12
    vocab = d2l.Vocab()
    vc = open(os.path.join(data_dir, 'vocab.txt'))
    tmp = vc.readlines()
    tmp2 = []
    for t in tmp:
        tmp2.append(t.strip())
    vocab.idx_to_token = tmp2
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    t = torch.load(os.path.join(data_dir,'pytorch_model.bin'))
    bert.load_state_dict(t, False)
    return bert, vocab