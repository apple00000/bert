import json
import os
import torch
from d2l import torch as d2l


# bert pytorch 特殊符号，需要和 chinese_L-12_H-768_A-12/vocab.txt 文件内容对应
CHAR_TOKEN_PAD = '[PAD]' # 填充符号
CHAR_TOKEN_CLS = '[CLS]' # 起始符号
CHAR_TOKEN_SEP = '[SEP]' # 分隔符号


"""
    转化为bert输入格式

    入参:
    sentenses: ['去北京','苹果好吃']
    max_len: 10
    vocab: bert词典，用于token->id的转化

    中间量：
    char_list_list:
    [['去', '北', '京'], ['苹', '果', '好', '吃']]

    return:
    all_token_ids: 每个句子拆分后的id编码
    [[101, 1343, 1266, 776, 102, 0, 0, 0, 0, 0], [101, 5741, 3362, 1962, 1391, 102, 0, 0, 0, 0]]
    all_segments: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    all_valid_lens: 有效长度
    [5, 6]
"""
def make_bert_data(sentenses, max_len, vocab):
    all_token_ids = []
    all_segments = []
    all_valid_lens = []

    char_list_list = []
    for sentense in sentenses:
        if len(sentense)==0:
            continue
        
        if len(sentense)>max_len-2:
            sentense = sentense[:max_len-2]
        char_list = []
        for s in sentense:
            char_list.append(s)
        char_list_list.append(char_list)

    for char_list in char_list_list:
        tokens, segments = get_tokens_and_segments(char_list)
        token_ids = vocab[tokens] + [vocab[CHAR_TOKEN_PAD]] * (max_len - len(tokens))
        segments = segments + [0] * (max_len - len(segments))
        valid_len = len(tokens)

        all_token_ids.append(token_ids)
        all_segments.append(segments)
        all_valid_lens.append(valid_len)

    return all_token_ids, all_segments, all_valid_lens


'''
    获取bert格式的token解析，添加标记为和生成掩码
    入参：
    tokens_a：拆分后的句子，如 ['去', '北', '京']
    tokens_b：同tokens_a，可不传

    返回：
    tokens：添加分隔符号的token，如 ['CLS', '去', '北', '京', 'SEP']
    segments：区分两个句子的0,1掩码
'''
def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = [CHAR_TOKEN_CLS] + tokens_a + [CHAR_TOKEN_SEP]
    # 0 and 1 表示 segment A and B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + [CHAR_TOKEN_SEP]
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


# 加载英文bert模型
def load_pretrained_model_english(data_dir, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    # bert.small.torch
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[num_hiddens],
                         ffn_num_input=num_hiddens, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                         max_len=max_len, key_size=num_hiddens, query_size=num_hiddens,
                         value_size=num_hiddens, hid_in_features=num_hiddens,
                         mlm_in_features=num_hiddens, nsp_in_features=num_hiddens)
    bert.load_state_dict(torch.load(os.path.join(data_dir,'pretrained.params')))
    return bert, vocab


# 加载中文bert模型
def load_pretrained_model_chinese(data_dir, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len):         
    # chinese_L-12_H-768_A-12
    vocab = d2l.Vocab()
    vc = open(os.path.join(data_dir, 'vocab.txt'))
    tmp = vc.readlines()
    tmp2 = []
    for t in tmp:
        tmp2.append(t.strip())
    vocab.idx_to_token = tmp2
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[num_hiddens],
                         ffn_num_input=num_hiddens, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                         max_len=max_len, key_size=num_hiddens, query_size=num_hiddens,
                         value_size=num_hiddens, hid_in_features=num_hiddens,
                         mlm_in_features=num_hiddens, nsp_in_features=num_hiddens)
    t = torch.load(os.path.join(data_dir,'pytorch_model.bin'))
    bert.load_state_dict(t, False)
    return bert, vocab