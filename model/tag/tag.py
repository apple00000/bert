import json
import multiprocessing
import os
from paddle import dtype
import torch
import util
from torch import nn
from d2l import torch as d2l
import pandas as pd
import model


# 每句最大限制
max_len = 100
type_len = 31

# 类别字典
LabelDict = {
'N':0, 
"address_b":1, "address_m":2, "address_e":3,
"book_b":4, "book_m":5, "book_e":6,
"company_b":7, "company_m":8, "company_e":9,
"game_b":10, "game_m":11, "game_e":12,
"government_b":13, "government_m":14, "government_e":15,
"movie_b":16, "movie_m":17, "movie_e":18,
"name_b":19, "name_m":20, "name_e":21,
"organization_b":22, "organization_m":23, "organization_e":24,
"position_b":25, "position_m":26, "position_e":27,
"scene_b":28, "scene_m":29, "scene_e":30
}

# 地址（address），
# 书名（book），
# 公司（company），
# 游戏（game），
# 政府（government），
# 电影（movie），
# 姓名（name），
# 组织机构（organization），
# 职位（position），
# 景点（scene）
def get_train_data(dir):
    sentences = []
    labels = []
    index = 0
    with open(dir, 'r') as f:
        strs = f.readlines()
        for str in strs:
            index = index + 1
            # if index >= 100:
            #     return sentences, labels

            str = str.rstrip('\n')
            s = json.loads(str)
            sentence = s['text'] 
            sentences.append(sentence)
            # "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}
            label = [0] * (max_len - 2)
            if 'label' in s:
                lebel_raw = s['label']
                # k:name, v:{"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}
                for k,v in lebel_raw.items():
                    # k2:叶老桂, v2:[[9, 11]]
                    for k2,v2 in v.items():
                        # v3:[9, 11]
                        for v3 in v2:
                            if v3[0] == v3[1]:
                                label[v3[0]] = LabelDict[k+"_e"]
                            else:
                                a = LabelDict[k+"_b"]
                                label[v3[0]] = LabelDict[k+"_b"]
                                label[v3[1]] = LabelDict[k+"_e"]
                                if v3[1]-v3[0] >= 2:
                                    for i in (v3[0]+1, v3[1]-1):
                                        label[i] = LabelDict[k+"_m"]
            labels.append(label)      
    return sentences, labels


# 标注任务
# ner标注
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.crf = model.CRF(type_len, True)
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, type_len)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        tmp = self.output(self.hidden(encoded_X[:, 1:max_len-1, :]))
        return tmp                    


def train(net, train_iter, trainer, num_epochs, devices=d2l.try_all_gpus()):
    # net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        print("epoch:",epoch)
        for i, (features, labels) in enumerate(train_iter):            
            train_batch(net, features, labels, trainer, devices)



def train_batch(net, X, y, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)

    pred_s = pred.argmax(dim=2)

    mask = []
    for x in X[2]:
        mask.append([1]*x+[0]*(max_len-2-x))
    mask = torch.tensor(mask, dtype=torch.bool)
    l = -net.crf.neg_log_likelihood_loss(pred, y, mask)

    print('pred_s:', pred_s[0])
    print("y:", y[0])
    print("loss:", l)

    sum = 0.0
    right = 0.0
    for i in range(len(pred_s)):
        for j in range(len(pred_s[i])):
            sum = sum + 1
            if pred_s[i][j] == y[i][j]:
                right = right + 1
    print("acc:", right/sum)

    l.sum().backward()
    trainer.step()



class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        sentenses, labels = get_train_data(dataset)

        all_premise_hypothesis_tokens = []
        for sentense in sentenses:
            sentense = str(sentense)
            if len(sentense) > max_len-2:
                sentense = sentense[:max_len-2]
            tmp = []
            for s in sentense:
                tmp.append(str(s))
            all_premise_hypothesis_tokens.append(tmp)
        print("all_premise_hypothesis_tokens", all_premise_hypothesis_tokens[0])

        self.labels = torch.tensor(labels)
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)

        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 使用4个进程
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens = premise_hypothesis_tokens
        tokens, segments = util.get_tokens_and_segments(p_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['[PAD]']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)



if __name__ == '__main__':
    batch_size = 512
    num_workers = 4
    devices = d2l.try_all_gpus()
    bert, vocab = util.load_pretrained_model_chinese(
    '../../data/chinese_L-12_H-768_A-12', num_hiddens=256, ffn_num_hiddens=512, num_heads=4, num_layers=2, dropout=0.1, max_len=512, devices=devices)

    train_set = SNLIBERTDataset('train.txt', max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)

    net = BERTClassifier(bert)

    lr, num_epochs = 1e-4, 100
    trainer = torch.optim.Adam(net.parameters(), lr=lr)

    train(net, train_iter, trainer, num_epochs, devices)

    torch.save(net.state_dict(), "tag.params")
    print("done")

