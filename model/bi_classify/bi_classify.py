import torch
import bert_pytorch_util as util
from torch import nn
from d2l import torch as d2l
import pandas as pd


# 单分类任务
# 情感分析
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        # 2表示分2类
        self.output = nn.Linear(256, 2)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        # 单分类任务，这里的0表示取编码第一位 [CLS]
        # encoded_X维度：batch_size，句子长度，单字向量维度(768)
        return self.output(self.hidden(encoded_X[:, 0, :]))                          


def train(net, train_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        print("epoch：", epoch)
        for i, (features, labels) in enumerate(train_iter):            
            train_batch(net, features, labels, loss, trainer, devices)
        
        torch.save(net.state_dict(),"bi_classify.params")


def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    print("loss：", l.mean())
    l.sum().backward()
    trainer.step()



class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_len, vocab):
        dataset = pd.read_csv(dataset_path)
        dataset = dataset.dropna(axis=0)

        sentenses = dataset['review'].tolist()
        labels = dataset['label'].tolist()

        all_token_ids, all_segments, all_valid_lens = util.make_bert_data(sentenses, max_len, vocab)

        self.all_token_ids = torch.LongTensor(all_token_ids)
        self.all_segments = torch.LongTensor(all_segments)
        self.all_valid_lens = torch.LongTensor(all_valid_lens)
        self.labels = torch.LongTensor(labels)
        print('[Dataset] load ' + str(len(self.all_token_ids)) + ' data')

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.all_valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)


if __name__ == '__main__':
    bert, vocab = util.load_pretrained_model_chinese(
    '../../data/chinese_L-12_H-768_A-12', num_hiddens=256, ffn_num_hiddens=512, num_heads=4, num_layers=2, dropout=0.1, max_len=512)

    train_set = Dataset('train.csv', 100, vocab)
    print('train_set', train_set)
    train_iter = torch.utils.data.DataLoader(train_set, 512, shuffle=True, num_workers=4)

    net = BERTClassifier(bert)

    lr, num_epochs = 1e-4, 500
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')

    train(net, train_iter, loss, trainer, num_epochs, d2l.try_all_gpus())

