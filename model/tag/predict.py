import torch
import util
from d2l import torch as d2l
import tag


if __name__ == '__main__':
    devices = d2l.try_all_gpus()
    bert, vocab = util.load_pretrained_model_chinese(
    '../../data/chinese_L-12_H-768_A-12', num_hiddens=256, ffn_num_hiddens=512, num_heads=4, num_layers=2, dropout=0.1, max_len=512, devices=devices)

    train_set = tag.SNLIBERTDataset('test.txt', tag.max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set, 1, shuffle=False, num_workers = 1)

    net = tag.BERTClassifier(bert)
    net.load_state_dict(torch.load('tag.params'))

    # 预测
    for i, (features, labels) in enumerate(train_iter):
        if i<=10:
            pred = net(features)
            print("i",i)
            # print("features",features)
            print("pred",pred)
            print("labels",labels)
            print("a0",pred[0][0])
            print("a1",pred[0][1])
            print(pred[0].argmax(dim=1))
