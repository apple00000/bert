import torch
import bert_pytorch_util
from d2l import torch as d2l
import bi_classify


if __name__ == '__main__':
    devices = d2l.try_all_gpus()
    bert, vocab = bert_pytorch_util.load_pretrained_model_chinese(
    '../../data/chinese_L-12_H-768_A-12', num_hiddens=256, ffn_num_hiddens=512, num_heads=4, num_layers=2, dropout=0, max_len=512, devices=devices)

    train_set = bi_classify.SNLIBERTDataset('test.csv', bi_classify.max_len+2, vocab)
    print("train_set",train_set.all_token_ids)
    train_iter = torch.utils.data.DataLoader(train_set, 1, shuffle=False, num_workers = 4)

    net = bi_classify.BERTClassifier(bert)
    net.load_state_dict(torch.load('bi_classify.params'))

    # 预测
    for i, (features, labels) in enumerate(train_iter):
        pred = net(features)
        print("i",i)
        print("features",features)
        print("pred",pred)
        print("labels",labels)
