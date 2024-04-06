import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, SubsetRandomSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

SAVE_PATH = 'weights/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BertDataset(Dataset):

    def __init__(self, data_path, max_len=100):
        super(BertDataset, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        self.tokens_list, self.atten_mask_list, self.label_list = self.process_data(data_path, max_len)

    def process_data(self, data_path, max_len):

        data = pd.read_csv(data_path)
        atten_mask_list = []
        tokens_list = []

        for line in data["Comment"].tolist():
            if isinstance(line, str):
                text_tokens = self.tokenizer.tokenize(line)

                tokens = ["[CLS]"] + text_tokens + ["[SEP]"]

                if len(tokens) < max_len:
                    diff = max_len - len(tokens)
                    attn_mask = [1] * len(tokens) + [0] * diff
                    tokens += ["[PAD]"] * diff
                else:
                    tokens = tokens[:max_len - 1] + ["[SEP]"]
                    attn_mask = [1] * max_len

                tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                atten_mask_list.append(attn_mask)
                tokens_list.append(tokens_ids)

        return tokens_list, atten_mask_list, data['Star'].tolist()

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, index):

        return torch.tensor(self.tokens_list[index]), \
               torch.tensor(self.atten_mask_list[index]), \
               0 if self.label_list[index] < 4 else 1


class BertLoader():

    def __init__(self, batch_size):
        train_dataset = BertDataset("../data/DMSC.csv")

        validation_split = .2
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        # 对数据进行shuffle
        np.random.seed(42)
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self._train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=batch_size,
                                                         sampler=train_sampler)
        self._val_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=batch_size,
                                                       sampler=valid_sampler)

    def get_train_loader(self):
        return self._train_loader

    def get_val_loader(self):
        return self._val_loader
    # def get_test_loader(self):
    #     return self._test_loader


class DMSCDModel(nn.Module):
    def __init__(self):
        super(DMSCDModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, atten_mask):
        bert_output = self.bert(input_ids, attention_mask=atten_mask)  # cont_reps是last_hidden_state
        features = bert_output[1]  # 这是序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。
        cls_rep = self.dropout(features)
        outs = self.classifier(cls_rep)
        return outs


def validate(model, data_loader, criteon):
    model.eval()
    dev_loss, label_list, pred_list = [], [], []
    for tokens, masks, labels in data_loader:
        tokens = tokens.to(DEVICE)
        masks = masks.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            preds = model(tokens, masks)

        # importance weighting loss
        loss = criteon(preds, labels)

        dev_loss.append(loss.item())
        pred_list.append(preds.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())

        torch.cuda.empty_cache()

    pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)
    label_list = np.concatenate(label_list, axis=0)

    correct = (pred_list == label_list).sum()
    return np.array(dev_loss).mean(), float(correct) / len(label_list)


def train(batch_size, optimizer, lr, weight_decay, epochs, clip):
    dataloader = BertLoader(batch_size)
    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    # construct data loader
    model = DMSCDModel()
    model = model.to(DEVICE)
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    criteon = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # log process
    for epoch in range(epochs):
        bar = tqdm(train_loader, total=len(train_loader))
        model.train()

        train_loss, label_list, pred_list = [], [], []
        for idx, (tokens, masks, labels) in enumerate(bar):
            tokens = tokens.to(DEVICE)
            masks = masks.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero gradient
            optimizer.zero_grad()

            preds = model(tokens, masks)
            loss = criteon(preds, labels)

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss.append(loss.item())
            label_list.append(labels.detach().cpu().numpy())
            pred_list.append(preds.detach().cpu().numpy())

            # empty cache
            torch.cuda.empty_cache()

            bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(epoch, idx, loss.item()))

        train_loss = np.array(train_loss).mean()
        val_loss, val_acc = validate(model, val_loader, criteon)

        pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)
        label_list = np.concatenate(label_list, axis=0)

        correct = (pred_list == label_list).sum()
        train_acc = float(correct) / len(label_list)

        torch.save(model.state_dict(), SAVE_PATH + "epoch{}_{:.4f}.pt".format(epoch, val_acc))
        print('Training loss:{}, Val loss:{}'.format(train_loss, val_loss))
        print("train acc:{:.4f}, val acc:{:4f}".format(train_acc, val_acc))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    plt.figure()
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.figure()
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'])
    plt.show()


def predict(sentence, max_len, weights_path):
    # weibo_loader = BertLoader(batch_size, ROOT_PATH, max_len)
    # test_loader = weibo_loader.get_test_loader()

    # construct data loader
    model = DMSCDModel()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    # criterion = nn.CrossEntropyLoss()

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    text_tokens = tokenizer.tokenize(sentence)

    tokens = ["[CLS]"] + text_tokens + ["[SEP]"]

    if len(tokens) < max_len:
        diff = max_len - len(tokens)
        attn_mask = [1] * len(tokens) + [0] * diff
        tokens += ["[PAD]"] * diff
    else:
        tokens = tokens[:max_len - 1] + ["[SEP]"]
        attn_mask = [1] * max_len

    tokens = tokenizer.convert_tokens_to_ids(tokens)

    tokens = torch.tensor([tokens]).to(DEVICE)
    masks = torch.tensor([attn_mask]).to(DEVICE)

    # print(tokens)
    # print(masks)

    # model.eval()
    # with torch.no_grad():
    preds = F.softmax(model(input_ids=tokens, atten_mask=masks), dim=-1)

    # entroy = nn.CrossEntropyLoss()
    # target1 = torch.tensor([0])
    # target2 = torch.tensor([1])
    #
    # print(entroy(preds, target1), entroy(preds, target2))
    return preds


if __name__ == "__main__":
    # SAVE_PATH = 'weights/'
    #
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # th number 3 has the highest priority
    train(lr=5e-4, weight_decay=1e-3, clip=0.8, epochs=50, optimizer="sgd", batch_size=128)
    # predict('好看的，赞，推荐给大家', max_len=200, weights_path="weights/epoch0_0.8407.pt")
    # predict('什么破烂反派，毫无戏剧冲突能消耗两个多小时生命，还强加爱情戏。脑残片好圈钱倒是真的', max_len=200, weights_path="weights/epoch0_0.8407.pt")
