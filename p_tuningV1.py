import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, SubsetRandomSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoConfig

SAVE_PATH = 'weights/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_template(num_prompt_tokens):
    """
    生成模板，并扩充 tokenizer。模板的形式多种多样，可根据需求设计模板生成函数。本代码仅用于生成如下前缀模板：
        '<prompt_1><prompt_2>[MASK] <prompt_3>...<prompt_n-2><prompt_n-1><prompt_n>'

    Args:
        num_prompt_tokens:

    Returns:
        tokenizer:
        template:
    """
    assert num_prompt_tokens >= 2
    prefix_template = '<prompt_1><prompt_2>[MASK][MASK]'

    for i in range(3, num_prompt_tokens + 1):
        prefix_template = prefix_template + '<prompt_{}>'.format(i)

    return prefix_template


def extend_tokenizer(num_prompt_tokens, tokenizer):
    """
    将 prompt tokens 加入到分词器中，方便做预处理

    Args:
        num_prompt_tokens:

    Returns:
        tokenizer:
    """
    prompt_tokens = []
    for i in range(1, num_prompt_tokens + 1):
        token = '<prompt_{}>'.format(i)
        prompt_tokens.append(token)

    tokenizer.add_special_tokens({"additional_special_tokens": prompt_tokens})

    return tokenizer


class PromptEncoder(nn.Module):
    def __init__(self, num_prompt_tokens, offset, embdding_dim=768):
        super(PromptEncoder, self).__init__()
        self.offset = offset
        self.embedding = torch.nn.Embedding(num_prompt_tokens, embdding_dim)  # [1,2,3,4,5,6]
        # 四个双向gru
        self.gru = torch.nn.GRU(embdding_dim, 128, bidirectional=True, num_layers=2, batch_first=True)
        self.mlp_1 = nn.Linear(256, 512)
        self.mlp_2 = nn.Linear(512, 768)

    def forward(self, prompt_token_ids, prompt_ids=None):
        """
            1. LSTM
            2. MLP
        """
        prompt_token_ids = prompt_token_ids - self.offset

        out = self.embedding(prompt_token_ids)
        # 给out增加batch维度
        out = torch.unsqueeze(out, 0)  # [1,64,1024]
        out, _ = self.gru(out)  # [1,64,1024]
        out = self.mlp_1(out)
        out = self.mlp_2(out)  # [1,64,1024]
        out = torch.squeeze(out, 0)  # [64,1024]

        return out


class BertDataset(Dataset):

    def __init__(self, data_path, max_len=250, num_prompt_tokens=20):
        super(BertDataset, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.prefix = generate_template(num_prompt_tokens)
        self.tokenizer = extend_tokenizer(num_prompt_tokens, self.tokenizer)

        self.tokens_list, self.atten_mask_list, self.label_list = self.process_data(data_path, max_len)

    def process_data(self, data_path, max_len):

        data = pd.read_csv(data_path)
        data = data[:5]
        atten_mask_list = []
        tokens_list = []
        labels_list = []

        for i, line in enumerate(data["Comment"].tolist()):
            if isinstance(line, str):
                text_tokens = self.tokenizer.tokenize(line)
                prefix = self.tokenizer.tokenize(self.prefix)

                star = data['Star'].tolist()
                if star[i] < 4:
                    star_token = self.tokenizer.tokenize("差评")
                else:
                    star_token = self.tokenizer.tokenize("好评")

                tokens = prefix + ["[SEP]"] + text_tokens + ["[SEP]"] + star_token + ["[SEP]"]
                labels = (len(prefix) + 1 + len(text_tokens) + 1) * [-100] + \
                         self.tokenizer.encode(star_token, add_special_tokens=False) + [-100]

                if len(tokens) < max_len:
                    diff = max_len - len(tokens)
                    attn_mask = [1] * len(tokens) + [0] * diff
                    tokens += ["[PAD]"] * diff

                    labels += [-100] * (max_len - len(labels))
                else:
                    tokens = tokens[:max_len - 1] + ["[SEP]"]
                    attn_mask = [1] * max_len

                    labels += labels[:max_len - 1] + [-100]

                tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                atten_mask_list.append(attn_mask)
                tokens_list.append(tokens_ids)
                labels_list.append(labels)

        return tokens_list, atten_mask_list, labels_list

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, index):

        return torch.tensor(self.tokens_list[index]), \
               torch.tensor(self.atten_mask_list[index]), \
               torch.tensor(self.label_list[index])


class BertLoader():

    def __init__(self, batch_size):
        train_dataset = BertDataset("data/DMSC.csv")

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
        self.num_prompt_tokens = 64
        self.prompt_embdding = PromptEncoder(self.num_prompt_tokens, 21128)  # 21128

        self.original_vocab_size = self.bert.config.vocab_size
        self.prompt_token_fn = lambda t: (t >= self.original_vocab_size)

        self.embedding = self.bert.get_input_embeddings()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 21128)

    def forward(self, input_ids, atten_mask):
        prompt_masks = self.prompt_token_fn(input_ids)
        input_ids_ = input_ids.clone()

        input_ids_[prompt_masks] = 0  # 方便直接调用 ber_embedding

        inputs_embeds = self.embedding(input_ids_)
        prompt_embeds = self.prompt_embdding(input_ids[prompt_masks]).to(device=inputs_embeds.device)

        inputs_embeds[prompt_masks] = prompt_embeds


        bert_output = self.bert(inputs_embeds=inputs_embeds,
                               attention_mask=atten_mask,
                               output_hidden_states=True)
        outs = bert_output['last_hidden_state']
        outs = self.dropout(outs)
        outs = self.classifier(outs)

        return outs


def validate(model, data_loader, criteon):
    model.eval()
    dev_loss, label_lists, pred_lists = [], [], []
    for tokens, masks, labels in data_loader:
        tokens = tokens.to(DEVICE)
        masks = masks.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            preds = model(tokens, masks)


        # importance weighting loss
        loss = criteon(preds.reshape(-1, preds.shape[-1]), labels.reshape(-1))

        dict_ver = {'2345-6397': 0, #'差评',
                    '1962-6397': 1} #'好评'

        row_indices = []
        label_list = []
        pred_list = []

        # 遍历每一行
        for row in labels.detach().cpu().numpy():
            # 使用np.where找到True元素的索引
            indices = [i for i, n in enumerate(row) if n != -100]
            # 将索引数组添加到列表中
            row_indices.append(indices)

        for i in range(preds.shape[0]):
            pro_best = []
            # 求两个token的dict_ver里的概率
            for key_token in dict_ver.keys():
                idx1, idx2 = key_token.split('-')
                score_tmp = preds[i, row_indices[i][0], int(idx1)] + preds[i, row_indices[i][1], int(idx2)]
                pro_best.append(score_tmp)
            # 求pro_best里的最大值的下标
            idx_best = np.argmax(pro_best)
            # 将最大值的下标对应的两个token加入到preds_temp和gts_temp中
            pred_list.append(idx_best)
            labels = labels.detach().cpu().numpy()
            label = "-".join([str(labels[i][row_indices[i][0]])] + [str(labels[i][row_indices[i][1]])])
            label_list.append(dict_ver[label])

        dev_loss.append(loss.item())
        label_lists.append(label_list)
        pred_lists.append(pred_list)


        torch.cuda.empty_cache()

    pred_lists = np.concatenate(pred_lists, axis=0)
    label_lists = np.concatenate(label_lists, axis=0)

    correct = (pred_lists == label_lists).sum()
    return np.array(dev_loss).mean(), float(correct) / len(label_lists)


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

        train_loss, label_lists, pred_lists = [], [], []
        for idx, (tokens, masks, labels) in enumerate(bar):
            tokens = tokens.to(DEVICE)
            masks = masks.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero gradient
            optimizer.zero_grad()

            preds = model(tokens, masks)
            loss = criteon(preds.reshape(-1, preds.shape[-1]), labels.reshape(-1))

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            dict_ver = {'2345-6397': 0,  # '差评',
                        '1962-6397': 1}  # '好评'
            row_indices = []
            label_list = []
            pred_list = []

            labels = labels.detach().cpu().numpy()
            # 遍历每一行
            for row in labels:
                # 使用np.where找到True元素的索引
                indices = [i for i, n in enumerate(row) if n != -100]
                # 将索引数组添加到列表中
                row_indices.append(indices)

            preds = preds.detach().cpu().numpy()
            for i in range(preds.shape[0]):
                pro_best = []
                # 求两个token的dict_ver里的概率
                for key_token in dict_ver.keys():
                    idx1, idx2 = key_token.split('-')
                    score_tmp = preds[i, row_indices[i][0], int(idx1)] + preds[i, row_indices[i][1], int(idx2)]
                    pro_best.append(score_tmp)
                # 求pro_best里的最大值的下标
                idx_best = np.argmax(pro_best)
                # 将最大值的下标对应的两个token加入到preds_temp和gts_temp中
                pred_list.append(idx_best)

                label = "-".join([str(labels[i][row_indices[i][0]])] + [str(labels[i][row_indices[i][1]])])
                label_list.append(dict_ver[label])

            train_loss.append(loss.item())
            label_lists.append(label_list)
            pred_lists.append(pred_list)

            # empty cache
            torch.cuda.empty_cache()

            bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(epoch, idx, loss.item()))

        train_loss = np.array(train_loss).mean()
        val_loss, val_acc = validate(model, val_loader, criteon)

        pred_lists = np.concatenate(pred_lists, axis=0)
        label_lists = np.concatenate(label_lists, axis=0)

        correct = (pred_lists == label_lists).sum()
        train_acc = float(correct) / len(label_lists)

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

    model.eval()
    with torch.no_grad():
        output,logtic_p = model(input_ids=tokens, atten_mask=masks)
        labels = logtic_p[:, 3:5, :].cpu().numpy()
        preds = F.softmax(output, dim=-1)

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
    train(lr=5e-4, weight_decay=1e-3, clip=0.8, epochs=1, optimizer="adam", batch_size=128)
    # predict('好看的，赞，推荐给大家', max_len=200, weights_path="weights/epoch0_0.8407.pt")
    # predict('什么破烂反派，毫无戏剧冲突能消耗两个多小时生命，还强加爱情戏。脑残片好圈钱倒是真的', max_len=200, weights_path="weights/epoch0_0.8407.pt")
