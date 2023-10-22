import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorForSeq2Seq
from transformers.modeling_outputs import SequenceClassifierOutput
from prefix_encoder import PrefixEncoder
import pandas as pd
from os import path
from torchkeras import KerasModel


class MyDataset(Dataset):
    def __init__(self, df, tokenizer,
                 prompt_col='prompt',
                 response_col='response',
                 history_col='history',
                 max_context_length=1024,
                 max_target_length=1024
                 ):
        super().__init__()
        self.__dict__.update(locals())

        self.df = df
        self.tokenizer = tokenizer
        self.prompt_col = prompt_col
        self.history_col = history_col
        self.response_col = response_col
        self.max_context_length = max_context_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def get(self, index):
        data = dict(self.df.iloc[index])
        example = {}
        # context根据prompt和history以及
        example['context'] = self.tokenizer.build_prompt(
            query=data[self.prompt_col],
            history=data.get(self.history_col, None))
        example['target'] = data[self.response_col]
        return example

    def __getitem__(self, index):
        example = self.get(index)
        a_ids = self.tokenizer.encode(text=example['context'],
                                      add_special_tokens=True, truncation=True,
                                      max_length=self.max_context_length)
        b_ids = self.tokenizer.encode(text=example['target'],
                                      add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]

        # 专注于 b_ids和 最后的eos_token_id的学习
        labels = [-100] * len(a_ids) + b_ids + [self.tokenizer.eos_token_id]
        return {'input_ids': input_ids, 'labels': labels}


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # [b,seq,dim]  [b,seq]->[b,seq,1]->[b,seq,dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # [b,dim]/sum(seq)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # [1,1,1,1,0,0] 4  [768] [0.01,...,0.03]*[0,...0]
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class PtuingModel(nn.Module):
    def __init__(self, ptv2_cfg):
        super(PtuingModel, self).__init__()

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.output_hidden_states = True
        self.ptv2_cfg = ptv2_cfg


        self.prefix_encoder = PrefixEncoder(self.ptv2_cfg)

        self.dropout = torch.nn.Dropout(self.ptv2_cfg.hidden_dropout_prob)

        self.bert = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = self.ptv2_cfg.pre_seq_len
        self.n_layer = self.ptv2_cfg.num_hidden_layers
        self.n_head = self.ptv2_cfg.num_attention_heads
        self.n_embd = self.ptv2_cfg.hidden_size // self.ptv2_cfg.num_attention_heads

        # [0,....127]
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        dim = ptv2_cfg.hidden_size

        # self.dropout = nn.Dropout(0.2)
        self.pooler = MeanPooling()
        self.cls = nn.Linear(dim, 36)

    def get_prompt(self, batch_size):

        # [127].unsqueeze(0) -> [1,127] -> [batch_size,127]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)


        # (batch-size, 128, 2*12*768)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        # [batch_size,127,2*12*768] -> [batch_size,128,2*12,768,127]
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values = self.dropout(past_key_values)
        # [2, 0, 3, 1, 4] means [n_layer*2,batch_size,n_head,pre_seq_len,n_embd]

        # [2*12,batch_size,12,128,64]->([12,batch_size,12,128,64],[12,batch_size,12,128,64])
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, attention_mask, labels=None):


        batch_size = input_ids.shape[0]
        attention_mask_orgin = attention_mask.clone()

        # kv 多头的kv向量
        # ([12,batch_size,12,128,64],[12,batch_size,12,128,64])
        past_key_values = self.get_prompt(batch_size)

        # 【1，1，1】 【1，1】 【1，1，1，1，1】
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # prefix_tuing 与 bert 拼接
        # ic(input_ids.shape)
        # ic(attention_mask.shape)
        # ic(past_key_values[0].shape)
        # ic(self.bert.embeddings.position_embeddings)
        # ([12,batch_size,12,128,64],[12,batch_size,12,128,64])
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
        # [b,seq,dim] ->[b,dim]
        output_last = bert_out.hidden_states[-1]
        # [b,seq,dim]
        output = self.pooler(output_last, attention_mask_orgin)

        output = self.dropout(output)
        # [b,36]
        output = self.cls(output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # focal_Loss, LabelSmoothingCrossEntropy, klloss
            # 958 / 175 = 5.4  958/958 = 1 985/2 = 492  985/985 = 1
            loss = loss_fn(output, labels)
        return SequenceClassifierOutput(logits=output, loss=loss)

class p_tuningv2_config():
    def __init__(self):
        # ptuning parameter
        self.prefix_projection = True
        self.pre_seq_len = 128
        self.prefix_hidden_size = auto_config.hidden_size
        # roberta parameter
        self.hidden_size = auto_config.hidden_size
        self.num_hidden_layers = auto_config.num_layers#auto_config.num_hidden_layers
        self.num_attention_heads = auto_config.num_attention_heads
        # hidden_dropout_prob = auto_config.hidden_dropout_prob
        self.hidden_dropout_prob = 0.3


if __name__ == "__main__":
    # 参数
    model_path = path.abspath('E:/2022CFF-Small-sample-data-classification-task/utils/chat_glm_model')
    ckpt_path = "output"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    auto_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    ptv2_cfg = p_tuningv2_config()
    model = PtuingModel(ptv2_cfg)  # AutoModel.from_pretrained(model_path, trust_remote_code=True)


    # 数据准备
    emotes = pd.read_csv("csv_6w_str.csv")
    icons = emotes["icons"].tolist()
    descriptions = emotes["titles"].tolist()
    # descriptions = ["what is the kaomoji used for the sentence " + "\"" + i + "\"" + "?" for i in descriptions]
    data = {'prompt': descriptions[:41384], 'response': icons[:41384]}
    dfdata = pd.DataFrame(data)

    data_val = {'prompt': descriptions[-2:-1], 'response': icons[-2:-1]}
    dfdata_val = pd.DataFrame(data_val)

    ds_train = MyDataset(dfdata, tokenizer)
    ds_val = MyDataset(dfdata_val, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=True
    )

    dl_train = DataLoader(ds_train, batch_size=24,
                          num_workers=2, shuffle=True, collate_fn=data_collator
                          )
    dl_val = DataLoader(ds_val, batch_size=1,
                        num_workers=2, shuffle=False, collate_fn=data_collator
                        )

    for batch in dl_train:
        print()
        break


    # 开始训练
    optimizer = torch.optim.adamw.AdamW(model.parameters(), lr=5e-05)

    keras_model = KerasModel(model, loss_fn=None,
                             optimizer=optimizer)

    dfhistory = keras_model.fit(train_data=dl_train,
                                val_data=dl_val,
                                epochs=10,
                                patience=4,
                                monitor='val_loss',
                                mode='min',
                                ckpt_path=ckpt_path,
                                gradient_accumulation_steps=2
                                )