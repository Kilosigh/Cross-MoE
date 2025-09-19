import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.data.processors.squad import squad_convert_example_to_features_init

from layers.MyLayers import MLP, CrossRanker
from models import Transformer
from transformers import BertConfig, BertModel, BertTokenizer



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.device = configs.device
        self.model = configs.model

        bert_config_path = './BERT_files/local_bert_config'
        if os.path.exists(bert_config_path):
            self.bert_config = BertConfig.from_pretrained(bert_config_path)
        else:
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.save_pretrained(bert_config_path)
        self.bert_config.num_hidden_layers = configs.llm_layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True
        self.bert_config.max_position_embeddings = 512
        bert_base_uncased_tokenizer_path = './BERT_files/bert-base-uncased-tokenizer'
        bert_base_uncased_encoder_path = './BERT_files/bert-base-uncased-text_encoder'
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                bert_base_uncased_tokenizer_path,
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Attempting to download them..")
            self.tokenizer = BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False
            )
            self.tokenizer.save_pretrained(bert_base_uncased_tokenizer_path)

        try:
            self.text_encoder = BertModel.from_pretrained(
                bert_base_uncased_encoder_path,
                trust_remote_code=True,
                local_files_only=True,
                config=self.bert_config,
            )
        except EnvironmentError:  # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            self.text_encoder = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
                config=self.bert_config,
            )
            self.text_encoder.save_pretrained(bert_base_uncased_encoder_path)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_encoder = self.text_encoder.to(self.device)

        # print(type(configs.d_model))
        # print(configs.d_model)
        self.text_mlp = MLP([configs.llm_dim, int(configs.llm_dim / 8), configs.d_model], dropout_rate=0.1)

        if not configs.use_Cross_ranker:
            return
        if configs.model == "PatchTST" or configs.model == "TimeLLM":
            patch_len = configs.patch_len
            if patch_len>configs.pred_len:
                patch_len = configs.pred_len
            text_top_k = int((configs.seq_len - patch_len) / configs.stride + 2)
        elif configs.model == "iTransformer":
            freq_dict = {'m': 1, 'w': 2, 'd': 3}
            text_top_k = configs.enc_in + freq_dict[configs.freq]
        elif configs.model == "TimeXer":
            patch_len = configs.patch_len
            if patch_len>configs.pred_len:
                patch_len = configs.pred_len
            text_top_k = int((configs.seq_len - configs.patch_len) / configs.patch_len + 2)
        elif configs.model == "TimesNet":
            text_top_k = configs.pred_len
        else:
            assert(0)
        if self.model == "PatchTST" or self.model == "TimeXer":
            text_top_k *= configs.enc_in
        self.cross = CrossRanker(configs.d_model, attention_dropout=configs.dropout, k=text_top_k)
        

        # self.proj_ts = nn.Linear(configs.d_model, configs.c_out, bias=True)
        # self.proj_text = nn.Linear(configs.d_model, configs.c_out, bias=True)
        # self.projection_1 = nn.Linear(configs.d_model, configs.c_out, bias=True)
        # self.projection_2 = nn.Linear(configs.text_top_k, configs.seq_len, bias=True)

    def forward(self, x_enc, x_mark_enc, text):
        return None

