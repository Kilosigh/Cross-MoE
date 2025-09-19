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

        print('loading pretrained time series model')
        self.ts_model = torch.load(os.path.join(configs.pretrain_save_path, 'ts_model.pth'))
        for param in self.ts_model.parameters():
            param.requires_grad = False

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

        self.text_mlp = MLP([configs.llm_dim, int(configs.llm_dim / 8), configs.d_model], dropout_rate=0.1)
        self.cross = CrossRanker(configs.d_model, attention_dropout=configs.dropout, k=configs.text_top_k)

        self.proj_ts = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.proj_text = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.projection_1 = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.projection_2 = nn.Linear(configs.text_top_k, configs.seq_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, text):
        # TS Embedding
        # ts_embedding = self.ts_model.enc_embedding(x_enc, x_mark_enc)
        # ts_embedding, attns = self.ts_model.encoder(ts_embedding, attn_mask=None)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x_enc, n_vars = self.ts_model.patch_embedding(x_enc)
        # BCND
        x_enc, attns = self.ts_model.encoder(x_enc)
        x_enc = torch.reshape(
        x_enc, (-1, n_vars, x_enc.shape[-2], x_enc.shape[-1]))
         # BND
        ts_embedding = x_enc.mean(1)

        # Text Embedding
        text = [t for t in text]
        token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # text_embedding = self.llm_model.get_input_embeddings()(token.to(self.device))  # (batch, token, dim)
        text_embedding = self.text_encoder(**token.to(self.device)).last_hidden_state

        # Text MLP
        text_embedding = self.text_mlp(text_embedding)

        # Cross Ranker
        text_embedding = self.cross(ts_embedding, text_embedding)

        # Concatenate
        # TODO: 改成投影后相加
        # fusion_embedding = torch.cat([ts_embedding, text_embedding], dim=-1)
        # ts_embedding = self.proj_ts(ts_embedding)
        # text_embedding = self.proj_text(text_embedding)
        # fusion_embedding = ts_embedding + text_embedding

        # Projection
        dec_out = self.projection_1(text_embedding)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = self.projection_2(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        # dec_out = ts_embedding + text_embedding

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        pass

    def anomaly_detection(self, x_enc):
        pass

    def classification(self, x_enc, x_mark_enc):
        pass

    def forward(self, x_enc, x_mark_enc, text):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, text)
            return dec_out  # [B, L, D]
        return None

