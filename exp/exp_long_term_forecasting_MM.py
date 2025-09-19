from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer,LlamaForCausalLM
import datetime
from datetime import datetime, timedelta
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import pandas as pd
from datetime import datetime
from collections import namedtuple
from collections import Counter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def safe_unpack(data: dict):
    """安全解包：返回命名元组"""
    return namedtuple('UnpackedData', data.keys())(**data)


def get_top_frequent_words(ranked_texts, top_n=10):

    all_indices = []
    
    for batch_keys in ranked_texts:
        if isinstance(batch_keys, torch.Tensor):
            batch_keys = batch_keys.cpu().numpy().flatten().tolist()
        else:
            batch_keys = np.array(batch_keys).flatten().tolist()
        
        all_indices.extend(batch_keys)
    print(all_indices[0])
    print(len(all_indices))
    word_counter = Counter(all_indices)

    most_common = word_counter.most_common(top_n)
    
    return most_common

def norm(input_emb):
    if input_emb is None:
        return 0
    input_emb=input_emb- input_emb.mean(1, keepdim=True).detach()
    input_emb=input_emb/torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return input_emb

class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)  
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  
                x = F.relu(x)
                x = self.dropout(x)  
        return x
warnings.filterwarnings('ignore')

class DynamicAuxLossWeight:
    def __init__(self, alpha=0.1, min_weight=0.001, max_weight=0.2):
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def __call__(self, main_loss, aux_loss):
        weight = self.alpha * (main_loss / (aux_loss + 1e-8))
        return torch.clamp(weight, self.min_weight, self.max_weight)

class Modal_eval(nn.Module):
    def __init__(self, tpye = 1):
        super(Modal_eval, self).__init__()

    def forward(self, x1, x2):
        # # x1_norm = F.normalize(x1, p=2, dim=1)  # L2 归一化
        # # x2_norm = F.normalize(x2, p=2, dim=1)
        # similarity = (x1 * x2).sum(dim=1)  # 点积
        # # 计算相似度 (逐元素相乘)
        # similarity = x1 * x2  # [B, L]

        # # 归一化相似度，确保权重和为 1
        # weight_x1 = similarity.sum(dim=1, keepdim=True) + 1e-8  # 避免除零
        # weight_x2 = 1 - weight_x1  # 确保权重之和为 1

        # weighted_sum = weight_x1 * x1 + weight_x2 * x2  # [B, L]
        if type ==2:
            cosine_similarity = torch.cosine_similarity(x1, x2, dim=-1).unsqueeze(-1)
            x1 = x2 * cosine_similarity + (1 - cosine_similarity) * x1


        return x1  


class Exp_Long_Term_Forecast_MM(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_MM, self).__init__(args)
        configs=args
        self.plot_tsne = configs.plot_tsne

        self.text_path=configs.text_path
        self.prompt_weight=configs.prompt_weight
        self.attribute="final_sum"
        self.type_tag=configs.type_tag
        self.text_len=configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len=configs.pred_len
        self.pool_type=configs.pool_type
        self.use_fullmodel=configs.use_fullmodel
        self.hug_token=configs.huggingface_token
        if configs.llm_model == 'Doc2Vec':
            print('Now using Doc2Vec')
            print("Training Doc2Vec model")

            from gensim.test.utils import common_texts
            from gensim.test.utils import common_texts
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
            def read_csv_column(file_path, column_name):
                df = pd.read_csv(file_path)
                
                column_data = df[column_name].replace('', np.nan).fillna('null')
                
                return column_data.to_list()
            result  = read_csv_column(file_path=os.path.join(configs.root_path,
                                          configs.data_path), column_name='Final_Search_4')
            train_len=int(len(result)*0.8)
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(result[:train_len])]
            text_model = Doc2Vec(documents, vector_size=configs.llm_dim, window=2, min_count=1, workers=4)
            self.text_model=text_model
            self.Doc2Vec=True
        else:
            if configs.llm_model == 'LLAMA2':
                # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
                self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
                self.llama_config.num_hidden_layers = configs.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'LLAMA3':
                # Automatically load the configuration, model, and tokenizer for LLaMA-3-8B
                llama3_path = "meta-llama/Meta-Llama-3-8B-Instruct"
                cache_path = "./"

                # Load the configuration with custom adjustments
                self.config =  LlamaConfig.from_pretrained(llama3_path,token=self.hug_token,cache_dir=cache_path)

                self.config.num_hidden_layers = configs.llm_layers
                self.config.output_attentions = True
                self.config.output_hidden_states = True

                self.llm_model  = LlamaModel.from_pretrained(
                    llama3_path,
                    config=self.config,
                    token=self.hug_token,cache_dir=cache_path
                )
                self.tokenizer = AutoTokenizer.from_pretrained(llama3_path,use_auth_token=self.hug_token,cache_dir=cache_path)
            elif configs.llm_model == 'GPT2':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2M':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-medium')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2L':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-large')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2XL':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-xl')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'BERT':
                bert_base_uncased_tokenizer_path = './BERT_files/bert-base-uncased-tokenizer'
                bert_base_uncased_encoder_path = './BERT_files/bert-base-uncased-text_encoder'
                bert_config_path = './BERT_files/local_bert_config'
                if os.path.exists(bert_config_path):
                    self.bert_config = BertConfig.from_pretrained(bert_config_path)
                else:
                    self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                try:
                    self.llm_model = BertModel.from_pretrained(
                        bert_base_uncased_encoder_path,
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.bert_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                    )
                    self.text_encoder.save_pretrained(bert_base_uncased_encoder_path)

                try:
                    self.tokenizer = BertTokenizer.from_pretrained(
                        bert_base_uncased_tokenizer_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False
                    )
                    self.tokenizer.save_pretrained(bert_base_uncased_tokenizer_path)
            
            else:
                raise Exception('LLM model is not defined')

            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model=self.llm_model.to(self.device)
        if args.init_method == 'uniform':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.uniform_(self.weight1.weight)
            nn.init.uniform_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        elif args.init_method == 'normal':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.normal_(self.weight1.weight)
            nn.init.normal_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        else:
            raise ValueError('Unsupported initialization method')
        
        
        self.learning_rate2=1e-2
        self.learning_rate3=1e-3

    def _build_model(self):
        if self.args.use_Cross_MoE:
            if self.args.mix_type == 2:
                if self.args.use_k_means_init:
                    self.args.cluster_init_text_data = self.data_set_dict["train"].get_full_text()
                self.args.d_model = self.args.llm_dim
            model = self.model_dict['Cross_MoE'].Model(self.args).float()
        elif self.args.use_Unified_model:
            model = self.model_dict['MoE_TS_Text_fuser'].Model(self.args).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if self.args.use_dist_training:
            model = DDP(model, device_ids=[self.args.rank])
            
        return model

    def _get_data(self, flag):
        # data_set, data_loader = data_provider(self.args, flag)
        # return data_set, data_loader
        return self.data_set_dict[flag], self.data_loader_dict[flag]

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_optimizer_weight(self):
        model_optim = optim.Adam([{'params': self.weight1.parameters()},
                            {'params': self.weight2.parameters()}], lr=self.args.learning_rate_weight)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
        

    def vali(self, vali_data, vali_loader, criterion):
        self.args.is_testing = 0
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.args.features == "S":
                    prior_y=torch.from_numpy(vali_data.get_prior_y(index)).float().to(self.device)
                else:
                    prior_y = 0

                batch_text=vali_data.get_text(index)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        ret_dict = self.model((batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_text))
                else:
                    ret_dict = self.model((batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_text))
                vars = safe_unpack(ret_dict)
                outputs, prompt_emb, aux_loss = vars.outputs, vars.prompt_emb, vars.aux_loss
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                dynamic_weight = DynamicAuxLossWeight()

                prompt_y = outputs

                if self.args.use_text:
                    prompt_y = norm(prompt_emb) + prior_y

                if self.args.use_eva:
                    outputs = self.prompt_weight_method(outputs, prompt_y)
                else:
                    outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y

                
                loss_ts = criterion(outputs, batch_y)
                loss = loss_ts
                if self.args.use_text and self.args.use_tx_moe or self.args.use_ts_moe:
                    loss += dynamic_weight(loss_ts, aux_loss).detach() * aux_loss
                
                # pred = outputs.detach().cpu()
                # true = batch_y.detach().cpu()

                # loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting):
        self.args.is_testing = 0
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        criterion = self._select_criterion()



        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            if self.args.use_dist_training:
                self.args.sampler.set_epoch(epoch)  
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                #0523
                if self.args.features == "S":
                    prior_y=torch.from_numpy(train_data.get_prior_y(index)).float().to(self.device)
                else:
                    prior_y = 0
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input 给定一定的前文提示，当做解码器的输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                batch_text=train_data.get_text(index)
                # encoder - decoder
                if self.args.calculate_overhead:
                    # flops, params = profile(self.model, inputs=((batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_text)))
                    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
                    # print('Params = ' + str(params / 1000 ** 2) + 'M')
                    # print(stat(self.model, (batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_text)))
                    from calculate_overhead import Overhead
                    Overhead(self.args)
                    return
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        ret_dict = self.model((batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_text))
                else:
                    ret_dict = self.model((batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_text))
                vars = safe_unpack(ret_dict)
                outputs, prompt_emb, aux_loss = vars.outputs, vars.prompt_emb, vars.aux_loss
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                dynamic_weight = DynamicAuxLossWeight()

                prompt_y = outputs

                if self.args.use_text:
                    prompt_y = norm(prompt_emb) + prior_y

                if self.args.use_eva:
                    outputs = self.prompt_weight_method(outputs, prompt_y)
                else:
                    outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
                
                loss_ts = criterion(outputs, batch_y)
                loss = loss_ts
                if self.args.use_text and self.args.use_tx_moe or self.args.use_ts_moe:
                    loss += dynamic_weight(loss_ts, aux_loss).detach() * aux_loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        self.args.is_testing = 1
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        ranked_texts = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(test_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.features == "S":
                    prior_y=torch.from_numpy(test_data.get_prior_y(index)).float().to(self.device)
                else:
                    prior_y = 0
                #input_start_dates,input_end_dates=test_data.get_date(index)
                batch_text=test_data.get_text(index)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        ret_dict = self.model((batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_text), i)
                else:
                    ret_dict = self.model((batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_text), i)
                vars = safe_unpack(ret_dict)
                outputs, prompt_emb, scores_avg_saved = vars.outputs, vars.prompt_emb, vars.scores_avg_saved
                tokens = vars.tokens
                special_tokens = vars.special_tokens

                ranked_text = []
                if scores_avg_saved is not None:
                    topk_scores, topk_indices = torch.topk(scores_avg_saved, k=10, dim=-1, sorted=True)

                    # ranked_text = torch.gather(batch_text, dim=1, index=topk_indices)
                    # print(batch_text.shape)
                    # 遍历batch中的每个样本
                    for j in range(len(batch_text)):
                        sample_indices = topk_indices[j].cpu().numpy()

                        valid_indices = [idx for idx in sample_indices 
                                if idx < len(tokens) 
                                and tokens[idx] not in special_tokens]

                        # 使用索引选择最重要的k个单词
                        selected_tokens = [tokens[idx] for idx in valid_indices]

                        # 添加到当前batch的单词列表
                        ranked_text.extend(selected_tokens)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                prompt_y = outputs
                if self.args.use_text:
                    prompt_y = norm(prompt_emb) + prior_y

                if self.args.use_eva:
                    outputs = self.prompt_weight_method(outputs, prompt_y)
                else:
                    outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y

                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                ranked_texts.append(ranked_text)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if len(preds[-1]) != len(preds[0]):  # 以第一个为基准
            preds = preds[:-1]  # 去掉最后一个
            trues = trues[:-1]  # 同步去掉
            ranked_texts = ranked_texts[:-1]  


        preds = np.array(preds)
        trues = np.array(trues)

        if len(ranked_texts[0]) != 0:
            top_words = get_top_frequent_words(ranked_texts, top_n=10)

            formatted_words = " ".join([f"{i+1}. {word}" for i, (word, count) in enumerate(top_words)])

            with open(folder_path + f'ranked_words_top_10', "w") as f:
                f.write(formatted_words)
        # print("出现频率最高的单词:")
        # for idx, count in top_words:
        #     print(f"单词索引 {idx}: 出现 {count} 次")

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        
        dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        if self.plot_tsne:
            self.model.plot_t_SNE()

        return mse
