import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from layers.MyLayers import MyCrossAttentionLayer, MyHead, MixerLayer
# from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, CrossAttentionLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, PatchEmbedding
# from models import MyModel_1
# from models.PatchTST import Transpose, FlattenHead

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from mixture_of_experts import MoE, HeirarchicalMoE
from exp.exp_model_dict import model_dict


def norm(input_emb):
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

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = configs.device
        self.ts_model = model_dict[self.configs.model].Model(self.configs).float()
        if self.configs.use_multi_gpu and self.configs.use_gpu:
            self.ts_model = nn.DataParallel(self.ts_model, device_ids=self.configs.device_ids)
        
        self.pool_type=configs.pool_type
        self.Doc2Vec=False
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

        self.use_fullmodel = configs.use_fullmodel

        mlp_sizes=[configs.llm_dim, int(configs.llm_dim/8),configs.text_emb]
        self.mlp = MLP(mlp_sizes,dropout_rate=0.3)

        self.use_ts_moe = self.configs.use_ts_moe
        self.use_tx_moe = self.configs.use_tx_moe
        if self.use_tx_moe == True:
            tx_dim = self.configs.llm_dim
            num_experts= self.configs.num_tx_experts# Define the number of experts for each MoE layer
            # self.tx_moe = HeirarchicalMoE(dim=tx_dim, num_experts=(num_experts, num_experts)).to(self.device)
            self.tx_moe = MoE(
                    dim = tx_dim,
                    num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
                    hidden_dim = tx_dim * 4,           # size of hidden dimension in each expert, defaults to 4 * dimension
                    activation = nn.ReLU,      # use your preferred activation, will default to GELU
                    second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
                    second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
                    second_threshold_train = 0.2,
                    second_threshold_eval = 0.2,
                    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                    loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
                ).to(self.device)

            print("HeirarchicalMoE added! with tx moe(dim={},num experts={})!!!!!!!!!".format(tx_dim, num_experts))
            self.moe_tx_dropout = nn.Dropout(p=0.2)# Add dropout layer with 50% probability
        if self.use_ts_moe == True:
            ts_dim = self.configs.c_out
            num_experts = self.configs.num_ts_experts # Define the number of experts for each MoE layer
            # self.ts_moe = HeirarchicalMoE(dim=ts_dim, num_experts=(num_experts, num_experts)).to(self.device)
            self.ts_moe = MoE(
                    dim = ts_dim,
                    num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
                    hidden_dim = ts_dim * 4,           # size of hidden dimension in each expert, defaults to 4 * dimension
                    activation = nn.ReLU,      # use your preferred activation, will default to GELU
                    second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
                    second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
                    second_threshold_train = 0.2,
                    second_threshold_eval = 0.2,
                    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                    loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
                ).to(self.device)
            print("HeirarchicalMoE added! with ts-moe(dim={},num experts={}) !!!!!!!!!".format(ts_dim,num_experts))
            self.moe_ts_dropout = nn.Dropout(p=0.2)# Add dropout layer with 50% probability

        # evaluation method
        if self.configs.use_eva:
            self.prompt_weight_method = Modal_eval(self.configs.use_eva)

        # self.mixer = MixerLayer(configs)

        # Prediction Head
        # self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        # self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
        #                         head_dropout=configs.dropout)

        # self.my_head = MyHead(configs.text_top_k, configs.enc_in, configs.d_model, configs.pred_len, head_dropout=configs.dropout)

    def text_model(self, text, ts_embedding):
        if self.Doc2Vec==False:
            prompt = [f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>" for text_info in text]
            # len(prompt) = B
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.device))  # (batch, prompt_token, dim)
        else:
            prompt = text
            prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(self.device)
        if self.use_fullmodel:
            prompt_emb =self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
        else:
            prompt_emb=prompt_embeddings 
        aux_loss_tx = 0
        if self.use_tx_moe:
            prompt_emb, aux_loss_tx = self.tx_moe(prompt_emb)
            prompt_emb = self.moe_tx_dropout(prompt_emb)
            
        prompt_emb = self.mlp(prompt_emb)  # (batch, prompt_token, pred_len)

        if self.Doc2Vec==False:
            if self.pool_type=="avg":   # B, pred_len, 1     
                global_avg_pool = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                prompt_emb=global_avg_pool.unsqueeze(-1)
            elif self.pool_type=="max":
                global_max_pool = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                prompt_emb=global_max_pool.unsqueeze(-1)
            elif self.pool_type=="min":
                global_min_pool = F.adaptive_max_pool1d(-1.0*prompt_emb.transpose(1, 2), 1).squeeze(2)
                prompt_emb=global_min_pool.unsqueeze(-1)
            elif self.pool_type == "attention":

                outputs_reshaped = ts_embedding#.transpose(1, 2) 
                outputs_norm = F.normalize(outputs_reshaped, p=2, dim=1) 
                prompt_emb_norm = F.normalize(prompt_emb, p=2, dim=2) 
                attention_scores = torch.bmm(prompt_emb_norm, outputs_norm) 
                attention_weights = F.softmax(attention_scores, dim=1) 
                
                weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)  

                prompt_emb = weighted_prompt_emb.unsqueeze(-1) 
        else:
            prompt_emb=prompt_emb.unsqueeze(-1)

        return prompt_emb, aux_loss_tx


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text):
        
        enc_out = self.ts_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.configs.output_attention:
            enc_out = enc_out[0]

        f_dim = -1 if self.configs.features == 'MS' else 0
        ts_out = enc_out[:, -self.configs.pred_len:, f_dim:]
        aux_loss = 0
        aux_loss_ts = 0
        if self.use_ts_moe:
            ts_out, aux_loss_ts = self.ts_moe(ts_out)
            ts_out = self.moe_ts_dropout(ts_out)
        # Text Embedding
        aux_loss += aux_loss_ts
        text_out, aux_loss_tx= self.text_model(text, enc_out)
        aux_loss += aux_loss_tx

        # Fusion
        # enc_out, aux_loss = self.mixer(enc_out, text_out)

        return ts_out, text_out, aux_loss

    def forward(self, inputs):
        x_enc, x_mark_enc, x_dec, x_mark_dec, text = inputs
        if self.task_name == 'long_term_forecast':
            ts_out, text_out, aux_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text)
            return ts_out, text_out, aux_loss  # [B, L, D]
        return None
