import os
import torch
import torch.nn as nn
import numpy as np
from nltk import sent_tokenize

from layers.MyLayers import MyCrossAttentionLayer, MyHead, MixerLayer, FeedForward
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, CrossAttentionLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, PatchEmbedding
from models import ToolModel
from models.PatchTST import Transpose, FlattenHead
from mixture_of_experts import MoE, HeirarchicalMoE
from exp.exp_model_dict import model_dict
from models import SwitchTransformer 


def get_first_return_value(*args):
    if isinstance(args[0], (list, tuple)):
        return args[0][0]
    return args[0] 


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = configs.device

        if configs.patch_len > configs.pred_len:
            configs.patch_len = configs.pred_len
            configs.stride = configs.patch_len // 2

        if configs.use_pretrained_ts_model:
            print('loading pretrained 1st stage model')
            self.stage_1_model = torch.load(os.path.join(configs.pretrain_save_path, 'stage_1_model.pth'))
            for param in self.stage_1_model.parameters():
                param.requires_grad = False
        else:
            print('using tool_model!!')
            self.stage_1_model = ToolModel.Model(configs).float()

        self.model_name = configs.model
        self.ts_model = model_dict[configs.model].Model(configs).float()
        self.enc_embedding = self.ts_model.enc_embedding

        self.encoder = self.ts_model.encoder

        if hasattr(self.ts_model, 'head'):
            self.head = self.ts_model.head
        else:
            self.head = None 

        if hasattr(self.ts_model, 'decoder'):
            self.decoder = self.ts_model.decoder
        else:
            self.decoder = None  
        

        if hasattr(self.ts_model, 'dec_embedding'):
            self.dec_embedding = self.ts_model.dec_embedding
        else:
            self.dec_embedding = None  

        # self.my_head = MyHead(configs.text_top_k, configs.enc_in, configs.d_model, configs.pred_len, head_dropout=configs.dropout)
        self.use_ts_moe = self.configs.use_ts_moe
        self.use_tx_moe = self.configs.use_tx_moe
        if self.use_tx_moe == True:
            tx_dim = self.configs.llm_dim
            num_experts= self.configs.num_tx_experts# Define the number of experts for each MoE layer
            # self.tx_moe = HeirarchicalMoE(dim=tx_dim, num_experts=(num_experts, num_experts)).to(self.device)
            # self.tx_moe = MoE(
            #         dim = tx_dim,
            #         num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
            #         hidden_dim = tx_dim * 4,           # size of hidden dimension in each expert, defaults to 4 * dimension
            #         activation = nn.ReLU,      # use your preferred activation, will default to GELU
            #         second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
            #         second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
            #         second_threshold_train = 0.2,
            #         second_threshold_eval = 0.2,
            #         capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            #         capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            #         loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
            #     ).to(self.device)
            # self.tx_dim_out = tx_dim
            self.tx_dim_out = configs.d_model
            self.tx_moe = SwitchTransformer.SwitchFeedForward(capacity_factor=configs.capacity_factor,
                                                drop_tokens=configs.drop_tokens,
                                                is_scale_prob=configs.is_scale_prob,
                                                n_experts=configs.num_tx_experts,
                                                # expert=FeedForward(tx_dim, tx_dim * 4, configs.dropout),
                                                expert=FeedForward(tx_dim, self.tx_dim_out, configs.dropout),
                                                d_in=tx_dim,
                                                d_out=self.tx_dim_out,
                                                output_routing_distribution = configs.output_routing)
            
            base_config = {
                'capacity_factor': configs.capacity_factor,
                'drop_tokens': configs.drop_tokens,
                'is_scale_prob': configs.is_scale_prob,
                'n_experts': configs.num_tx_experts,
                # 'expert': FeedForward(tx_dim, self.tx_dim_out, configs.dropout),
                'output_routing_distribution': configs.output_routing
            }

            # Create 3-layer MoE with uniform configuration
            layer_configs = SwitchTransformer.LayerConfigBuilder.create_uniform_config(
                num_layers=configs.num_tx_moe_layers,
                base_config=base_config,
                d_in=tx_dim,
                d_model=self.tx_dim_out,
                drop_out=configs.dropout
            )

            self.tx_moe = SwitchTransformer.MultiLayerSwitchFeedForwardWithResidual(
                layer_configs=layer_configs,
                d_model=tx_dim
            )

            print("MoE added! with tx moe(dim={},num experts={})!!!!!!!!!".format(tx_dim, num_experts))
            self.moe_tx_dropout = nn.Dropout(p=0.2)# Add dropout layer with 50% probability
        if self.use_ts_moe == True:
            ts_dim = self.configs.d_model
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
            print("MoE added! with ts-moe(dim={},num experts={}) !!!!!!!!!".format(ts_dim,num_experts))
            self.moe_ts_dropout = nn.Dropout(p=0.2)# Add dropout layer with 50% probability
        
        self.mix_type = configs.mix_type

        if self.mix_type == 2 and configs.use_k_means_init:
            normalized_path = os.path.normpath(configs.root_path)
            dataset = os.path.basename(normalized_path)
            # print(normalized_path)
            # print(dataset)
            cache_filename = f"./data/{dataset}/init_embeddings_{configs.model_id}_{configs.data_path}.npy"
            # print(cache_filename)
    
            # 检查缓存是否存在
            if os.path.exists(cache_filename):
                print(f"Loading cached embeddings from {cache_filename}")
                configs.cluster_init_text_data = np.load(cache_filename)
            else:
                """
                处理长文本，为每个token生成嵌入
                返回形状为 [总token数, 隐藏层维度] 的嵌入矩阵
                """
                # 1. 分句处理
                sentences = sent_tokenize(configs.cluster_init_text_data)

                # 2. 存储所有token嵌入和原始token
                all_token_embeddings = []
                token_positions = []  # 记录每个token的原始位置

                for sent_idx, sentence in enumerate(sentences):
                    # 3. 对每个句子进行tokenization
                    inputs = self.stage_1_model.tokenizer(
                        sentence, 
                        return_tensors='pt',
                        truncation=True,
                        add_special_tokens=False  # 关键：不添加[CLS]/[SEP]
                    )
                    # (text, return_tensors="pt", padding=True, truncation=True, max_length=512)

                    # 4. 获取每个token的嵌入
                    with torch.no_grad():
                        outputs = self.stage_1_model.text_encoder(**inputs.to(self.device))
                        token_embeddings = outputs.last_hidden_state.squeeze(0).to("cpu").numpy()

                    # 5. 存储结果 (每个token独立嵌入)
                    all_token_embeddings.append(token_embeddings)

                    # 记录token位置信息 (可选)
                    for token_idx in range(token_embeddings.shape[0]):
                        token_positions.append({
                            "sentence_idx": sent_idx,
                            "token_idx": token_idx,
                            "token": self.stage_1_model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][token_idx].item())
                        })

                # 6. 拼接所有token嵌入
                configs.cluster_init_text_data = np.vstack(all_token_embeddings)

                np.save(cache_filename, configs.cluster_init_text_data)
                print(f"Saved new embeddings to {cache_filename}")

            # return full_embedding_matrix, token_positions
        
        self.mixer = MixerLayer(configs)


    def text_encoder(self, text, ts_embedding):
        # text = [t for t in text]
        text = [f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>" for text_info in text]
        token = self.stage_1_model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = self.stage_1_model.tokenizer.convert_ids_to_tokens(token['input_ids'][0])
        # text_embedding = self.llm_model.get_input_embeddings()(token.to(self.device))  # (batch, token, dim)
        text_embedding = self.stage_1_model.text_encoder(**token.to(self.device)).last_hidden_state
        aux_loss_tx = 0
        scores_avg_saved = None
        # print(text_embedding.shape)
        # print("nn")

        # print(f"text_embedding.shape:{text_embedding.shape}")
        if self.use_tx_moe and self.configs.mix_type != 2:
            text_embedding, aux_loss_tx = self.tx_moe(text_embedding)
            text_embedding = self.moe_tx_dropout(text_embedding)
        if text_embedding.shape[-1] == self.configs.llm_dim and self.configs.mix_type != 2:  
            text_embedding = self.stage_1_model.text_mlp(text_embedding)
        if self.configs.use_Cross_ranker:
            text_embedding, scores_avg_saved = self.stage_1_model.cross(ts_embedding, text_embedding)
        ret = {
            "text_embedding" : text_embedding,
            "tokens" : tokens,
            "all_special_tokens" : self.stage_1_model.tokenizer.all_special_tokens
        }
        if aux_loss_tx != 0: 
            ret["aux_loss_tx"] = aux_loss_tx

        if scores_avg_saved is not None: 
            ret["scores_avg_saved"] = scores_avg_saved
        # return text_embedding, aux_loss_tx, scores_avg_saved, tokens, self.stage_1_model.tokenizer.all_special_tokens
        return ret

    def data_manage_enc_out(self, enc_out, n_vars, pred_len):
        if self.model_name == "PatchTST" or self.model_name == "TimeXer":
            # z: [bs x nvars x patch_num x d_model]
            enc_out = torch.reshape(
                enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
            # z: [bs x nvars x d_model x patch_num]
            batch_size, nvars, patch_num, d_model = enc_out.shape
            enc_out = enc_out.reshape(batch_size, -1, d_model)
        elif self.model_name == "TimeLLM":
            batch_size, nvars, d_model, patch_num = enc_out.shape
            enc_out = enc_out.permute(0, 1, 3, 2).reshape(batch_size, -1, d_model)
        elif self.model_name == "TimesNet":
            enc_out = enc_out[:,-pred_len:,:]
            
        return enc_out
    
    def data_manage_mixer_out(self, x, nvars):
        if self.model_name == "PatchTST" or self.model_name == "TimeXer" or self.model_name == "TimeLLM":
            x = x.reshape(x.shape[0], nvars, -1, x.shape[-1])
            x = x.permute(0, 1, 3, 2)  # [bs x nvars x d_model x patch_num]

        return x
    
    def data_manage_head_out(self, head_out, N):
        if self.model_name == "PatchTST" or self.model_name == "TimeXer":
            head_out = head_out.permute(0, 2, 1)
        elif self.model_name == "iTransformer":
            head_out = head_out.permute(0, 2, 1)[:, :, :N]
        return head_out
    
    def plot_t_SNE(self):
        self.mixer.plot_t_SNE()


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text, batch_idx):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # B L C
        _, N, n_vars = x_enc.shape
        _, N_mark, _ = x_mark_enc.shape

        # u: [bs * nvars x patch_num x d_model]
        if self.enc_embedding is not None:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        else:
            enc_out = x_enc

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        
        enc_out, attns = self.encoder(enc_out, x_mark_enc)

        # print(enc_out.shape)

        aux_loss = 0
        if self.use_ts_moe:
            enc_out, aux_loss_ts = self.ts_moe(enc_out)
            enc_out = self.moe_ts_dropout(enc_out)
            aux_loss += aux_loss_ts

        enc_out = self.data_manage_enc_out(enc_out, n_vars, self.pred_len)

        # Text Embedding
        # text_embedding, aux_loss_tx, scores_avg_saved, tokens, special_tokens = self.text_encoder(text, enc_out)
        ret_dict = self.text_encoder(text, enc_out)
        text_embedding = ret_dict["text_embedding"]

        aux_loss_tx = ret_dict.get("aux_loss_tx", 0)
        scores_avg_saved = ret_dict.get("scores_avg_saved", None)
        tokens = ret_dict["tokens"]
        special_tokens = ret_dict["all_special_tokens"]

        aux_loss += aux_loss_tx
        # Fusion
        # print(f"enc_out.shape{enc_out.shape}")
        # print(f"text_embedding.shape{text_embedding.shape}")
        enc_out, aux_loss_ = self.mixer(enc_out, text_embedding, batch_idx) 
        aux_loss += aux_loss_ 


        enc_out = self.data_manage_mixer_out(enc_out, n_vars)

        # print(enc_out.shape)

        if self.dec_embedding is not None:
            dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # Decoder
        if self.decoder is not None:
            dec_out = self.decoder(dec_out, enc_out)
        elif self.head is not None:
            dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        else:
            assert(0)

        dec_out = self.data_manage_head_out(dec_out, n_vars)

        # De-Normalization from Non-stationary Transformer
        # print(dec_out.shape)
        # print(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1).shape)
        if self.model_name != "Informer":
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        ret_dict = {
            "outputs" : dec_out,
            "prompt_emb" : None,
            "aux_loss" : aux_loss,
            "scores_avg_saved" : scores_avg_saved,
            "tokens" : tokens,
            "special_tokens" : special_tokens
        }
        return ret_dict

    def forward(self, inputs, batch_idx = -1):
        x_enc, x_mark_enc, x_dec, x_mark_dec, text = inputs
        if self.task_name == 'long_term_forecast':
            ret_dict = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text, batch_idx)
            return ret_dict  # [B, L, D]
        return None
