import os

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import BertTokenizer, BertModel, BertConfig

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from mixture_of_experts import MoE, HeirarchicalMoE
from models.MoE_Attention import MoEClusteredAttention
from visualization.attn_heat_map import AttentionHeatmapVisualizer

def softmax_np(x, axis=-1):
    """稳定的NumPy Softmax函数"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class FeedForward(nn.Module):
    """
    ## FFN module
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        # self.layer1 = nn.Linear(d_model, 4*d_model, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        # self.layer2 = nn.Linear(4*d_model, d_ff, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$
        # depending on whether it is gated
        # x = self.layer2(x)
        return x


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class CrossRanker(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1, k=24):
        super(CrossRanker, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.k = k

    def forward(self, queries, keys):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        scale = 1. / sqrt(D)

        ori_keys = keys

        scores = torch.einsum("bld,bsd->bls", queries, keys)
        scores = self.dropout(torch.softmax(scale * scores, dim=-1))  # (B, L, S)
        scores_avg = torch.mean(scores, dim=1)  # (B, S)
        scores_avg_saved = scores_avg
        # scores_avg = torch.softmax(scores_avg, dim=-1)
        # topk_scores, topk_indices = torch.topk(scores_avg, k=self.k, dim=-1, sorted=False)  # (B, L)
        topk_scores, topk_indices = torch.topk(scores_avg, k=self.k, dim=-1, sorted=True)  # (B, L) , assert(L == k)
        topk_scores = torch.softmax(topk_scores, dim=-1)

        batch_indices = torch.arange(B).view(-1, 1).expand(-1, self.k)  # (B, k)

        if ori_keys.is_cuda:
            batch_indices = batch_indices.to(ori_keys.device)

        selected_keys = ori_keys[batch_indices, topk_indices]
        
        return torch.einsum("bld,bl->bld", selected_keys, topk_scores), scores_avg_saved  # (B, k, D)


class MyCrossAttentionLayer(nn.Module):
    def __init__(self, cross_attention_1, cross_attention_2, d_model, d_ff=None,
                dropout=0.1, activation="relu"):
        super(MyCrossAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention_1 = cross_attention_1
        self.cross_attention_2 = cross_attention_2
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, ts, text, x_mask=None, cross_mask=None, tau=None, delta=None):
        x_0 , attn = self.cross_attention_1(ts, text, text)
        x = ts + self.dropout(x_0)
        # x = self.dropout(self.cross_attention_1(ts, text, text)[0])
        # x = x + self.dropout(self.cross_attention_2(x, x, x)[0])
        # x = x + x_2
        # return self.norm2(x)
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # return self.norm3(x + y)
        return x+y, attn


class MyHead(nn.Module):
    def __init__(self, k, n_vars, d_model, target_window, head_dropout=0):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, target_window)
        self.linear_2 = nn.Linear(k, n_vars)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x seq_len x d_model]
        x = self.linear_1(x)
        x = x.permute(0, 2, 1)
        x = self.linear_2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x


class MixerLayer(nn.Module):
    def __init__(self, configs):
        super(MixerLayer, self).__init__()
        self.configs = configs
        self.mix_type = configs.mix_type
        self.model = configs.model
        
        if self.mix_type == 1:
            self.my_cross = MyCrossAttentionLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.plot_attn),
                    configs.d_model, configs.n_heads),
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.plot_attn),
                    configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
        elif self.mix_type == 2 :
            init_text = configs.cluster_init_text_data
            self.moe_enhanced_cross = MoEClusteredAttention(configs, configs.d_model, configs.num_tx_experts, 0.01,
                                                             init_text, use_trainable_center = configs.use_trainable_center)
    
    def plot_t_SNE(self):
        assert(self.mix_type == 2)
        self.moe_enhanced_cross.plot_t_SNE()

    def forward(self, ts, text, batch_idx):
        aux_loss = 0
        if self.mix_type == 1:
            
            x, attention_weights = self.my_cross(ts, text, batch_idx)
            
        
            if self.configs.plot_attn and self.configs.is_testing:
                folder_path = './attn_results/Native_attn_1/'
                folder_path  += f"num_centers:{self.configs.num_tx_experts}/"

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                base_filepath = os.path.join(folder_path, f"{self.configs.model_id}_b_out_idx:{batch_idx}")
                
                # 步骤 1: 转换并保存【原始】注意力权重 (保持不变)
                attention_weights_np = attention_weights.detach().cpu().numpy()
                np.save(base_filepath + '_attention_weights.npy', attention_weights_np)
                print(f"Saved original attention_weights to {base_filepath}_attention_weights.npy")

                # --- ✨ 新增：为绘图准备“逐行Top-K + Softmax”数据 ✨ ---
                
                # a. 设置K值
                K = 20
                
                # b. 将多维注意力权重展平为2D矩阵 (N, M)
                key_len = attention_weights_np.shape[-1]
                attn_2d = attention_weights_np.reshape(-1, key_len)
                
                # c. 逐行找出Top-K值的索引
                # np.argsort(...) 返回的是从小到大排序的索引，所以我们取最后K个
                topk_indices = np.argsort(attn_2d, axis=1)[:, -K:]
                
                # d. 创建一个稀疏矩阵，只保留Top-K位置的原始值
                # np.take_along_axis 从attn_2d中根据topk_indices提取出实际的Top-K值
                topk_values = np.take_along_axis(attn_2d, topk_indices, axis=1)
                
                # 创建一个填充了-inf的矩阵
                filtered_attn = np.full_like(attn_2d, -np.inf)
                # 使用 np.put_along_axis 将Top-K值放回原位
                np.put_along_axis(filtered_attn, topk_indices, topk_values, axis=1)

                # e. 对这个新的稀疏矩阵，逐行重新计算Softmax
                plot_attn_data = softmax_np(filtered_attn, axis=1)
                
                # --- ✨ 新增代码结束 ✨ ---

                # 步骤 2: 使用【处理后】的数据进行绘图
                attn_visualizer = AttentionHeatmapVisualizer(self.configs)
                
                # 确保plot_full_path有唯一的图片文件名后缀
                plot_full_path = base_filepath + f"_top{K}_softmax_heatmap"
                
                
                fig1 = attn_visualizer.plot_attention(
                            plot_attn_data,  # <--- 使用我们新计算的数据
                            title=f"Row-wise Top-{K} Softmax Attention ({plot_attn_data.shape[0]}x{plot_attn_data.shape[1]})", # <--- 更新标题
                            colormap='cool',
                            save_path=plot_full_path,
                            show_values=False, # 对于处理后的大图，强烈建议设为False
                            grid=True,
                )
                plt.close(fig1) # 确保关闭图像

        elif self.mix_type == 0:
            # print(ts.shape)
            # print(text.shape)
            x = ts + text
        else:
            x, aux_loss= self.moe_enhanced_cross(ts,text,text, batch_idx)
        # x = self.self(ts, x)
        
        return x, aux_loss


class MyBert(nn.Module):
    def __init__(self, llm_layers, device):
        super().__init__()
        self.device = device
        bert_config_path = './BERT_files/local_bert_config'
        if os.path.exists(bert_config_path):
            self.bert_config = BertConfig.from_pretrained(bert_config_path)
        else:
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.save_pretrained(bert_config_path)
        self.bert_config.num_hidden_layers = llm_layers
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

    def forward(self, text):
        text = [t for t in text]
        token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # text_embedding = self.llm_model.get_input_embeddings()(token.to(self.device))  # (batch, token, dim)
        return self.text_encoder(**token.to(self.device)).last_hidden_state  # (batch, token, 768)

    def forward_2(self, text):
        token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return self.text_encoder(**token.to(self.device)).last_hidden_state  # (batch, token, 768)
