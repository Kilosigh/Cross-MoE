import torch
from mixture_of_experts import MoE, HeirarchicalMoE
from layers.MyLayers import MLP, CrossRanker, FeedForward, MixerLayer
import torch.nn as nn
from torchstat import stat
from thop import profile
from models import SwitchTransformer


class Overhead(nn.Module):
    def __init__(self, configs):
        super(Overhead, self).__init__()
        self.model = configs.model
        self.text_mlp = MLP([configs.llm_dim, int(configs.llm_dim / 8), configs.d_model], dropout_rate=0.1)
        text_top_k  = self.get_top_k(configs)
        self.cross = CrossRanker(configs.d_model, attention_dropout=configs.dropout, k=text_top_k).to(configs.device)
        self.tx_moe = SwitchTransformer.SwitchFeedForward(capacity_factor=configs.capacity_factor,
                                                            drop_tokens=configs.drop_tokens,
                                                            is_scale_prob=configs.is_scale_prob,
                                                            n_experts=configs.num_tx_experts,
                                                            # n_experts=1,
                                                            expert=FeedForward(configs.llm_dim, configs.d_model, configs.dropout),
                                                            d_in=configs.llm_dim,
                                                            d_out=configs.d_model,
                                                            output_routing_distribution=0).to(configs.device)
        self.mixer = MixerLayer(configs).to(configs.device)
        mlp_sizes=[configs.llm_dim,int(configs.llm_dim/8),configs.pred_len]
        self.mlp = MLP(mlp_sizes,dropout_rate=0.3).to(configs.device)
        self.k_projection = nn.Linear(configs.llm_dim, configs.d_model).to(configs.device)
        # stat(self.tx_moe, (configs.batch_size, 512, configs.llm_dim))
        # print(stat(self.cross, ))
        T = 512
        text = torch.arange(configs.batch_size*T*configs.llm_dim).reshape(configs.batch_size, T, configs.llm_dim).to(configs.device).float()
        # self.tx_moe(text)
        flops, params = profile(self.tx_moe, inputs=(text,))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 1) + 'K')

        print("K projection")
        
        flops_, params_ = profile(self.k_projection, inputs=(text,))
        print('FLOPs = ' + str(flops_ / 1000 ** 3) + 'G')
        print('Params = ' + str(flops_ / 1000 ** 1) + 'K')

        print("Cross-Ranker")
        T = self.get_top_k(configs)
        ts = torch.arange(configs.batch_size*T*configs.d_model).reshape(configs.batch_size, T, configs.d_model).to(configs.device).float()
        text = torch.arange(configs.batch_size*512*configs.d_model).reshape(configs.batch_size, 512, configs.d_model).to(configs.device).float()
        flops_, params_ = profile(self.cross, inputs=(ts,text))
        flops += flops_
        params += params_
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 1) + 'K')

        T = self.get_top_k(configs)
        ts = torch.arange(configs.batch_size*T*configs.d_model).reshape(configs.batch_size, T, configs.d_model).to(configs.device).float()
        if configs.mix_type != 1:
            text = torch.arange(configs.batch_size*T*configs.d_model).reshape(configs.batch_size, T, configs.d_model).to(configs.device).float()
        else:
            text = torch.arange(configs.batch_size*512*configs.d_model).reshape(configs.batch_size, 512, configs.d_model).to(configs.device).float()
        flops_, params_ = profile(self.mixer, inputs=(ts,text))
        flops += flops_
        params += params_
        if configs.mix_type == 1:
            flops = flops_
            params = params_
        print("Fusion, configs.mix_type:" + str(configs.mix_type))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 1) + 'K')
        # print(stat(self.tx_moe, ))
        # print(stat(self.tx_moe, ))

        # Avg Pooling
        print("Avg Pooling")
        cost = 0 
        T_text = 512
        text = torch.arange(configs.batch_size*T_text*configs.llm_dim).reshape(configs.batch_size, T_text, configs.llm_dim).to(configs.device).float()
        flops, params = profile(self.mlp, inputs=(text,))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 1) + 'K')
    

    def get_top_k(self, configs):
        if configs.model == "PatchTST":
            patch_len = configs.patch_len
            if patch_len>configs.pred_len:
                patch_len = configs.pred_len
            text_top_k = int((configs.seq_len - patch_len) / configs.stride + 2)
        elif configs.model == "iTransformer":
            freq_dict = {'m': 1, 'w': 2, 'd': 3}
            text_top_k = configs.enc_in + freq_dict[configs.freq]
        elif configs.model == "TimeXer":
            text_top_k =  int((configs.seq_len - configs.patch_len) / configs.stride + 1)
        elif configs.model == "TimesNet":
            text_top_k = configs.pred_len
        else:
            assert(0)
        if self.model == "PatchTST" or self.model == "TimeXer":
            text_top_k *= configs.enc_in
        return text_top_k