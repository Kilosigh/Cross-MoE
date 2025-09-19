import os
import torch

from data_provider.data_factory import data_provider
from models import Cross_MoE, MoE_TS_Text_fuser
from exp.exp_model_dict import model_dict
model_dict['Cross_MoE'] = Cross_MoE
model_dict['MoE_TS_Text_fuser'] = MoE_TS_Text_fuser


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = model_dict
        self.device = self._acquire_device()
        self.data_loader_dict = {}
        self.data_set_dict = {}
        for f in ["train", "val", "test"]:    
            data_set, data_loader = data_provider(self.args, f)
            self.data_set_dict[f] = data_set
            self.data_loader_dict[f] = data_loader
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
