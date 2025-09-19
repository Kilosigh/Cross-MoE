import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, TimeXer, TimeLLM



model_dict = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Nonstationary_Transformer': Nonstationary_Transformer,
    'DLinear': DLinear,
    'FEDformer': FEDformer,
    'Informer': Informer,
    'LightTS': LightTS,
    'Reformer': Reformer,
    'ETSformer': ETSformer,
    'PatchTST': PatchTST,
    'Pyraformer': Pyraformer,
    'MICN': MICN,
    'Crossformer': Crossformer,
    'FiLM': FiLM,
    'iTransformer': iTransformer,
    'Koopa': Koopa,
    'TiDE': TiDE,
    'FreTS': FreTS,
    'TimeMixer': TimeMixer,
    'TSMixer': TSMixer,
    'SegRNN': SegRNN,
    'TimeXer': TimeXer,
    'TimeLLM': TimeLLM
}
