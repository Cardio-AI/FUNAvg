from collections import OrderedDict
import torch

class FedAVG:
    def __init__(self, client_weights):
        self.client_weights = client_weights
    
    def __call__(self, model_state_dicts):
        weight_names = list(model_state_dicts[0].keys())
        new_state_dict = OrderedDict([])
        for layer_name in weight_names:
            ws = [msd[layer_name] for msd in model_state_dicts]
            new_w = torch.zeros_like(ws[0])
            for w, cw in zip(ws, self.client_weights):
                new_w += (cw * w).type(w.dtype)
            new_state_dict.update({layer_name: new_w})
        return new_state_dict
    

# https://arxiv.org/pdf/2206.10897.pdf
def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))