import torch
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter


class quantizedCGRU(nn.Module):
    def __init__(
        self,
        weights,
        bias,
        masks,
        device = "cpu",
        
    ):
        super(quantizedCGRU, self).__init__()

        # PARAMETERS
        self.input_size = weights["weight_ih"].size()[1]
        self.hidden_size = int(weights["weight_ih"].size()[0]/3)
        self.output_size = bias.size()[0]
        self.device = torch.device(device)
        self.h0 = np.zeros((1, self.hidden_size))
        self.c0 = np.zeros((1, self.hidden_size))


        # LAYERS
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.gru.to(self.device)

        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.linear.to(self.device)

        self.upload_weights(weights,masks,bias)


    def upload_weights(self, weights, masks, bias):
        for name, weight in weights.items():
            temp = weight * masks[name]
            if name == "weight":
                self.linear.weight = Parameter(temp,requires_grad=False)
                continue
            elif "bias" in name:
                setattr(self.gru,name,Parameter(temp,requires_grad=False))
            else:
                setattr(self.gru,name+"_l0",Parameter(temp,requires_grad=False))
        self.linear.bias =  Parameter(bias,requires_grad=False)

        

    
    def forward(self, x):
        input_f = x.to(self.device)

        out_h, _ = self.gru(input_f, self._build_initial_state(x))

        out = self.linear(out_h[:, -1, :])

        return out

    def _build_initial_state(self, x):
        s = torch.from_numpy(np.tile(self.h0, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)