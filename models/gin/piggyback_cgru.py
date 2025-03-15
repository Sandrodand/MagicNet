# this file is for Main network that contains piggyback layers with masks.
# here we just
import torch
import torch.nn as nn
import models.gin.piggyback_layers as nl


class PiggyBackGRU(nn.Module):
    def __init__(
        self,
        input_size=4,
        device=torch.device("cpu"),
      	num_layers=1,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        bias=True,
        dropout=0.0,
        training=False,
        bidirectional=False,
        mask_init='1s',
      	mask_scale=2e-2,
        threshold_fn='sigmoid',
      	threshold=None,
        seq_len=5,
        cGRU_weights = None
        ):
        super(PiggyBackGRU, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        
        self.seq_len=seq_len
        
        
        if cGRU_weights is not None:
          GRU_weights = cGRU_weights.columns.columns[0].gru
          linear_weights = cGRU_weights.columns.columns[0].linear
        else:
           GRU_weights = None
           linear_weights = None



        self.classifier = nn.Sequential(
            nl.ElementWiseGRU(input_size=input_size, device=device, num_layers=num_layers, hidden_size=hidden_size, bias=bias, dropout=dropout,
                bidirectional=bidirectional, training=training, mask_init=mask_init,
                mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold,
                seq_len=self.seq_len,GRU_weights = GRU_weights),

            nl.ElementWiseLinear(in_features=hidden_size, out_features=output_size,
                mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,
				        threshold=threshold, Linear_mask_weights=self.Linear_mask_weights,linear_weights = linear_weights)
        )
        self.classifier.to(self.device)
        


    def forward(self,input):
        out = self.classifier(input)
        return out
    
    def reinit_piggymask(self, mask_init,freeze_masks = None, masks = None):
        
        for module in self.classifier:
          module.reinit_piggymask(mask_init,self.mask_scale,freeze_masks = freeze_masks, masks = masks)
    
    def reinit_linear_bias(self):
        self.classifier[1].reinit_linear_bias()


    def expand_hidden(self,multiplier):
      self.hidden_size = round(self.hidden_size + multiplier)
      freeze_masks = {}
      for module in self.classifier:
        freeze_masks.update(module.expand_hidden(multiplier))
      self.classifier.to(self.device)
      return freeze_masks
    
    def create_freezemask(self):
      freeze_mask = {}
      for module in self.classifier:
          freeze_mask.update(module.create_freezemask())
      return freeze_mask
    
    def get_piggymasks(self):
      sigmoid = nn.Sigmoid()
      piggymasks = {}
      
      piggymasks["mask_real_weight_ih"] = sigmoid(self.classifier[0].mask_real_weight_ih)
      piggymasks["mask_real_weight_hh"] = sigmoid(self.classifier[0].mask_real_weight_hh)
      piggymasks["mask_real_bias_ih_l0"] = sigmoid(self.classifier[0].mask_real_bias_ih_l0)
      piggymasks["mask_real_bias_hh_l0"] = sigmoid(self.classifier[0].mask_real_bias_hh_l0)
      piggymasks["mask_real_weight"] = sigmoid(self.classifier[1].mask_real_weight)
      return piggymasks

        
