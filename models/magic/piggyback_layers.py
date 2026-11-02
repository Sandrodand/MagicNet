import torch
from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch._VF import gru as _VF_gru
from torch._VF import lstm as _VF_lstm
from models.magic.activations import (
    Binarizer,
    Ternarizer,
    Thresholder,
    inverseCappedSigmoid,
    CAP_VALUE,
)

from models.magic.checkpoint_compat import recurrent_checkpoint_state, legacy_argument

DEFAULT_THRESHOLD = 5e-3


def GRUBlockMath(
    input,
    hn,
    weight_thresholded_ih,
    weight_thresholded_hh,
    bias_ih_l0,
    bias_hh_l0,
    batch_size=None,
    bias=True,
    num_layers=1,
    dropout=0.0,
    training=False,
    bidirectional=False,
    batch_first=False,
):
    """
    Executes the low-level mathematical operations of a GRU cell using the provided
    (and potentially masked) weights and biases.

    Parameters
    ----------
    input : torch.Tensor
        Input features.
    hn : torch.Tensor
        Initial hidden state.
    weight_thresholded_ih : torch.Tensor
        Masked input-hidden weights.
    weight_thresholded_hh : torch.Tensor
        Masked hidden-hidden weights.
    bias_ih_l0 : torch.Tensor
        Masked input-hidden biases.
    bias_hh_l0 : torch.Tensor
        Masked hidden-hidden biases.

    Returns
    -------
    tuple
        A tuple containing the output tensor and the new hidden state.
    """
    tensors = [weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0, bias_hh_l0]
    return _VF_gru(
        input, hn, tensors, bias, num_layers, dropout, training,
        bidirectional, batch_first,
    )


def LSTMBlockMath(
    input,
    hn,
    weight_thresholded_ih,
    weight_thresholded_hh,
    bias_ih_l0,
    bias_hh_l0,
    bias=True,
    num_layers=1,
    dropout=0.0,
    training=False,
    bidirectional=False,
    batch_first=False,
):
    """Run a masked LSTM sequence from the supplied (hidden, cell) states."""
    tensors = [weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0, bias_hh_l0]
    output, _, _ = _VF_lstm(
        input,
        hn,
        tensors,
        bias,
        num_layers,
        dropout,
        training,
        bidirectional,
        batch_first,
    )
    return output


class ElementWiseRNN(nn.Module):
    """Architecture-independent recurrent masking, freezing and gate expansion."""

    gate_count = None
    gate_names = ()
    mask_parameter_names = ("weight_ih", "weight_hh", "bias_ih_l0", "bias_hh_l0")

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
        batch_first=True,
        frozen_mask_param=2e-2,
        threshold_fn="sigmoid",
        threshold=5e-3,
        seq_len=10,
        recurrent_weights=None,
        recurrent_mask_weights=None,
    ):
        super().__init__()
        if self.gate_count is None:
            raise TypeError("Instantiate a concrete ElementWiseRNN subclass")

        # PARAMETERS
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = round(hidden_size)
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.bias = bias
        self.dropout = dropout
        self.training = training
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.frozen_mask_param = frozen_mask_param
        self.threshold_fn = threshold_fn
        self.threshold = threshold
        self.seq_len = (seq_len,)
        self.recurrent_mask_weights = (
            [] if recurrent_mask_weights is None else
            [torch.as_tensor(mask).detach().clone() for mask in recurrent_mask_weights]
        )
        self.inverse_thresholder = inverseCappedSigmoid

        self._reset_initial_state()

        if recurrent_weights is not None:
            self.weight_ih = Parameter(recurrent_weights.weight_ih_l0.detach().clone())
            self.weight_hh = Parameter(recurrent_weights.weight_hh_l0.detach().clone())
            self.bias_ih_l0 = Parameter(recurrent_weights.bias_ih_l0.detach().clone())
            self.bias_hh_l0 = Parameter(recurrent_weights.bias_hh_l0.detach().clone())
        else:
            self.weight_ih = Parameter(
                torch.empty(
                    self.gate_count * self.hidden_size, input_size, device=self.device
                ).uniform_(
                    -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
                )
            )
            self.weight_hh = Parameter(
                torch.empty(
                    self.gate_count * self.hidden_size, self.hidden_size, device=self.device
                ).uniform_(
                    -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
                )
            )

            self.bias_ih_l0 = Parameter(
                torch.empty(
                    self.gate_count * self.hidden_size, device=self.device
                ).uniform_(
                    -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
                )
            )
            self.bias_hh_l0 = Parameter(
                torch.empty(
                    self.gate_count * self.hidden_size, device=self.device
                ).uniform_(
                    -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
                )
            )

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.threshold = threshold
        self.threshold_fn_name = threshold_fn
        if threshold_fn == "binarizer":
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == "ternarizer":
            self.threshold_fn = Ternarizer()
        else:
            self.threshold_fn = Thresholder(function=threshold_fn)
            self.frozen_mask_param = CAP_VALUE

        self.initialize_piggymask(self.frozen_mask_param)
        if self.recurrent_mask_weights:
            if len(self.recurrent_mask_weights) != len(self.mask_parameter_names):
                raise ValueError("Provide one initial mask for each recurrent parameter")
            with torch.no_grad():
                for name, values in zip(self.mask_parameter_names, self.recurrent_mask_weights):
                    parameter = getattr(self, "mask_real_" + name)
                    if values.shape != parameter.shape:
                        raise ValueError(f"Initial mask shape does not match {name}")
                    parameter.copy_(values.to(parameter))

    def __setstate__(self, state):
        super().__setstate__(recurrent_checkpoint_state(state))

    def forward(self, input):
        initial_state = self._initial_state(input)

        # Apply threshold function to real valued masks
        mask_thresholded_ih = self.threshold_fn(self.mask_real_weight_ih)
        mask_thresholded_hh = self.threshold_fn(self.mask_real_weight_hh)
        mask_thresholded_bias_ih_l0 = self.threshold_fn(self.mask_real_bias_ih_l0)
        mask_thresholded_bias_hh_l0 = self.threshold_fn(self.mask_real_bias_hh_l0)

        # Mask weights with above mask.
        weight_thresholded_ih = mask_thresholded_ih * self.weight_ih
        weight_thresholded_hh = mask_thresholded_hh * self.weight_hh
        weight_thresholded_bias_ih = mask_thresholded_bias_ih_l0 * self.bias_ih_l0
        weight_thresholded_bias_hh = mask_thresholded_bias_hh_l0 * self.bias_hh_l0

        return self._recurrent_forward(
            input,
            initial_state,
            weight_thresholded_ih,
            weight_thresholded_hh,
            weight_thresholded_bias_ih,
            weight_thresholded_bias_hh,
        )

    def _initial_state(self, input):
        raise NotImplementedError

    def _recurrent_forward(self, input, state, weight_ih, weight_hh, bias_ih, bias_hh):
        raise NotImplementedError

    def reinit_piggymask(self, mask_param_init, freeze_masks=None, mask_params=None):
        if mask_params is not None:
            if self.threshold_fn_name == "binarizer":
                # Funzione per invertire la maschera binaria (0/1) in pesi latenti
                def invert_bin(m):
                    res = m.clone().fill_(-mask_param_init)  # i 0 diventano negativi
                    res[m == 1] = mask_param_init  # gli 1 diventano positivi
                    return res

                self.mask_real_weight_ih = Parameter(
                    invert_bin(mask_params["mask_real_weight_ih"])
                )
                self.mask_real_weight_hh = Parameter(
                    invert_bin(mask_params["mask_real_weight_hh"])
                )
                self.mask_real_bias_ih_l0 = Parameter(
                    invert_bin(mask_params["mask_real_bias_ih_l0"])
                )
                self.mask_real_bias_hh_l0 = Parameter(
                    invert_bin(mask_params["mask_real_bias_hh_l0"])
                )
            else:
                self.mask_real_weight_ih = Parameter(
                    self.inverse_thresholder(mask_params["mask_real_weight_ih"])
                )
                self.mask_real_weight_hh = Parameter(
                    self.inverse_thresholder(mask_params["mask_real_weight_hh"])
                )
                self.mask_real_bias_ih_l0 = Parameter(
                    self.inverse_thresholder(mask_params["mask_real_bias_ih_l0"])
                )
                self.mask_real_bias_hh_l0 = Parameter(
                    self.inverse_thresholder(mask_params["mask_real_bias_hh_l0"])
                )
        else:
            self.initialize_piggymask(mask_param_init, freeze_masks=freeze_masks)

    def create_freezemask(self):
        """
        Create initial freeze masks.
        """
        freeze_masks = {}
        freeze_masks["weight_ih"] = torch.zeros_like(self.weight_ih)
        freeze_masks["weight_hh"] = torch.zeros_like(self.weight_hh)
        freeze_masks["bias_ih_l0"] = torch.zeros_like(self.bias_ih_l0)
        freeze_masks["bias_hh_l0"] = torch.zeros_like(self.bias_hh_l0)
        return freeze_masks

    def expand_hidden(self, addendum):
        """
        Parameters are expanded by a factor of "addendum". New weights are initialized normally while
        previously learned ones are kept and frozen.
        Each gate retains its old weight and bias blocks at the expanded offsets.
        """
        old_hidden_size = self.hidden_size
        self.hidden_size = round(self.hidden_size + addendum)
        freeze_masks = {}

        # 1. Creiamo i nuovi tensori
        temp_weight_ih = self.weight_ih.new_empty(
            self.gate_count * self.hidden_size, self.input_size
        ).uniform_(
            -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
        )
        temp_weight_hh = self.weight_hh.new_empty(
            self.gate_count * self.hidden_size, self.hidden_size
        ).uniform_(
            -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
        )
        temp_bias_ih_l0 = self.bias_ih_l0.new_empty(
            self.gate_count * self.hidden_size
        ).uniform_(
            -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
        )
        temp_bias_hh_l0 = self.bias_hh_l0.new_empty(
            self.gate_count * self.hidden_size
        ).uniform_(
            -sqrt(1 / self.hidden_size), sqrt(1 / self.hidden_size)
        )

        freeze_masks["weight_ih"] = torch.zeros_like(temp_weight_ih)
        freeze_masks["weight_hh"] = torch.zeros_like(temp_weight_hh)
        freeze_masks["bias_ih_l0"] = torch.zeros_like(temp_bias_ih_l0)
        freeze_masks["bias_hh_l0"] = torch.zeros_like(temp_bias_hh_l0)

        for i in range(self.gate_count):
            old_start = i * old_hidden_size
            old_end = (i + 1) * old_hidden_size
            new_start = i * self.hidden_size
            new_end = new_start + old_hidden_size

            temp_weight_ih[new_start:new_end, :].copy_(
                self.weight_ih[old_start:old_end, :]
            )
            freeze_masks["weight_ih"][new_start:new_end, :].fill_(1)

            temp_weight_hh[new_start:new_end, :old_hidden_size].copy_(
                self.weight_hh[old_start:old_end, :]
            )
            freeze_masks["weight_hh"][new_start:new_end, :old_hidden_size].fill_(1)

            temp_bias_ih_l0[new_start:new_end].copy_(self.bias_ih_l0[old_start:old_end])
            freeze_masks["bias_ih_l0"][new_start:new_end].fill_(1)

            temp_bias_hh_l0[new_start:new_end].copy_(self.bias_hh_l0[old_start:old_end])
            freeze_masks["bias_hh_l0"][new_start:new_end].fill_(1)

        self.weight_ih = Parameter(temp_weight_ih)
        self.weight_hh = Parameter(temp_weight_hh)
        self.bias_ih_l0 = Parameter(temp_bias_ih_l0)
        self.bias_hh_l0 = Parameter(temp_bias_hh_l0)

        self._reset_initial_state()
        self.adjust_piggymask(mask_param_init=self.frozen_mask_param)

        return freeze_masks

    def adjust_piggymask(self, mask_param_init):
        old_hidden_size = self.mask_real_weight_ih.size(0) // self.gate_count

        temp_mask_real_weight_ih = self.weight_ih.data.new(self.weight_ih.size()).fill_(
            mask_param_init
        )
        temp_mask_real_weight_hh = self.weight_hh.data.new(self.weight_hh.size()).fill_(
            mask_param_init
        )
        temp_mask_real_bias_ih_l0 = self.weight_ih.data.new(
            self.bias_ih_l0.size()
        ).fill_(mask_param_init)
        temp_mask_real_bias_hh_l0 = self.weight_hh.data.new(
            self.bias_hh_l0.size()
        ).fill_(mask_param_init)

        # Trasloco a blocchi anche per le maschere continue!
        for i in range(self.gate_count):
            old_start = i * old_hidden_size
            old_end = (i + 1) * old_hidden_size
            new_start = i * self.hidden_size
            new_end = new_start + old_hidden_size

            temp_mask_real_weight_ih[new_start:new_end, :].copy_(
                self.mask_real_weight_ih[old_start:old_end, :]
            )
            temp_mask_real_weight_hh[new_start:new_end, :old_hidden_size].copy_(
                self.mask_real_weight_hh[old_start:old_end, :]
            )
            temp_mask_real_bias_ih_l0[new_start:new_end].copy_(
                self.mask_real_bias_ih_l0[old_start:old_end]
            )
            temp_mask_real_bias_hh_l0[new_start:new_end].copy_(
                self.mask_real_bias_hh_l0[old_start:old_end]
            )

        self.mask_real_weight_ih = Parameter(temp_mask_real_weight_ih)
        self.mask_real_weight_hh = Parameter(temp_mask_real_weight_hh)
        self.mask_real_bias_ih_l0 = Parameter(temp_mask_real_bias_ih_l0)
        self.mask_real_bias_hh_l0 = Parameter(temp_mask_real_bias_hh_l0)

    def initialize_piggymask(self, mask_param_init, freeze_masks=None):
        temp_mask_real_weight_ih = self.weight_ih.data.new(self.weight_ih.size())
        temp_mask_real_weight_hh = self.weight_hh.data.new(self.weight_hh.size())
        temp_mask_real_bias_ih_l0 = self.weight_ih.data.new(self.bias_ih_l0.size())
        temp_mask_real_bias_hh_l0 = self.weight_hh.data.new(self.bias_hh_l0.size())

        temp_mask_real_weight_ih.fill_(mask_param_init)
        temp_mask_real_weight_hh.fill_(mask_param_init)
        temp_mask_real_bias_ih_l0.fill_(mask_param_init)
        temp_mask_real_bias_hh_l0.fill_(mask_param_init)

        if freeze_masks is not None and self.threshold_fn_name == "sigmoid":
            temp_mask_real_weight_ih[
                freeze_masks["weight_ih"].eq(1)
            ] = temp_mask_real_weight_ih[freeze_masks["weight_ih"].eq(1)].uniform_(
                -1 * mask_param_init, mask_param_init
            )
            temp_mask_real_weight_hh[
                freeze_masks["weight_hh"].eq(1)
            ] = temp_mask_real_weight_hh[freeze_masks["weight_hh"].eq(1)].uniform_(
                -1 * mask_param_init, mask_param_init
            )
            temp_mask_real_bias_ih_l0[
                freeze_masks["bias_ih_l0"].eq(1)
            ] = temp_mask_real_bias_ih_l0[freeze_masks["bias_ih_l0"].eq(1)].uniform_(
                -1 * mask_param_init, mask_param_init
            )
            temp_mask_real_bias_hh_l0[
                freeze_masks["bias_hh_l0"].eq(1)
            ] = temp_mask_real_bias_hh_l0[freeze_masks["bias_hh_l0"].eq(1)].uniform_(
                -1 * mask_param_init, mask_param_init
            )

        self.mask_real_weight_ih = Parameter(temp_mask_real_weight_ih)
        self.mask_real_weight_hh = Parameter(temp_mask_real_weight_hh)
        self.mask_real_bias_ih_l0 = Parameter(temp_mask_real_bias_ih_l0)
        self.mask_real_bias_hh_l0 = Parameter(temp_mask_real_bias_hh_l0)

    def _reset_initial_state(self):
        self.h0 = np.zeros((1, self.hidden_size))

    def _build_initial_state(self, x, state):
        return torch.as_tensor(
            np.tile(state, (1, x.size(0 if self.batch_first else 1), 1)),
            dtype=x.dtype, device=x.device
        ).requires_grad_(True)


class ElementWiseGRU(ElementWiseRNN):
    """Reset, update and candidate gates with one hidden state."""

    gate_names = ("reset", "update", "new")
    gate_count = len(gate_names)

    def __init__(self, *args, GRU_weights=None, GRU_mask_weights=None, **kwargs):
        if GRU_weights is not None:
            kwargs["recurrent_weights"] = legacy_argument(
                kwargs.get("recurrent_weights"), GRU_weights
            )
        if GRU_mask_weights is not None:
            kwargs["recurrent_mask_weights"] = legacy_argument(
                kwargs.get("recurrent_mask_weights"), GRU_mask_weights
            )
        super().__init__(*args, **kwargs)

    def _initial_state(self, input):
        self.hn = self._build_initial_state(input, self.h0)
        return self.hn

    def _recurrent_forward(self, input, state, weight_ih, weight_hh, bias_ih, bias_hh):
        output, _ = GRUBlockMath(
            input, state, weight_ih, weight_hh, bias_ih, bias_hh,
            bias=self.bias, num_layers=self.num_layers, dropout=self.dropout,
            training=self.training, bidirectional=self.bidirectional,
            batch_first=self.batch_first,
        )
        return output


class ElementWiseLSTM(ElementWiseRNN):
    """Input, forget, cell and output gates with hidden and cell states."""

    gate_names = ("input", "forget", "cell", "output")
    gate_count = len(gate_names)

    def _reset_initial_state(self):
        super()._reset_initial_state()
        self.c0 = np.zeros((1, self.hidden_size))

    def _initial_state(self, input):
        self.hn = self._build_initial_state(input, self.h0)
        self.cn = self._build_initial_state(input, self.c0)
        return self.hn, self.cn

    def _recurrent_forward(self, input, state, weight_ih, weight_hh, bias_ih, bias_hh):
        return LSTMBlockMath(
            input, state, weight_ih, weight_hh, bias_ih, bias_hh,
            bias=self.bias, num_layers=self.num_layers, dropout=self.dropout,
            training=self.training, bidirectional=self.bidirectional,
            batch_first=self.batch_first,
        )


class ElementWiseLinear(nn.Module):
    """Modified linear layer."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        frozen_mask_param=2e-2,
        threshold_fn="sigmoid",
        threshold=5e-3,
        linear_mask_weights=None,
        linear_weights=None,
        *,
        Linear_mask_weights=None,
    ):
        super(ElementWiseLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.frozen_mask_param = frozen_mask_param
        linear_mask_weights = legacy_argument(linear_mask_weights, Linear_mask_weights)
        if isinstance(linear_mask_weights, (list, tuple)) and not linear_mask_weights:
            linear_mask_weights = None
        self.linear_mask_weights = (
            None if linear_mask_weights is None else
            torch.as_tensor(linear_mask_weights).detach().clone()
        )
        self.inverse_thresholder = inverseCappedSigmoid

        if linear_weights is not None:
            self.weight = Parameter(linear_weights.weight.detach().clone())
            self.bias = Parameter(linear_weights.bias.detach().clone())
        else:
            self.weight = Parameter(
                torch.Tensor(out_features, in_features).uniform_(
                    -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
                )
            )
            if bias:
                self.bias = Parameter(
                    torch.Tensor(out_features).uniform_(
                        -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
                    )
                )
            else:
                self.register_parameter("bias", None)

        # Initialize the thresholder.
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.threshold = threshold
        self.threshold_fn_name = threshold_fn
        if threshold_fn == "binarizer":
            self.threshold_fn = Binarizer(threshold)
        elif threshold_fn == "ternarizer":
            self.threshold_fn = Ternarizer()
        else:
            self.threshold_fn = Thresholder(function=threshold_fn)
            self.frozen_mask_param = CAP_VALUE
        self.initialize_piggymask(self.frozen_mask_param, self.linear_mask_weights)

    def __setstate__(self, state):
        super().__setstate__(recurrent_checkpoint_state(state))

    def reinit_piggymask(self, mask_param_init, freeze_masks=None, mask_params=None):
        if mask_params is not None:
            if self.threshold_fn_name == "binarizer":

                def invert_bin(m):
                    res = m.clone().fill_(-mask_param_init)
                    res[m == 1] = mask_param_init
                    return res

                self.mask_real_weight = Parameter(
                    invert_bin(mask_params["mask_real_weight"])
                )
            else:
                self.mask_real_weight = Parameter(
                    self.inverse_thresholder(mask_params["mask_real_weight"])
                )
        else:
            self.initialize_piggymask(mask_param_init, freeze_mask=freeze_masks)

    def reinit_linear_bias(self):
        if self.bias is not None:
            self.bias = Parameter(
                torch.Tensor(self.out_features).uniform_(
                    -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
                )
            )
        else:
            self.register_parameter("bias", None)

    def create_freezemask(self):
        freeze_masks = {}
        freeze_masks["weight"] = torch.zeros_like(self.weight)
        return freeze_masks

    def initialize_piggymask(
        self, mask_param_init, linear_mask_weights=None, freeze_mask=None
    ):
        temp_mask_real_weight = self.weight.data.new(self.weight.size())
        temp_mask_real_weight.fill_(mask_param_init)

        if freeze_mask is not None and self.threshold_fn_name == "sigmoid":
            temp_mask_real_weight[freeze_mask["weight"].eq(1)] = temp_mask_real_weight[
                freeze_mask["weight"].eq(1)
            ].uniform_(-1 * mask_param_init, mask_param_init)

        if linear_mask_weights is not None:
            values = torch.as_tensor(linear_mask_weights).to(self.weight)
            if values.shape != self.weight.shape:
                raise ValueError("Initial linear mask shape does not match the weights")
            self.mask_real_weight = Parameter(values.detach().clone())
        else:
            self.mask_real_weight = Parameter(temp_mask_real_weight)

    def expand_hidden(self, multiplier):
        freeze_mask = {}
        self.in_features = round(self.in_features + multiplier)
        temp_weight = torch.Tensor(self.out_features, self.in_features).uniform_(
            -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
        )
        freeze_mask["weight"] = torch.zeros_like(temp_weight)
        temp_weight[: self.weight.size(0), : self.weight.size(1)].copy_(self.weight)
        freeze_mask["weight"][: self.weight.size(0), : self.weight.size(1)].fill_(1)
        self.weight = Parameter(temp_weight)
        if self.bias is not None:
            self.bias = Parameter(
                torch.Tensor(self.out_features).uniform_(
                    -sqrt(1 / self.in_features), sqrt(1 / self.in_features)
                )
            )
        else:
            self.register_parameter("bias", None)
        self.adjust_piggymask(self.frozen_mask_param)
        return freeze_mask

    def adjust_piggymask(self, mask_param_init):
        temp_mask_real_weight = self.weight.data.new(self.weight.size())

        temp_mask_real_weight.fill_(mask_param_init)

        temp_mask_real_weight[
            : self.mask_real_weight.size(0), : self.mask_real_weight.size(1)
        ].copy_(self.mask_real_weight)

        self.mask_real_weight = Parameter(temp_mask_real_weight)

    def forward(self, input):
        # Get transformed mask from real-valued mask.
        mask_thresholded = self.threshold_fn(self.mask_real_weight)
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        a = F.linear(input[:, -1, :], weight_thresholded, self.bias)
        # Get output using modified weight.
        return a
