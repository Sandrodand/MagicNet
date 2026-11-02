"""Masked GRU backbone sharing MAGIC's recurrent model lifecycle."""

from models.magic.piggyback_rnn import PiggyBackRNN
from models.magic.checkpoint_compat import legacy_argument
from models.magic.piggyback_layers import ElementWiseGRU


class PiggyBackGRU(PiggyBackRNN):
    """Three-gate GRU with shared heads, historical masks and expansion handling."""

    recurrent_name = "gru"
    recurrent_layer = ElementWiseGRU

    def __init__(self, *args, cGRU_weights=None, **kwargs):
        if cGRU_weights is not None:
            kwargs["base_learner_weights"] = legacy_argument(
                kwargs.get("base_learner_weights"), cGRU_weights
            )
        super().__init__(*args, **kwargs)
