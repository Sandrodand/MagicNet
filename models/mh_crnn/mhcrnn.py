"""Head-only ablation of MAGIC Net, sharing its streaming interface."""

from models.crnn.cgru import cGRULinear
from models.mh_crnn.mhcrnn_manager import MultiHeadCrnnManager
from models.magic.magic_net import MagicNet


class MultiHeadCrnn(MagicNet):
    """Compare Ignore and a new head at every drift, without weight masks.

    The first accepted expansion freezes the recurrent backbone permanently.
    ``base_learner_weights`` accepts the shared cPNN base learner, a cGRULinear
    or cLSTMLinear column, or its state dict (with a matching ``model_class``).
    Full training checkpoints, including pending drift candidates and optimizer
    states, can be saved and resumed with pickle, as for MagicNet.
    """

    def __init__(self, model_class=cGRULinear, **kwargs):
        super().__init__(
            model_class=model_class, manager_class=MultiHeadCrnnManager, **kwargs
        )
