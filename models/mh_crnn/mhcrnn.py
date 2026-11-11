"""Unmasked ablation of MAGIC Net, sharing its streaming interface."""

from models.crnn.cgru import cGRULinear
from models.mh_crnn.mhcrnn_manager import MultiHeadCrnnManager
from models.magic.magic_net import MagicNet


class MultiHeadCrnn(MagicNet):
    """Compare a new head with optional Ignore and hidden-size expansion.

    ``expand_option`` enables hidden-size growth. ``freeze_backbone`` controls
    whether new candidates start from rollback and freeze existing units.
    ``base_learner_weights`` accepts the shared cPNN base learner, a cGRULinear
    or cLSTMLinear column, or its state dict (with a matching ``model_class``).
    Full training checkpoints, including pending drift candidates and optimizer
    states, can be saved and resumed with pickle, as for MagicNet.
    """

    def __init__(
        self,
        model_class=cGRULinear,
        name="MultiHeadCRNN",
        freeze_backbone: bool = True,
        ignore_option: bool = True,
        expand_option: bool = True,
        hidden_mult=None,
        ensemble_batches=50,
        selection_batches=40,
        **kwargs,
    ):
        """Configure the unmasked recurrent model and its drift candidates.

        HeadOnly is always available. ignore_option and expand_option enable
        Ignore and hidden-size expansion. With freeze_backbone, HeadOnly
        freezes the recurrent layer and Expand freezes the existing blocks.
        Both start from rollback in that mode. Otherwise they start from
        the current network with trainable recurrent weights.

        hidden_mult sets the number of units added by Expand. By default the
        comparison lasts 50 minibatches and selection uses a rolling window
        of 25 minibatches. kwargs forwards the remaining MagicNet settings.
        """
        super().__init__(
            model_class=model_class,
            manager_class=MultiHeadCrnnManager,
            name=name,
            freeze_backbone=freeze_backbone,
            ignore_option=ignore_option,
            expand_option=expand_option,
            hidden_mult=hidden_mult,
            ensemble_batches=ensemble_batches,
            selection_batches=selection_batches,
            **kwargs,
        )
