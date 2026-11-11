"""Continual-learning inference over the head-only MAGIC model pool."""

from copy import deepcopy

from models.magic.inference_magic import InferenceMagicNet


class InferenceMultiHeadCRNN(InferenceMagicNet):
    """Evaluate historical MultiHeadCRNN heads and archived snapshots."""
    def prepare_task_models(self):
        """Resolve a copy of the checkpoint and build a model for each retained task."""
        model_copy = deepcopy(self.model)
        # A CL checkpoint can fall in the middle of a drift comparison. Decide
        # only on this copy so the saved training state remains resumable.
        model_copy.manager.force_decision()
        model_copy.manager.in_grace_period = False
        models = []
        for task_id in model_copy.manager.get_available_task_ids():
            task_model = deepcopy(model_copy)
            manager = task_model.manager
            if task_id in manager.forgotten_models:
                manager.model = deepcopy(manager.forgotten_models[task_id])
            else:
                manager.restore_historical_head(task_id)
            manager.curr_task_idx = task_id
            manager.model.eval()
            for param in manager.model.parameters():
                param.requires_grad_(False)
            task_model.reset_previous_data_points()
            models.append(task_model)
        return models
