"""Continual-learning inference over the head-only MAGIC model pool."""

from copy import deepcopy

from river import metrics, utils

from models.magic.inference_magic import InferenceMagicNet


class InferenceMultiHeadCRNN(InferenceMagicNet):
    def initialize(self):
        super().initialize()
        if self.rolling_window is not None:
            self.metrics = [
                utils.Rolling(metrics.CohenKappa(), window_size=self.rolling_window)
                for _ in self.models
            ]

    def prepare_task_models(self):
        model_copy = deepcopy(self.model)
        # A CL checkpoint can fall in the middle of a drift comparison. Decide
        # only on this copy so the saved training state remains resumable.
        model_copy.manager.force_decision()
        model_copy.manager.in_grace_period = False
        models = []
        for task_id in range(
            model_copy.manager.initial_task_id, model_copy.manager.curr_task_idx + 1
        ):
            task_model = deepcopy(model_copy)
            manager = task_model.manager
            if task_id in manager.forgotten_models:
                manager.model = deepcopy(manager.forgotten_models[task_id])
            manager.curr_task_idx = task_id
            manager.model.eval()
            for param in manager.model.parameters():
                param.requires_grad_(False)
            task_model.reset_previous_data_points()
            models.append(task_model)
        return models
