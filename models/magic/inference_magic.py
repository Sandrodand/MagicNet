from models.magic.magic_net import MagicNet
import numpy as np
import torch
from river import metrics
import river.utils as river_utils
import pickle


class InferenceMagicNet:
    """Evaluate a MAGIC checkpoint using the historical candidates in its pool."""
    def __init__(self, model: MagicNet, ensemble_data_points=500, rolling_window=None):
        """Copy a trained MAGIC model and prepare its historical candidates.

        ensemble_data_points defaults to 500 label updates before keeping
        the best candidate. Use -1 to keep all candidates throughout.
        rolling_window optionally sets the Kappa window in observations.
        With None, use cumulative Kappa for candidate selection.
        """
        self.model: MagicNet = pickle.loads(pickle.dumps(model))
        self._previous_data_points = None
        self.metrics = None
        self.no_preparation = False
        self.selected = None
        self.models = []
        self.rolling_window = rolling_window
        self.ensemble_predictions = {}
        self.reset_previous_data_points()
        self.ensemble_data_points = ensemble_data_points
        self.count = 0
        self.predictions = {}

    def predict_one(self, x, timestamp=-1):
        """Predict the label of x using the currently selected historical candidate.

        The latest task is selected initially. Keep predictions from all
        active candidates under timestamp for update_inference. Use class
        zero during recurrent warm-up.
        """
        self.predictions[timestamp] = []
        for i, m in enumerate(self.models):
            pred = m.predict_one(
                x,
                previous_data_points=self._previous_data_points,
            )
            if pred is None:
                pred = 0
            pred = int(pred)
            self.predictions[timestamp].append(pred)
            self.ensemble_predictions[m.manager.curr_task_idx].append(pred)
        if self._previous_data_points is None:
            self._previous_data_points = np.array(x).reshape(1, -1)
        else:
            self._previous_data_points = np.concatenate(
                [self._previous_data_points, np.array(x).reshape(1, -1)]
            )[-(self.model.get_seq_len() - 1) :]
        return self.predictions[timestamp][self.selected]

    def update_inference(self, y, timestamp=-1):
        """Score predictions at timestamp using y and update candidate selection.

        After ensemble_data_points label updates, retain the best candidate.
        Use the timestamp supplied to predict_one for delayed labels.
        """
        y = int(y)
        if timestamp in self.predictions:
            for i in range(len(self.predictions[timestamp])):
                self.metrics[i] = self.metrics[i].update(
                    y, self.predictions[timestamp][i]
                )
            self.selected = np.argmax([m.get() for m in self.metrics])
            del self.predictions[timestamp]
        self.count += 1
        if self.count == self.ensemble_data_points:
            self.models = [self.models[self.selected]]
            self.metrics = [self.metrics[self.selected]]
            self.selected = 0
            self.predictions = {}

    def prepare_task_models(self):
        """Build independent evaluation models for the available task IDs.

        Resolve a pending comparison on a copy. Use archived networks when
        available and restore historical masks, heads, or biases on copies
        of the shared network for the remaining tasks.
        """
        models = []
        model_copy = pickle.loads(pickle.dumps(self.model))
        if model_copy.manager.in_expansion:
            model_copy.manager.force_decision()
        model_copy.manager.in_grace_period = False
        model_copy.manager.in_expansion = False
        for task_id in model_copy.manager.get_available_task_ids():
            task_model = pickle.loads(pickle.dumps(model_copy))
            if task_id in task_model.manager.forgotten_models:
                task_model.manager.model = pickle.loads(
                    pickle.dumps(task_model.manager.forgotten_models[task_id])
                )
                task_model.manager.model.eval()
            else:
                task_model.manager.restore_historical_head(task_id)
                task_model.manager.model.eval()
                if task_id in task_model.manager.piggymask_list:
                    historical_mask = pickle.loads(
                        pickle.dumps(task_model.manager.piggymask_list[task_id])
                    )
                    task_model.manager.model.reinit_piggymask(
                        mask_params=historical_mask
                    )
                if (
                    not task_model.manager.multi_head
                    and task_id in task_model.manager.biases
                ):
                    historical_bias = pickle.loads(
                        pickle.dumps(task_model.manager.biases[task_id])
                    )
                    if historical_bias is not None:
                        # Restore the historical bias on the evaluation model's device.
                        task_model.manager.model.classifier[
                            1
                        ].bias = torch.nn.Parameter(
                            historical_bias.clone().to(task_model.manager.model.device)
                        )
            task_model.manager.curr_task_idx = task_id
            models.append(task_model)
        return models

    def initialize(self):
        """Rebuild the evaluation ensemble and reset its context and scores."""
        self.predictions = {}
        self._previous_data_points = None

        self.models = self.prepare_task_models()
        # Reset the freshly reconstructed copies, not the models discarded
        # by prepare_task_models. The source training checkpoint stays intact.
        for model in self.models:
            model.reset_previous_data_points()
        if self.rolling_window is not None:
            self.metrics = [
                river_utils.Rolling(
                    metrics.CohenKappa(), window_size=self.rolling_window
                )
                for _ in range(len(self.models))
            ]
        else:
            self.metrics = [metrics.CohenKappa() for _ in range(len(self.models))]
        self.selected = len(self.models) - 1
        for m in self.models:
            self.ensemble_predictions[m.manager.curr_task_idx] = []
        self.count = 0

    def reset_previous_data_points(self):
        """Start a fresh CL evaluation sequence with the full historical pool."""
        self.initialize()
