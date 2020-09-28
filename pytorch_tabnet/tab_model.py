import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader


class TabNetClassifier(TabModel):
    def weight_updater(self, weights):
        """
        Updates weights dictionnary according to target_mapper.

        Parameters
        ----------
            weights : bool or dict
                Given weights for balancing training.
        Returns
        -------
            bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.long())

    def get_default_metric(self):
        if self.output_dim == 2:
            return "AUC"
        else:
            return "accuracy"

    def get_default_loss(self):
        return torch.nn.functional.cross_entropy

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            index: class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def convert_score(self, y_score):
        return softmax(y_score, axis=1)

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs)

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
            data: a :tensor: `torch.Tensor`
                Input data
            target: a :tensor: `torch.Tensor`
                Target data

        Returns
        -------
            batch_outs: dict
        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X), batch_size=self.batch_size, shuffle=False
        )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, M_loss = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res


class TabNetRegressor(TabModel):
    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.reshape(-1, 1))

    def get_default_metric(self):
        return "MSE"

    def get_default_loss(self):
        return torch.nn.functional.mse_loss

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        self.output_dim = 1

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def convert_score(self, y_score):
        return y_score
