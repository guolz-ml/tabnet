from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, log_loss


@dataclass
class MetricContainer:
    """Container holding a list of metrics.

    Parameters
    ----------
    metric_names : list of str
        List of metric names.
    prefix : str
        Prefix of metric names.

    """

    metric_names: List[str]
    prefix: str = ""

    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]

    def __call__(self, y_true, y_pred):
        """Compute all metrics and store into a dict.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_pred: np.ndarray
            Score matrix or vector

        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).

        """
        logs = {}
        for metric in self.metrics:
            logs[self.prefix + metric._name] = metric(y_true, y_pred)
        return logs


class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError("Custom Metrics must implement this function")

    @classmethod
    def get_metrics_by_names(cls, names):
        """Get list of metric classes.

        Parameters
        ----------
        cls : Metric
            Metric class.
        names : list
            List of metric names.

        Returns
        -------
        metrics : list
            List of metric classes.

        """
        available_metrics = cls.__subclasses__()
        available_names = [metric()._name for metric in available_metrics]
        metrics = []
        for name in names:
            assert name in available_names, f"{name} is not available"
            idx = available_names.index(name)
            metrics.append(available_metrics[idx]())
        return metrics


class AUC(Metric):
    """
    Root Mean Squared Error.
    """

    def __init__(self):
        self._name = "AUC"
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            MSE of predictions vs targets.
        """
        return roc_auc_score(y_true, y_score)


class Accuracy(Metric):
    """
    Root Mean Squared Error.
    """

    def __init__(self):
        self._name = "accuracy"
        self._maximize = True

    def __call__(self, y_true, y_score):
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            MSE of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)


class LogLoss(Metric):
    """
    Root Mean Squared Error.
    """

    def __init__(self):
        self._name = "logloss"
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            MSE of predictions vs targets.
        """
        return log_loss(y_true, y_score)


class MSE(Metric):
    """
    Root Mean Squared Error.
    """

    def __init__(self):
        self._name = "MSE"
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_pred: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            MSE of predictions vs targets.
        """
        return mean_squared_error(y_true, y_score)


class RMSE(Metric):
    """
    Root Mean Squared Error.
    """

    def __init__(self):
        self._name = "RMSE"
        self._maximize = False

    def __call__(self, y_true, y_score):
        """
        Compute RMSE (Root Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_pred: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            RMSE of predictions vs targets.
        """
        return np.sqrt(mean_squared_error(y_true, y_score))
