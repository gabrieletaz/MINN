from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.nn_model.amn_qp import MechanisticLossWeighted
from src.utils.loggers import DataFrameLogger

class LossBalancer(ABC):
    @abstractmethod
    def balance_losses(self, losses: list):
        """
        Balance the losses based on the current state or policy.
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> dict:
        """
        Return the current weights.
        """
        pass


class NoLossBalancer(LossBalancer):
    def __init__(self, *args, **kwargs):
        pass
        
    def balance_losses(self, losses: list):
        """
        Do nothing.
        """
        pass
    
    def get_weights(self) -> dict:
        """
        Return an empty dictionary.
        """
        return {}


class MovingAverageLossBalancer(LossBalancer):
    def __init__(self, loss_module: MechanisticLossWeighted, window_size: int = 10):
        self._loss_logger = DataFrameLogger(clearml_task=None)
        self._loss_module = loss_module
        
        self.window_size = window_size
        
        self.weights = {"data_driven": 1, "mechanistic": 1}

    def balance_losses(self, losses: list):
        """
        Adjust the weights based on the current state or policy.
        """
        self._loss_logger.log_results({"data_driven": losses[0], "mechanistic": sum(losses[1:])})        
        losses_df = self._loss_logger.get_results()
        
        if losses_df.shape[0] >= self.window_size:
            
            last_losses_df = losses_df.sort_values(by='epoch', ascending=False).head(self.window_size)
            
            mean_losses = last_losses_df.mean()
            
            self.weights["data_driven"] = 1/mean_losses['data_driven']
            self.weights["mechanistic"] = 1/mean_losses['mechanistic']
        
        self._loss_module.data_driven_loss_balance = self.weights["data_driven"]
        self._loss_module.mechanistic_loss_balance = self.weights["mechanistic"]
    
    def get_weights(self) -> dict:
        """
        Return the current weights.
        """
        return self.weights


class AdaLossBalancer(LossBalancer):
    def __init__(self, loss_module: MechanisticLossWeighted, alpha: float = 0.2):
        self._loss_logger = DataFrameLogger(clearml_task=None)
        self._loss_module = loss_module
        
        # Alpha for EMA calculation
        self.alpha = alpha
        
        self.weights = {"data_driven": 1, "mechanistic": 1}

    def balance_losses(self, losses: list):
        """
        Adjust the weights based on the exponential moving average of the losses using the full DataFrame.
        """
        # Log the losses to the DataFrame
        self._loss_logger.log_results({"data_driven": losses[0], "mechanistic": sum(losses[1:])})
        losses_df = self._loss_logger.get_results()
        
        # Calculate EMA for the entire DataFrame at once
        emas = losses_df.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1]
        
        # Set weights as the inverse of the EMA
        self.weights["data_driven"] = 1 / (emas['data_driven'] + 1e-9)
        self.weights["mechanistic"] = 1 / (emas['mechanistic'] + 1e-9)
        
        # Apply the new weights to the loss module
        self._loss_module.data_driven_loss_balance = self.weights["data_driven"]
        self._loss_module.mechanistic_loss_balance = self.weights["mechanistic"]
    
    def get_weights(self) -> dict:
        """
        Return the current weights.
        """
        return self.weights


# ----- get loss balancer -----

def get_loss_balancer(balancer_cfg: DictConfig, loss_module: MechanisticLossWeighted):
    if balancer_cfg.type == 'no_balancer':
        return NoLossBalancer()
    elif balancer_cfg.type == 'ada':
        return AdaLossBalancer(loss_module, **balancer_cfg.params)
    elif balancer_cfg.type == 'moving_average':
        return MovingAverageLossBalancer(loss_module, **balancer_cfg.params)
    else:
        raise NotImplementedError(f'{balancer_cfg.type} loss balancer type not yet implemented')