from abc import ABC, abstractmethod

from omegaconf import DictConfig
import torch

class LossAggregator(ABC):    
    @abstractmethod
    def aggregate_losses(self):
        """
        Aggregate losses based on the current state or policy.
        """
        pass


class SumLossAggregator(LossAggregator):
    def __init__(self, *args, **kwargs):
        pass
        
    def aggregate_losses(self, losses: list, weights: list) -> float:
        assert len(losses) == len(weights), ValueError(f'Expected the same number of losses and weights, got {len(losses)} losses and {len(weights)} weights')
        total_loss = 0
        for loss, weight in zip(losses, weights):
            total_loss += loss * weight
        return total_loss


class LinearBoundLossAggregator(LossAggregator):
    def __init__(self, mechanistic_bound=None, mechanistic_coefficient=2, data_driven_bound=None, data_driven_coefficient=2):
        self.mechanistic_bound = mechanistic_bound
        self.mechanistic_coefficient = mechanistic_coefficient
        
        self.data_driven_bound = data_driven_bound
        self.data_driven_coefficient = data_driven_coefficient

    def aggregate_losses(self, losses: list, weights: list) -> float:
        assert len(losses) == len(weights), ValueError(f'Expected the same number of losses and weights, got {len(losses)} losses and {len(weights)} weights')
        assert len(losses) == 2, ValueError(f'Expected 2 losses, got {len(losses)}')
        
        total_loss = 0
        for loss, weight, bound, coefficient in zip(losses, weights, [self.data_driven_bound, self.mechanistic_bound], [self.data_driven_coefficient, self.mechanistic_coefficient]):
            total_loss += self.__loss_with_bound(loss, bound, coefficient) * weight
        return total_loss

    def __loss_with_bound(self, loss, bound, coefficient):
        if bound is not None:
            # Create a mask for elements where loss is greater than or equal to bound
            mask = loss >= bound
            
            if mask.any():
                # Adjust losses where the mask is True without modifying the original tensor
                # adjusted_losses = bound + (torch.pow(exp_base, loss - bound) - 1)
                adjusted_losses = bound + coefficient*(loss - bound)
                loss = torch.where(mask, adjusted_losses, loss)
            # # Apply the operation only to elements where the mask is True
            # loss[mask] = bound + (torch.pow(exp_base, loss[mask] - bound) - 1) # -1 so that the exp function starts from 0 and not 1
        return loss


# ----- get loss aggregator -----

def get_loss_aggregator(loss_aggregator_cfg: DictConfig):
    if loss_aggregator_cfg.type == 'sum':
        return SumLossAggregator()
    elif loss_aggregator_cfg.type == 'double_linear_bound':
        return LinearBoundLossAggregator(**loss_aggregator_cfg.params)
    else:
        raise NotImplementedError(f'{loss_aggregator_cfg.type} loss aggregator type not yet implemented')