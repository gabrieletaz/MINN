from typing import List, Union  

from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

from src.nn_model.amn_qp import MechanisticLoss, MechanisticLossWeighted
from collections import defaultdict


class LossWeightsScheduler(ABC):
    @abstractmethod
    def adjust_weights(self):
        """
        Adjust the weights based on the current state or policy.
        """
        pass
    
    @abstractmethod
    def update_current_epoch(self, epoch):
        """
        Updates the current epoch counter.
        """
        pass

    @abstractmethod
    def get_loss_module(self):
        """
        Return the loss computation module/object.
        """
        pass
    
    @abstractmethod
    def get_weights(self):
        """
        Return the current weights.
        """
        pass
    

class BaseLossWeightsScheduler(LossWeightsScheduler):
    def __init__(self, max_epoch, loss_module: Union[MechanisticLoss, MechanisticLossWeighted]):
        self.current_epoch = 0
        self.max_epoch = max_epoch
        self.loss_module = loss_module
        
        self.losses = defaultdict(list)

    def adjust_weights(self):
        """
        Adjust the weights based on the current state or policy.
        """
        pass
    
    def update_current_epoch(self, epoch):
        """
        Updates the current epoch counter.
        """
        self.current_epoch = epoch

    def get_loss_module(self):
        """
        Return the loss computation module/object.
        """
        return self.loss_module
    
    def get_weights(self):
        """
        Return the current weights.
        """
        pass


class NoWeightsScheduler(BaseLossWeightsScheduler):        
    def adjust_weights(self):
        pass

    def get_weights(self) -> dict:
        return {"data_driven": 1, "mechanistic": 1}


class ConstantWeightsScheduler(BaseLossWeightsScheduler):
    def __init__(self, max_epoch, loss_module: MechanisticLossWeighted, data_driven_weight: float = 0., mechanistic_weight: Union[float, None] = None):
        super().__init__(max_epoch, loss_module)
        self.data_driven_weight = data_driven_weight
        self.mechanistic_weight = mechanistic_weight if mechanistic_weight is not None else 1 - data_driven_weight
        
    def adjust_weights(self):
        self.loss_module.data_driven_weight = self.data_driven_weight
        self.loss_module.mechanistic_weight = self.mechanistic_weight

    def get_weights(self) -> dict:
        return {"data_driven": self.data_driven_weight, "mechanistic": self.mechanistic_weight}


class DataDrivenLinearScheduler(BaseLossWeightsScheduler):
    def __init__(self, max_epoch, loss_module: MechanisticLossWeighted, initial_data_driven_weight: float = 0., final_data_driven_weight: float = 1.):
        super().__init__(max_epoch, loss_module)
        self.initial_data_driven_weight = initial_data_driven_weight
        self.final_data_driven_weight = final_data_driven_weight
        assert self.initial_data_driven_weight >= 0 and self.initial_data_driven_weight <= 1 and self.final_data_driven_weight >= 0 and self.final_data_driven_weight <= 1, "Weights must be between 0 and 1."
        assert self.initial_data_driven_weight != self.final_data_driven_weight, ValueError("Initial and final weights must be different.")
        
    def adjust_weights(self):
        step = (self.final_data_driven_weight - self.initial_data_driven_weight) / self.max_epoch
        
        data_driven_weight = self.initial_data_driven_weight + step * self.current_epoch
        
        self.loss_module.data_driven_weight = data_driven_weight
        self.loss_module.mechanistic_weight = 1 - data_driven_weight
        
    def get_weights(self) -> dict:
        return {"data_driven": self.loss_module.data_driven_weight, "mechanistic": self.loss_module.mechanistic_weight}


class MultiPhaseScheduler(BaseLossWeightsScheduler):
    def __init__(self, max_epoch, loss_module: MechanisticLossWeighted, phase_shedulers: DictConfig):
        super().__init__(max_epoch, loss_module)
        
        self.schedulers_cfg_list = [phase_scheduler_cfg for _, phase_scheduler_cfg in sorted(phase_shedulers.items(), key=lambda item: item[0])]
        self.current_scheduler_index = None
        
        self.__check_epochs()
        
        self.__switch_scheduler()
    
    def __check_epochs(self):
        switch_phase_epochs = [phase_scheduler.epochs for phase_scheduler in self.schedulers_cfg_list]
        
        assert all(isinstance(x, int) for x in switch_phase_epochs), TypeError("The list epochs must contain all ints")
        assert sum(switch_phase_epochs) <= self.max_epoch, ValueError("The last epoch must be less than the maximum epoch.")
        
    def __make_phase_scheduler_instance(self, phase_sheduler_config, epochs: int):
        return get_loss_weight_scheduler(phase_sheduler_config, max_epoch=epochs, loss_module=self.loss_module)
    
    def adjust_weights(self):
        self.current_scheduler.adjust_weights()
    
    def __switch_scheduler(self):
        if self.current_scheduler_index is None:
            self.current_scheduler_index = 0
        else:
            self.current_scheduler_index += 1
        
        scheduler_cfg = self.schedulers_cfg_list[self.current_scheduler_index]
        
        if self.current_scheduler_index < len(self.schedulers_cfg_list)-1:
            scheduler_epochs = scheduler_cfg.epochs
        else:
            scheduler_epochs = self.max_epoch - self.current_epoch
        
        self.current_scheduler = self.__make_phase_scheduler_instance(scheduler_cfg, scheduler_epochs)
    
    def update_current_epoch(self, epoch):
        epochs_passed = epoch - self.current_epoch
        super().update_current_epoch(epoch)
        self.current_scheduler.update_current_epoch(self.current_scheduler.current_epoch+epochs_passed)
        
        if self.current_scheduler.current_epoch >= self.current_scheduler.max_epoch:
            self.__switch_scheduler()
    
    def get_weights(self) -> dict:
        return self.current_scheduler.get_weights()
    
    
# ----- get loss weight scheduler -----

def get_loss_weight_scheduler(scheduler_cfg: DictConfig, max_epoch: int, loss_module: Union[MechanisticLoss, MechanisticLossWeighted]):
    if scheduler_cfg.type == 'no_scheduler':
        return NoWeightsScheduler(max_epoch, loss_module)
    elif scheduler_cfg.type == 'constant':
        return ConstantWeightsScheduler(max_epoch, loss_module, **scheduler_cfg.params)
    elif scheduler_cfg.type == 'linear':
        return DataDrivenLinearScheduler(max_epoch, loss_module, **scheduler_cfg.params)
    elif scheduler_cfg.type == 'multi_phase':
        return MultiPhaseScheduler(max_epoch, loss_module, **scheduler_cfg.params)
    else:
        raise NotImplementedError(f'{scheduler_cfg.type} scheduler type not yet implemented')