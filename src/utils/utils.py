from clearml import Task
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import json

from src.schedulers.loss_aggregator import LinearBoundLossAggregator

# ----- Random Seed Control -----

def fix_random_seed(seed=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# ----- log losses to ClearML -----

def log_losses_to_clearml(task: Task, epoch: int, losses: list, losses_names: list=['data_driven', 'stedy_state', 'upper_bound', 'flux_positivity'], loo_counter: int=None):
    for loss_name, loss in zip(losses_names, losses):
        task.get_logger().report_scalar(title="Losses" if loo_counter is None else f"Losses - LOO observation n. {loo_counter:2d}", series=Prittifier().prettify_string(loss_name), value=loss, iteration=epoch)
    task.get_logger().report_scalar(title='Epoch', series='Epoch', value=epoch, iteration=epoch)

# ----- log losses with l1 constant to ClearML -----

def log_losses_with_l1_constant_to_clearml(task: Task, epoch: int, l1_constant, losses: list, losses_names: list=['data_driven', 'stedy_state', 'upper_bound', 'flux_positivity'], loo_counter: int=None):
    losses[0] = losses[0] * l1_constant
    for loss_name, loss in zip(losses_names, losses):
        task.get_logger().report_scalar(title="Losses with l1 constant" if loo_counter is None else f"Losses with l1 constant - LOO observation n. {loo_counter:2d}", series=Prittifier().prettify_string(loss_name), value=loss, iteration=epoch)
    task.get_logger().report_scalar(title='Epoch', series='Epoch', value=epoch, iteration=epoch)

# ----- log losses balanced to ClearML -----

def log_losses_balanced_to_clearml(task: Task, epoch: int, loss_balancer, losses: list, losses_names: list=['data_driven', 'mechanistic'], loo_counter: int=None):
    data_driven_loss = losses[0] * loss_balancer.get_weights()["data_driven"]
    mechanistic_loss = sum(losses[1:]) * loss_balancer.get_weights()["mechanistic"]
    for loss_name, loss in zip(losses_names, [data_driven_loss, mechanistic_loss]):
        task.get_logger().report_scalar(title="Losses balanced" if loo_counter is None else f"Losses balanced - LOO observation n. {loo_counter:2d}", series=Prittifier().prettify_string(loss_name), value=loss, iteration=epoch)
    task.get_logger().report_scalar(title='Epoch', series='Epoch', value=epoch, iteration=epoch)

# ----- log weights to ClearML -----

def log_weights_to_clearml(task: Task, epoch: int, weights: dict, loo_counter: int=None):
    for weight_name, weight in weights.items():
        task.get_logger().report_scalar(title="Weights" if loo_counter is None else f"Weights - LOO observation n. {loo_counter:2d}", series=Prittifier().prettify_string(weight_name), value=weight, iteration=epoch)

# ----- log hyperparameters to ClearML -----

def get_best_hyperparameters_from_previous_task(task_id, artifact_name='best_hyperparameters'):
    task: Task = Task.get_task(task_id=task_id)

    # Get the artifact
    local_json = task.artifacts[artifact_name].get_local_copy()
    
    with open(local_json, 'r') as handle:
        best_param_dict = json.load(handle)
    
    # turn keys to int
    best_param_dict = {int(k):v for k,v in best_param_dict.items()}
    
    return best_param_dict

# ----- log plot of modified loss after bound -----

def plot_modified_loss_with_aggregate(task: Task, aggregator_with_bound: LinearBoundLossAggregator):
    bound = aggregator_with_bound.mechanistic_bound
    
    # Generate loss values before and after the bound
    losses_before_bound = torch.linspace(0, bound, 100)
    losses_after_bound = torch.linspace(bound, bound*3, 20 + 1)[1:]  # skip the first to avoid duplicate
    losses = torch.cat([losses_before_bound, losses_after_bound])

    # Calculate modified losses using aggregate_losses
    modified_losses = aggregator_with_bound.aggregate_losses([torch.zeros_like(losses), losses], [0.0, 1.0]).numpy()
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.plot(losses.numpy(), losses.numpy(), label='Original Loss', linestyle='--')
    plt.plot(losses.numpy(), modified_losses, label='Modified Loss')
    plt.axvline(bound, linestyle='--', alpha=0.5, color="red", label='Bound')
    plt.xlabel('Loss Value')
    plt.ylabel('Loss Output')
    plt.title(f'Loss Modification with Bound={bound} and Exp Base={aggregator_with_bound.mechanistic_coefficient}')
    plt.legend()
    
    task.get_logger().report_matplotlib_figure("Loss modification after bound", "Loss modification after bound", plt, 0, report_image=True)
    

class Prittifier:
    def prettify_string(self, string):
        # Replace underscores with spaces
        modified_string = string.replace('_', ' ')
        # Capitalize the first letter of the modified string
        pretty_string = modified_string.capitalize()
        return pretty_string
    
    def prittify_list_of_strings(self, list_of_strings):
        return [self.prettify_string(string) for string in list_of_strings]
    
    def prittify_column_names(self, df) -> pd.DataFrame:
        df.columns = self.prittify_list_of_strings(df.columns)
        return df
    

def assign_hyperparams_to_confg(cfg, params):
    if "data_driven_weight" in params.keys():
        data_driven_weight = params['data_driven_weight']
        mechanistic_weight = params['mechanistic_weight'] if "mechanistic_weight" in params.keys() else 1 - data_driven_weight
        
        if cfg.loss_weight_scheduler.type == "constant":
            cfg.loss_weight_scheduler.data_driven_weight = data_driven_weight
            cfg.loss_weight_scheduler.mechanistic_weight = mechanistic_weight
        elif cfg.loss_weight_scheduler.type == "multi_phase":
            cfg.loss_weight_scheduler.params.phase_shedulers.phase_2.params.final_data_driven_weight = data_driven_weight
            cfg.loss_weight_scheduler.params.phase_shedulers.phase_3.params.data_driven_weight = data_driven_weight
            cfg.loss_weight_scheduler.params.phase_shedulers.phase_3.params.mechanistic_weight = mechanistic_weight
        else:
            raise Warning("Data driven weight is not supported for this loss weight scheduler")
    
    if "mechanistic_bound" in params.keys():
        if cfg.loss_aggregator.type == "double_linear_bound":
            cfg.loss_aggregator.params.mechanistic_bound = params['mechanistic_bound']
        else:
            raise Warning("Mechanistic bound is not supported for this loss aggregator")