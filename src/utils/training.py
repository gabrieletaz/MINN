from clearml import Task
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from src.nn_model.amn_qp import *
from src.schedulers.loss_aggregator import get_loss_aggregator
from src.schedulers.loss_balancer import LossBalancer, get_loss_balancer
from src.schedulers.weights_schedulers import get_loss_weight_scheduler
from src.utils.import_data import *
from src.utils.import_GEM import *
from src.utils.loggers import DataFrameLogger
from src.utils.utils import assign_hyperparams_to_confg, log_losses_balanced_to_clearml, log_losses_to_clearml, log_losses_with_l1_constant_to_clearml, log_weights_to_clearml, plot_modified_loss_with_aggregate

def train_step(model, criterion, optimizer, train_loader, loss_balancer: LossBalancer=None, device='cpu'):
    model.train()
    loss_tot = 0
    len_train = 0
    Vref_pred = []
    Vref_true = []
    losses_n = np.zeros(4)
    for i, (x, Vref, Vin) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward pass
        x, Vref, Vin = x.to(device), Vref.to(device), Vin.to(device)
        # V = Vref predicted
        V, V0 = model(x, Vref, Vin)
        for i in range(V.size(0)):
            Vref_pred.append(V[i].tolist())
            Vref_true.append(Vref[i].tolist())
        # compute loss
        loss, losses = criterion(V, Vref, Vin)
        # back-prop
        loss.backward()
        optimizer.step()
        # gather statistics
        loss_tot += loss.item()
        losses_n = losses_n + np.array([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
        len_train += x.size(0)
        
        if loss_balancer is not None:
            loss_balancer.balance_losses([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
        
    return {'loss': loss_tot/len_train, 'Vref_pred':Vref_pred, 'losses': losses_n/len_train, 'Vref_true': Vref_true}



def test_step(model, criterion, test_loader, device='cpu'):
    model.eval()
    loss_tot = 0
    len_test = 0
    Vref_pred = []
    Vref_true = []
    Vin_all = []
    Vin_reservoir = []
    losses_n = np.zeros(4)
    with torch.no_grad():
        for i, (x, Vref, Vin) in enumerate(test_loader):
            # forward pass
            x, Vref, Vin = x.to(device), Vref.to(device), Vin.to(device)
            # V = Vref predicted
            V, V0 = model(x, Vref, Vin)
            for i in range(V.size(0)):
                Vref_pred.append(V[i].tolist())
                Vref_true.append(Vref[i].tolist())
                Vin_all.append(Vin[i].tolist())
                Vin_reservoir.append(V0[i].tolist())
            loss, losses = criterion(V, Vref, Vin)
            # gather statistics
            loss_tot += loss.item()
            losses_n = losses_n + np.array([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
            len_test += x.size(0)
    return {'loss': loss_tot/len_test, 'Vref_pred':Vref_pred, 'losses': losses_n/len_test, 'Vref_true': Vref_true, 'Vin':Vin_all,
            'Vin_reservoir': Vin_reservoir}


# kfold evaluation
def kfold_evaluation(cfg, task: Task, X, y, Vin, n_distribution, fit_model, params, loo_count=None):
    # Initialize KFold
    kf = KFold(n_splits=cfg.hpo.n_fold, shuffle=True, random_state=cfg.seed)

    # Lists to store results for each fold
    fold_losses = []

    # Perform k-fold cross-validation
    for fold, (train_idx_fold, val_idx_fold) in enumerate(kf.split(X, y)):

        # Split data into training and validation sets
        X_train_fold, X_val_fold = X[train_idx_fold], X[val_idx_fold]
        y_train_fold, y_val_fold = y[train_idx_fold], y[val_idx_fold]
        Vin_train_fold, Vin_val_fold = Vin[train_idx_fold], Vin[val_idx_fold]

        # Train and evaluate model
        results = train_test_evaluation(cfg, task, X_train_fold, X_val_fold, y_train_fold, y_val_fold, Vin_train_fold, Vin_val_fold, n_distribution, fit_model, params, fold_id=fold, loo_count=loo_count)
        
        #RMSE_value_fold = np.sqrt(np.mean(np.square(Vref_true-Vref_pred_te)))
        #print(f'Fold_{fold} TRAIN losses: {results["train"]["losses"]}')
        #print(f'Fold_{fold} TEST losses: {results["test"]["losses"]}')
        fold_losses.append(results["test"]['losses'][0])

    return fold_losses

def train_test_evaluation(cfg, task: Task, X_train, X_test, y_train, y_test, Vin_train, Vin_test, n_distribution, fit_model, params, fold_id=None, loo_count=None):
    
    assign_hyperparams_to_confg(cfg, params)
    
    # Initialize AMN_QP module
    if cfg.model.model_reservoir:
        
        amn_qp = AMN_QP_reservoir(input_size=X_train.shape[1], hidden_size=params['hidden_size'], output_size=5, 
                    drop_rate= params['drop_rate'], hyper_params = cfg, model=fit_model).to(cfg.model.device)
    else:
        amn_qp = MINN(input_size=X_train.shape[1], hidden_size=params['hidden_size'], output_size=n_distribution, 
                    drop_rate= params['drop_rate'], hyper_params = cfg, model=fit_model).to(cfg.model.device) 
           
    
    loss_aggregator = get_loss_aggregator(cfg.loss_aggregator)
    if cfg.loss_aggregator.type == "double_linear_bound" and loo_count == 1 and fold_id is None:
        plot_modified_loss_with_aggregate(task, loss_aggregator)
        
    criterion = MechanisticLossWeighted(fit_model, params['l1_constant'], cfg, loss_aggregator=loss_aggregator)

    # Define optimizer for backpropagation
    optimizer = torch.optim.Adam(amn_qp.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    # Basic Data Preprocessing
    sc_X_fold = MinMaxScaler()
    #sc_y_fold = MinMaxScaler()
    X_train = sc_X_fold.fit_transform(X_train)
    #y_train_fold = sc_y_fold.fit_transform(y_train_fold)
    X_test = sc_X_fold.transform(X_test)
    #y_val_fold = sc_y_fold.transform(y_val_fold)

    # Create data loaders for the training and validation sets
    train_dataset = CustomTensorDataset(data=(X_train, y_train, Vin_train))
    valid_dataset = CustomTensorDataset(data=(X_test, y_test, Vin_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.hpo.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.hpo.batch_size)

    # best epoch is chosen in based on the first fold metric
    loss_balancer = get_loss_balancer(cfg.loss_balancer, criterion)
    loss_weight_scheduler = get_loss_weight_scheduler(cfg.loss_weight_scheduler, cfg.hpo.epochs, criterion)
    
    for epoch in range(cfg.hpo.epochs):
        # update epoch and loss weights
        loss_weight_scheduler.update_current_epoch(epoch)
        loss_weight_scheduler.adjust_weights()
        
        tr_loss = train_step(amn_qp, criterion, optimizer, train_loader, loss_balancer=loss_balancer, device=cfg.model.device)
        
        if task != None:
            if fold_id is None:
                # log losses and weights to clearml
                log_losses_to_clearml(task, epoch, tr_loss["losses"], loo_counter=loo_count)
                if criterion.l1_constant != 1:
                    log_losses_with_l1_constant_to_clearml(task, epoch, criterion.l1_constant, tr_loss["losses"], loo_counter=loo_count)
                if cfg.loss_balancer.type != 'no_balancer':
                    log_losses_balanced_to_clearml(task, epoch, loss_balancer, tr_loss["losses"], loo_counter=loo_count)
                log_weights_to_clearml(task, epoch, loss_weight_scheduler.get_weights(), loo_counter=loo_count)
            
    te_loss = test_step(amn_qp, criterion, valid_loader, device=cfg.model.device)
    
    return {"train": tr_loss, "test": te_loss}