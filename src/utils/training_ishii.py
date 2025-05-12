import numpy as np
import torch
import torch.nn as nn
from src.nn_model.amn_qp import *
from src.utils.import_data import *
from src.utils.import_GEM import *
from src.utils.args import *
import optuna

def train_step(model, criterion, optimizer, train_loader, device=args.device):
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
        V = model(x, Vref, Vin)
        for i in range(V.size(0)):
            Vref_pred.append(V[i].tolist())
            Vref_true.append(Vref[i].tolist())
        loss, losses = criterion(V, Vref, Vin)
        # back-prop
        loss.backward()
        optimizer.step()
        # gather statistics
        loss_tot += loss.item()
        losses_n = losses_n + np.array([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
        len_train += x.size(0)
    return {'loss': loss_tot/len_train, 'Vref_pred':Vref_pred, 'losses': losses_n/len_train, 'Vref_true': Vref_true}



def test_step(model, criterion, test_loader, device=args.device):
    model.eval()
    loss_tot = 0
    len_test = 0
    Vref_pred = []
    Vref_true = []
    Vin_all = []
    losses_n = np.zeros(4)
    with torch.no_grad():
        for i, (x, Vref, Vin) in enumerate(test_loader):
            # forward pass
            x, Vref, Vin = x.to(device), Vref.to(device), Vin.to(device)
            # V = Vref predicted
            V = model(x, Vref, Vin)
            for i in range(V.size(0)):
                Vref_pred.append(V[i].tolist())
                Vref_true.append(Vref[i].tolist())
                Vin_all.append(Vin[i].tolist())
            loss, losses = criterion(V, Vref, Vin)
            # gather statistics
            loss_tot += loss.item()
            losses_n = losses_n + np.array([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
            len_test += x.size(0)
    return {'loss': loss_tot/len_test, 'Vref_pred':Vref_pred, 'losses': losses_n/len_test, 'Vref_true': Vref_true, 'Vin':Vin_all}


def kfold_cv(trial, seed):
    torch.manual_seed(seed)
    # define search space for tuning hyperparameters
    
    params = { 
            'learning_rate':trial.suggest_categorical('learning_rate', [0.0001]),
            'dim': trial.suggest_int('dim', 200, 200, step=64),
            'dropout': trial.suggest_categorical('dropout1', [0.1, 0.3, 0.5]),
            'batch_size': trial.suggest_int('batch_size', 32, 64, step=16),
            }

    max_epochs = 200
    # Loss function
    criterion = nn.MechanisticLoss()

    n_splits = 5
    # define splits for the k-fold
    splits= KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # scores of a configuration of hyperparameters on each fold
    scores_tr = []
    scores_val = []
    #scores_per_epoch = {'train': np.zeros(n_epoch), 'val': np.zeros(n_epoch)}
    for fold, (train_idx, val_idx) in enumerate(splits.split(data_1[0][:][0], data_1[0][:][1])):

        np.random.shuffle(train_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = val_idx
        loader_kwargs = {'batch_size':params['batch_size']}

        # omic1
        train_loader_1 = DataLoader(data_1[0], **loader_kwargs, sampler= train_sampler)
        val_loader_1  = DataLoader(data_1[0], **loader_kwargs, sampler= valid_sampler)  


        # best epoch is chosen in based on the first fols metric
        if fold == 0:
            epochs = max_epochs
        # best epoch for the other folds 
        else:
            epochs = best_epoch


        patience = 15
        count_patience = 0
        best_loss = np.inf
        for epoch in range(1, epochs+1):
            train_stats = train_step(model, criterion, optimizer, train_loader)
            valid_stats = valid_step(model, criterion, val_loader)
            #print(valid_stats['loss'])
            if fold == 0:
                if valid_stats['loss'] < best_loss:
                    if epoch >20:
                        best_loss = valid_stats['loss']
                        best_train_acc = train_stats['accuracy']
                        best_val_acc = valid_stats['accuracy']
                        best_epoch = epoch
                        count_patience = 0

                else:
                    count_patience += 1
                
                if count_patience == patience:
                    break
            else:

                best_train_acc = train_stats['accuracy']
                best_val_acc = valid_stats['accuracy']

        trial.suggest_int("best_epoch", best_epoch , best_epoch)
        scores_tr.append(best_train_acc)
        scores_val.append(best_val_acc)

    # average of metric on the k folds for 1 set of hyperparameters
    return np.mean(scores_val)




if __name__ == "__main__":
    pass