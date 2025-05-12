import numpy as np
import numpy as np
import torch.nn as nn
import torch
from src.nn_model.amn_qp_old_code import *
from src.utils.import_GEM import *
import pandas as pd

from src.schedulers.loss_aggregator import LossAggregator

def normalized_error(true, pred, axis):
    error_norm = torch.norm(true - pred, dim=axis)
    true_norm = torch.norm(true, dim=axis)
    normalized_error = torch.nan_to_num(error_norm / true_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized_error

def get_loss(V, Vref, Pref, Pin, Vin, S, gradient=False, hyper_params=None, loss_aggregator: LossAggregator = None, l1_constant=None, data_driven_weight=1, mechanistic_weight=1):
    
    #2nd element of the loss (steady-state costraint)cle
    S = torch.from_numpy(np.float32(S)).to(hyper_params.model.device)
    SV = torch.matmul(V, S.T)
    L2 = (torch.norm(SV, dim=1, p=2)**2)/S.shape[0]

    #3rd element of the loss (upper bound costraint)
    Pin = torch.from_numpy(np.float32(Pin)).to(hyper_params.model.device)
    #Vin = torch.from_numpy(np.float32(Vin)).to(args.device)
    Vin_pred = torch.matmul(V, Pin.T)
    relu_Vin= torch.relu(Vin_pred - Vin)
    L3 = (torch.norm(relu_Vin, dim=1, p=2)**2)/Pin.shape[0]

    #4rd element of the loss (flux positivity costraint)
    relu_V = torch.relu(-V)
    L4 = (torch.norm(relu_V, dim=1, p=2)**2)/V.shape[1]

    if gradient == False:
        #1st element of the loss (to fit reference fluxes)
        Pref = torch.from_numpy(np.float32(Pref)).to(hyper_params.model.device)
        Vref_pred = torch.matmul(V, Pref.T)
        if hyper_params.model.l1_type=='MSE':
            L1 = (torch.norm((Vref_pred - Vref), dim=1, p=2)**2)/Pref.shape[0]
        elif hyper_params.model.l1_type == 'NE':
            L1= normalized_error(Vref, Vref_pred, axis=1)

        #TODO: da capire la normalizzazione delle loss se vogliamo usare un'altra loss rispetto a MSE
        #if hyper_params.model.model_name == 'amn_qp':
        #L = l1_constant*(L1) + (1-l1_constant)*(L2+L3+L4)
        if hyper_params.model.model_name in ['amn_qp', 'amn_qp_MSE']:
            if loss_aggregator is not None:
                L = loss_aggregator.aggregate_losses(losses=[L1, L2+L3+L4], weights=[data_driven_weight*l1_constant, mechanistic_weight])
            else:
                L = data_driven_weight*l1_constant*(L1) + mechanistic_weight*(L2+L3+L4)
        else:
            L=L1

    # when used by QP to refine the solution
    if gradient ==True:
        #TODO way to do this automatically
        # old MM_Qp_solver
        #dV1 = torch.matmul(Veb_pred - Veb, Peb)*(2/Pref.shape[0])
        dV2 = torch.matmul(SV, S)*(2/S.shape[0])

        dV3 = torch.where(relu_Vin != 0, 1, torch.zeros_like(relu_Vin))
        dV3 = (relu_Vin*dV3)
        dV3 = torch.matmul(dV3, Pin)*(2/Pin.shape[0]) 

        dV4 = torch.where(relu_V != 0, 1, torch.zeros_like(relu_V))
        dV4 = (relu_V*dV4)*(-2/V.shape[1])

        dV = dV2+dV3+dV4
        
        
        return dV
    
    # when used to backpropagate the error through the NN
    else:
        return L, [L1.sum(), L2.sum(), L3.sum(), L4.sum()]
    

class MechanisticLoss(nn.Module):
    def __init__(self, model, l1_constant, hyper_params):
        super(MechanisticLoss, self).__init__()
        # S [mxn]: stochiometric matrix
        # Pin [n_in x n]: to go from reactions to medium fluxes
        # Pref [n_ref x n]: to go from reactions to measured fluxes
        # Vin [n_batch x n_in]
        # V  [n_batch x n]
        # Vref  [n_batch x n_ref]
        self.Pref = model.Pref
        self.Pin = model.Pin
        self.S = model.S
        self.l1_constant = l1_constant
        self.hyper_params = hyper_params
       
    
    def forward(self, V, Vref, Vin):

        L, losses = get_loss(V, Vref, Pref = self.Pref, Pin=self.Pin, Vin=Vin, S=self.S, gradient=False, 
                             l1_constant=self.l1_constant, hyper_params=self.hyper_params)
        
        return L.sum(), losses
    

class MechanisticLossWeighted(MechanisticLoss):
    def __init__(self, model, l1_constant, hyper_params, loss_aggregator: LossAggregator, data_driven_weight=1, mechanistic_weight=1, data_drivern_loss_balance=1, mechanistic_loss_balance=1):
        super().__init__(model, l1_constant, hyper_params)
        
        self.loss_aggregator = loss_aggregator
        
        self.data_driven_weight = data_driven_weight
        self.mechanistic_weight = mechanistic_weight
        
        self.data_driven_loss_balance = data_drivern_loss_balance
        self.mechanistic_loss_balance = mechanistic_loss_balance
    
    def forward(self, V, Vref, Vin):

        L, losses = get_loss(V, Vref, Pref = self.Pref, Pin=self.Pin, Vin=Vin, S=self.S, gradient=False, loss_aggregator=self.loss_aggregator, 
                             l1_constant=self.l1_constant, hyper_params=self.hyper_params, 
                             mechanistic_weight=self.mechanistic_weight*self.mechanistic_loss_balance, 
                             data_driven_weight=self.data_driven_weight*self.data_driven_loss_balance)
        
        return L.sum(), losses


def Gradientdescent_QP(V0, Vref, Pref, Pin, Vin, S, lr=0.01, n_iteration=8, decay_rate=0.9, hyper_params=None):
    V = V0
    diff = 0
    for i in range(n_iteration):
        dV = get_loss(V, Vref, Pref, Pin, Vin, S, gradient=True, hyper_params=hyper_params)
        diff = decay_rate*diff - lr*dV
        V = V + diff

    return V



class MINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate, hyper_params, model):
        super(MINN, self).__init__()

        self.model = model
        self.hyper_params = hyper_params

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, output_size),
           )
        
    def forward(self, input, Vref, Vin):
        # output of pure NN
        V0 = self.layers(input)

        if self.hyper_params.model.model_type == 'AMN':
            Vout = Gradientdescent_QP(V0, Vref, self.model.Pref, self.model.Pin, Vin, self.model.S, 
                                      lr=self.hyper_params.model.qp_lr, 
                                      n_iteration=self.hyper_params.model.qp_iter, 
                                      decay_rate=self.hyper_params.model.qp_decay_rate,  
                                      hyper_params=self.hyper_params)
            return Vout, Vout
        else:
            return V0, V0




class PretrainedBlock(nn.Module):
    def __init__(self, model_GEM):
        super(PretrainedBlock, self).__init__()
        
        self.model_GEM = model_GEM
        
        self.nn = AMN_QP(input_size=5, hidden_size=500, output_size=587, drop_rate=0.25, model=self.model_GEM)
        
        # Load the pretrained state dictionary into self.nn
        self.nn.load_state_dict(torch.load("pretrained_block_reservoir_state_dict.pth"))
        
        # Assign self.nn to self.model so it can be called in forward
        self.model = self.nn
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, Vref, Vin):
        return self.model(x, Vref, Vin)  # Call self.model, which now refers to self.nn
        
        
class AMN_QP_reservoir(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate, hyper_params, model):
        super(AMN_QP_reservoir, self).__init__()

        self.model = model
        self.hyper_params = hyper_params
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, output_size),
           )
        
        self.pretrained_block = PretrainedBlock(self.model)
        

        
    def forward(self, input, Vref, Vin):
        # output of pure NN
        V0 = self.layers(input)
       
        
        Vout = self.pretrained_block(V0, Vref, Vin)
              
        return Vout, V0
        

if __name__ == "__main__":
    pass