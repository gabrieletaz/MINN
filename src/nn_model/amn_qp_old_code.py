import numpy as np
import numpy as np
import torch.nn as nn
import torch
from src.utils.args import *

def normalized_error(true, pred, axis, l1_constant):
    error_norm = torch.norm(true - pred, dim=axis)
    true_norm = torch.norm(true, dim=axis)
    
    normalized_error = torch.nan_to_num(error_norm / true_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    #per avere le loss che pesano come con MSE(massimo nel csv fluxes)
    return normalized_error*l1_constant

def get_loss(V, Vref, Pref, Pin, Vin, S, gradient=False, l1_constant=None):
    
    #2nd element of the loss (steady-state costraint)cle
    S = torch.from_numpy(np.float32(S)).to(args.device)
    SV = torch.matmul(V, S.T)
    L2 = (torch.norm(SV, dim=1, p=2)**2)/S.shape[0]

    #3rd element of the loss (upper bound costraint)
    Pin = torch.from_numpy(np.float32(Pin)).to(args.device)
    #Vin = torch.from_numpy(np.float32(Vin)).to(args.device)
    Vin_pred = torch.matmul(V, Pin.T)
    #print(Vin_pred)
    relu_Vin= torch.relu(Vin_pred - Vin)
    L3 = (torch.norm(relu_Vin, dim=1, p=2)**2)/Pin.shape[0]

    #4rd element of the loss (flux positivity costraint)
    relu_V = torch.relu(-V)
    L4 = (torch.norm(relu_V, dim=1, p=2)**2)/V.shape[1]

    if gradient == False:
        #1st element of the loss (to fit reference fluxes)
        Pref = torch.from_numpy(np.float32(Pref)).to(args.device)
        Vref_pred = torch.matmul(V, Pref.T)
        if args.L1=='MSE':
        #print(f'Vref_pred: {Vref_pred.shape}  |  Vref: {Vref.shape}')
            L1 = (torch.norm((Vref_pred - Vref), dim=1, p=2)**2)/Pref.shape[0]
        elif args.L1 == 'NE':
            L1= normalized_error(Vref, Vref_pred, axis=1, l1_constant=l1_constant)
        else: 
            L1 = (torch.mean(torch.abs(Vref- Vref_pred), dim=1)) / Pref.shape[0]

        L = L1+L2+L3+L4

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
    def __init__(self, model, l1_constant):
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
    
    def forward(self, V, Vref, Vin):

        L, losses = get_loss(V, Vref, Pref = self.Pref, Pin=self.Pin, Vin=Vin, S=self.S, gradient=False, l1_constant=self.l1_constant)
        
        return L.sum(), losses
    

def Gradientdescent_QP(V0, Vref, Pref, Pin, Vin, S, lr=0.01, n_iteration=args.Qp_iteration, decay_rate=0.9):
    V = V0
    diff = 0
    for i in range(n_iteration):
        dV = get_loss(V, Vref, Pref, Pin, Vin, S, gradient=True)
        diff = decay_rate*diff - lr*dV
        V = V + diff

    return V



class AMN_QP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate, model):
        super(AMN_QP, self).__init__()

        self.model = model

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
        
        self.out = nn.ReLU()
        
    def forward(self, input, Vref, Vin):
        # output of pure NN
        V0 = self.layers(input)

        if args.model == 'AMN':
            Vout = Gradientdescent_QP(V0, Vref, self.model.Pref, self.model.Pin, Vin, self.model.S)
            #Vout = self.out(Vout)
            return Vout
        else:
            return V0


if __name__ == "__main__":
    pass