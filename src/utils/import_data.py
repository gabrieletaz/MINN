import numpy as np
from .import_GEM import *
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class CustomTensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        x = self.data[0][index]
        y = self.data[1][index]
        Vin = self.data[2][index]

        return x, y, Vin



def load_ishii(seed=None, dataset=None, fluxes_removed_from_reference=None, fluxes_to_add_input= None, kos_genes=None, gem=None):
    
    # load csv
    if gem=='ecoli_core':
        fluxes = pd.read_csv('data/ishii_data/fluxomics_ecore_correct.csv', index_col='experiment')

    else:
        if dataset == 'ref_47_fluxes_fit':
            fluxes = pd.read_csv('data/ishii_data/fluxomics_iAF1260_reduced_split_fit.csv', index_col='experiment')
            
        elif dataset == 'fluxomics_iNF517_reduced_split': #TODO
            fluxes = pd.read_csv('data/ishii_data/fluxomics_iNF517_reduced_split.csv', index_col='experiment') #TODO
        else:
            fluxes = pd.read_csv('data/ishii_data/fluxomics_iAF1260_reduced_split_filtered_iNF517.csv', index_col='experiment')
        
    
    transcriptomics =  pd.read_csv('data/ishii_data/transcriptomics.csv', index_col='experiment')
    proteomics =  pd.read_csv('data/ishii_data/proteomics.csv', index_col='experiment')

    # Left join using indices
    data = pd.merge(transcriptomics, proteomics, on='experiment', how='left')
    data = pd.merge(data, fluxes[fluxes_to_add_input], on='experiment', how='left')

    if gem == 'iAF1260_reduced_split':
    # step 1: build/load and split model 
        model = import_GEM(filename='GEMs/iAF1260_reduced_split.xml', split=False)
    elif gem == 'ecoli_core':
        model = import_GEM(filename='GEMs/e_coli_core_ishii.xml', split=True)
    elif gem == 'iNF517_reducedFVA_splitted':
        model = import_GEM(filename='GEMs/iNF517_reducedFVA_splitted.xml', split=True)
    elif gem == 'iAF1260_reducedFVA_splitted':
        model = import_GEM(filename='GEMs/iAF1260_reducedFVA_splitted.xml', split=True)
    elif gem == 'iAF1260_split_FBA':
        model = import_GEM(filename='GEMs/iAF1260_split_FBA_reduction.xml', split=False)
    elif gem == 'iAF1260_full':
        model = import_GEM(filename='GEMs/iAF1260.xml', split=True)
        

    # step 2: initialize object GEM which will have all the matrices as attributes
    model = GEM(model=model)
    #print(len(model.reactions))
    
    
    #TODO add import KOs dataframe and a if to decide with a hydra parameter
    if kos_genes==False:
        Vin = pd.DataFrame({'InfColumn': np.inf}, index=data.index)
        input_ub = ['R_EX_co2_e_fwd']
    else:
        Vin = pd.read_csv('data/ishii_data/KOs_Vin.csv', index_col='experiment')
        Vin.replace(1, np.inf, inplace=True)
        input_ub = Vin.columns
        
    fluxes_ref = fluxes.drop(columns=fluxes_removed_from_reference)
    reference =  fluxes_ref.columns

    # step 4 : extract matrices Pin, Pred, S
    S, Pin, Pref = model.build_GEM_matrices(input_ub, reference)

    #input training
    X = data

    #target training
    y = fluxes[reference]

    #train and test split
    X, y, Vin = X.values.astype(np.float32), y.values.astype(np.float32), Vin.values.astype(np.float32)

    return X, y, Vin, model, reference


        
def whole_flux_distribution():
        
    fluxes = pd.read_csv('data/ishii_data/fluxomics_ecore_correct.csv', index_col='experiment')
    transcriptomics =  pd.read_csv('data/ishii_data/transcriptomics.csv', index_col='experiment')
    proteomics =  pd.read_csv('data/ishii_data/proteomics.csv', index_col='experiment')

    # Left join using indices
    data = pd.merge(transcriptomics, proteomics, on='experiment', how='left')
    data = pd.merge(data, fluxes[['R_EX_glc__D_e_rev', 'R_EX_o2_e_rev']], on='experiment', how='left')

    Vin = pd.DataFrame({'InfColumn': np.inf}, index=data.index)

    # step 1: build/load and split model
    model = import_GEM(filename='GEMs/e_coli_core.xml', split=True)

    # step 2: initialize object GEM which will have all the matrices as attributes
    model = GEM(model=model)


    input_ub = ['R_EX_co2_e_fwd']

    reference =  fluxes.columns

    # step 4 : extract matrices Pin, Pred, S
    S, Pin, Pref = model.build_GEM_matrices(input_ub, reference)

    #input training
    X = data

    #target training
    y = fluxes

    #train and test split
    X, y, Vin = X.values.astype(np.float32), y.values.astype(np.float32), Vin.values.astype(np.float32)

    return X, y, Vin, model, reference




if __name__ == "__main__":
    pass
