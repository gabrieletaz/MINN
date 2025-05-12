import numpy as np
import cbmpy


def import_GEM(filename='', split=False):
    model = cbmpy.CBRead.loadModel(filename)
    # splits reversible reations into reaction_fwd and reaction_rev

    if split == True:
        model = cbmpy.CBTools.splitReversibleReactions(model)
        model.buildStoichMatrix()

    return model

def get_index_from_id(name, list_id):
        try:
            return list_id.index(name)
        except ValueError:
            raise ValueError(f"The id '{name}' is not present in the list of reactions")


class GEM:
    def __init__(self, model):
        self.model = model

        #list of all reactions in the model
        self.reactions = self.model.getReactionIds()
    
    def build_GEM_matrices(self, input_ub, reference):
        # Get matrices for AMN_QP 
        # Return
        # - S [mxn]: stochiometric matrix
        # - Pin [n_in x n]: to go from reactions to medium fluxes
        # - Pref [n_ref x n]: to go from reactions to measured fluxes
        # m = metabolite, n = reaction/v/flux, p = medium

        #list of reaction with ub in the loss
        self.input_ub = input_ub

        #list of reaction to fit in the loss
        self.reference = reference
        
        #export stoichioemetric matrix in a np.array
        self.S = self.model.N.array

        # n = n_reactions, n_in = n_bounded reactions , n_ref = n_reactions to fit
        n, n_in, n_ref = self.S.shape[1], len(input_ub), len(reference)

        # Projection matrix from V to Vin
        Pin = np.zeros((n_in,n))
        for i, rid in enumerate(input_ub):
            j = get_index_from_id(rid, self.reactions)
            Pin[i][j] = 1
        self.Pin = Pin

        # Projection matrix from V to Vref
        Pref = np.zeros((n_ref,n))
        for i, rid in enumerate(reference):
            j = get_index_from_id(rid, self.reactions)
            Pref[i][j] = 1

        self.Pref = Pref

        return self.S, self.Pin, self.Pref


if __name__ == "__main__":
    pass
