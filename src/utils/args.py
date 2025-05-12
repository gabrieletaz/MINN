import torch
import argparse


# ----- Parser -----

import argparse
import sys

import argparse
import torch

def parser():
    PARSER = argparse.ArgumentParser(description='Training parameters.')

    PARSER.add_argument('--Qp_iteration', default=8, type=int,
                        help='Number of Qp iteration.')
    
    PARSER.add_argument('--prior_knowledge', default=False, type=bool,
                        help='Add L1 to mechanistic LOSS')
    
    PARSER.add_argument('--exp_name', default='test', type=str,
                        help='Name of the experiment.')

    PARSER.add_argument('--L1', default='MSE', type=str,
                        choices=['MSE', 'MAE', 'NE'],
                        help='L1 to use.')

    PARSER.add_argument('--model', default='AMN', type=str,
                        choices=['AMN', 'NN'],
                        help='NN model to use.')
    
    PARSER.add_argument('--device', default='cpu', type=str,
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], 
                        help='Device to run the experiment.')

    # Parse known arguments and ignore any unknown arguments
    ARGS, unknown = PARSER.parse_known_args()

    # Automatically select device if none is specified
    if ARGS.device == 'cpu' and torch.cuda.is_available():
        ARGS.device = 'cuda:0'

    return ARGS

args = parser()
print(args)
