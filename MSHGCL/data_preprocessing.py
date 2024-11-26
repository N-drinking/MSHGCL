from operator import index
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

'''
The D-MPNN takes line graphs instead of node graphs used in common GNN as input. 
This is because the D-MPNN operates on edges/bonds instead of nodes. 
So we have to convert the node graph to the line graph.
'''

class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''
    def __inc__(self, key, value, *args, **kwargs):
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"
