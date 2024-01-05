import numpy as np
from icecream import ic

class Consensus_Variables:
    '''Implementation following Anthony F. Genovese: "The Interacting Multiple Model Algorithm for
        Accurate State Estimation of Maneuvring Targets." '''
    
    def __init__(self):
        self.a_ca = 0
        self.a_cv = 0
        self.a_ct = 0

        self.F_ca = 0
        self.F_cv = 0
        self.F_ct = 0

