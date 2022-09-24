import numpy as np
import pickle
 
class Logger():
    def __init__(self, S, A, H, P, R, gap_min, algO_alpha, algP_alpha, log_path):
        self.log_path = log_path
        self.doc = {
            'S': S,
            'A': A,
            'H': H,
            'P': P,
            'R': R,
            'gap_min': gap_min,
            'algO_alpha': algO_alpha,
            'algP_alpha': algP_alpha,
            'results': []
        }
 
    def update_info(self, iter, R_algO, R_algP):
        self.doc['results'].append((iter, R_algO, R_algP))
 
    def dump(self):
        with open(self.log_path, 'wb') as f:
            pickle.dump(self.doc, f)