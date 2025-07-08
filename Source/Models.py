#==========================================================================
# Import: Libraries
#==========================================================================

import math
import time
import argparse
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

import torch
import torch.autograd as autograd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from _dependency.ising import Ising





#==========================================================================
# Define: 2D Ising Model
#==========================================================================

class Ising2D():
    def __init__(self, size):
        super(Ising2D, self).__init__()
        
        
        self.size = size
        self.P_mat = self._generate_edge_mat(self.size)
        self.sample_model = Ising(self.size**2)
    
    
    def sample(self, param, num_sample, *args):
        self.sample_model.set_ferromagnet(self.size, float(param))
        return torch.Tensor(self.sample_model.sample(num_iters=1e5, num_samples=num_sample))
    
    
    def _generate_edge_mat(self, dim):
        padmat = torch.zeros(dim, dim, dim, dim)
        for i in range(dim):
            for j in range(dim):
                if not i - 1 == -1:
                    padmat[i][j][i-1, j] = 1
                if not j - 1 == -1:
                    padmat[i][j][i, j-1] = 1
                if not i + 1 == dim:
                    padmat[i][j][i+1, j] = 1
                if not j + 1 == dim:
                    padmat[i][j][i, j+1] = 1
        return padmat.reshape(dim*dim, dim*dim)
    
    
    def stat_m(self, X):
        return - 2 * X * ( X @ self.P_mat )
        
        
    def stat_p(self, X):
        return 2 * X * ( X @ self.P_mat )
        
        
    def ratio_m(self, param, SX_m):
        gamma = param[0]  # horizontal parameter
        beta_c = param[1:]  # vertical parameters
        
        # Apply parameters to respective statistics
        h_term = torch.exp(SX_m[:, 0] / gamma)  # horizontal term
        
        # For vertical term, we need to handle each column separately
        v_terms = []
        for c in range(self.C):
            v_terms.append(torch.exp(SX_m[:, 1] / beta_c[c]))
        v_term = torch.stack(v_terms, dim=1).prod(dim=1)  # multiply all vertical terms
        
        return h_term * v_term
    
    
    def ratio_p(self, param, SX_p):
        gamma = param[0]  # horizontal parameter
        beta_c = param[1:]  # vertical parameters
        
        # Apply parameters to respective statistics
        h_term = torch.exp(SX_p[:, 0] / gamma)  # horizontal term
        
        # For vertical term, we need to handle each column separately
        v_terms = []
        for c in range(self.C):
            v_terms.append(torch.exp(SX_p[:, 1] / beta_c[c]))
        v_term = torch.stack(v_terms, dim=1).prod(dim=1)  # multiply all vertical terms
        
        return h_term * v_term
    
    
    def uloglikelihood(self, param, X):
        return torch.sum( X * ( X @ self.P_mat ) ) / param
    
    
    def shift_p(self, X, state=torch.tensor([-1.0,1.0]), diff=2.0):
        X_shift = ( X.unsqueeze(-1) + torch.eye(X.shape[1])*diff ).transpose(1, 2)
        return torch.where((X_shift==state[-1]+diff), state[0], X_shift).transpose(0, 1)
    
    
    def pseudologlikelihood(self, param, X):
        tmp_a = torch.exp( - X @ self.P_mat / param )
        tmp_b = torch.exp( X @ self.P_mat / param )
        tmp_Xa = tmp_a * ( ( X - 1 ).abs() / 2 )
        tmp_Xb = tmp_b * ( ( X + 1 ).abs() / 2 )
        tmp = ( tmp_Xa + tmp_Xb ) / ( tmp_a + tmp_b ) 
        return torch.sum( tmp.log() , axis = 1 )

#==========================================================================
# Define: Anisotropic 2D Ising Model
#==========================================================================

class IsingAnisotropic():
    def __init__(self, n, C):
        super(IsingAnisotropic, self).__init__()
        
        
        self.n = n  # rows
        self.C = C  # columns
        self.size = n * C
        self.H_mat, self.V_mats = self._generate_edge_matrices(n, C)
        self.sample_model = Ising(self.size)
    
    
    def sample(self, param, num_sample, *args):
        # Pass the full parameter array for anisotropic case
        self.sample_model.set_ferromagnet([self.n, self.C], param, isotropic=False)
        return torch.Tensor(self.sample_model.sample(num_iters=1e5, num_samples=num_sample))
    
    
    def _generate_edge_matrices(self, n, C):
        # Horizontal connections matrix (one parameter)
        H_mat = torch.zeros(n, C, n, C)
        
        # Vertical connections matrices (C parameters, one per column)
        V_mats = torch.zeros(C, n, C, n, C)
        
        for i in range(n):
            for c in range(C):
                # Horizontal neighbors (left-right within same row)
                if c > 0:  # left neighbor
                    H_mat[i][c][i, c-1] = 1
                if c < C - 1:  # right neighbor
                    H_mat[i][c][i, c+1] = 1
                
                # Vertical neighbors for each column c
                if i > 0:  # up neighbor
                    V_mats[c][i][c][i-1, c] = 1
                if i < n - 1:  # down neighbor
                    V_mats[c][i][c][i+1, c] = 1
        
        # Reshape to adjacency matrices
        H_mat_flat = H_mat.reshape(n*C, n*C)
        V_mats_flat = V_mats.reshape(C, n*C, n*C)
        
        return H_mat_flat, V_mats_flat
    
    
    def _compute_interactions(self, X, param):
        """
        Compute weighted interactions using:
        - param[0] (gamma) for horizontal direction
        - param[1:C+1] (beta_c) for vertical directions
        """
        gamma = param[0]  # horizontal parameter
        beta_c = param[1:]  # vertical parameters
        
        # Horizontal interactions (single parameter gamma)
        h_interactions = X @ self.H_mat * gamma
        
        # Vertical interactions (C parameters beta_c)
        v_interactions = torch.zeros_like(X)
        for c in range(self.C):
            v_interactions += (X @ self.V_mats[c]) * beta_c[c]
        
        return h_interactions + v_interactions
    
    
    def stat_m(self, X):
        # Separate horizontal and vertical interactions
        h_interactions = X @ self.H_mat
        v_interactions = torch.zeros_like(X)
        for c in range(self.C):
            v_interactions += X @ self.V_mats[c]
            
        # Return separate statistics for each dimension
        h_stat = -2 * X * h_interactions  # horizontal statistics
        v_stat = -2 * X * v_interactions  # vertical statistics
        
        return torch.stack([h_stat, v_stat], dim=1)  # shape: (n_samples, 2)
        
        
    def stat_p(self, X):
        # Separate horizontal and vertical interactions
        h_interactions = X @ self.H_mat
        v_interactions = torch.zeros_like(X)
        for c in range(self.C):
            v_interactions += X @ self.V_mats[c]
            
        # Return separate statistics for each dimension
        h_stat = 2 * X * h_interactions  # horizontal statistics
        v_stat = 2 * X * v_interactions  # vertical statistics
        
        return torch.stack([h_stat, v_stat], dim=1)  # shape: (n_samples, 2)
        
        
    def ratio_m(self, param, SX_m):
        gamma = param[0]  # horizontal parameter
        beta_c = param[1:]  # vertical parameters
        
        # Apply parameters to respective statistics
        h_term = torch.exp(SX_m[:, 0] * gamma)  # horizontal term
        
        # For vertical term, we need to handle each column separately
        v_terms = []
        for c in range(self.C):
            v_terms.append(torch.exp(SX_m[:, 1] * beta_c[c]))
        v_term = torch.stack(v_terms, dim=1).prod(dim=1)  # multiply all vertical terms
        
        return h_term * v_term
    
    
    def ratio_p(self, param, SX_p):
        gamma = param[0]  # horizontal parameter
        beta_c = param[1:]  # vertical parameters
        
        # Apply parameters to respective statistics
        h_term = torch.exp(SX_p[:, 0] / gamma)  # horizontal term
        
        # For vertical term, we need to handle each column separately
        v_terms = []
        for c in range(self.C):
            v_terms.append(torch.exp(SX_p[:, 1] * beta_c[c]))
        v_term = torch.stack(v_terms, dim=1).prod(dim=1)  # multiply all vertical terms
        
        return h_term * v_term
    
    
    def uloglikelihood(self, param, X):
        interactions = self._compute_interactions(X, param)
        return torch.sum(X * interactions)
    
    
    def shift_p(self, X, state=torch.tensor([-1.0, 1.0]), diff=2.0):
        X_shift = (X.unsqueeze(-1) + torch.eye(X.shape[1]) * diff).transpose(1, 2)
        return torch.where((X_shift == state[-1] + diff), state[0], X_shift).transpose(0, 1)
    
    
    def pseudologlikelihood(self, param, X):
        interactions = self._compute_interactions(X, param)
        tmp_a = torch.exp(-interactions)
        tmp_b = torch.exp(interactions)
        tmp_Xa = tmp_a * ((X - 1).abs() / 2)
        tmp_Xb = tmp_b * ((X + 1).abs() / 2)
        tmp = (tmp_Xa + tmp_Xb) / (tmp_a + tmp_b)
        return torch.sum(tmp.log(), axis=1)
    
