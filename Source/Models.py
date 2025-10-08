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

import jax
import jax.numpy as jnp
from jax import random
from _dependency.ising_jax import IsingJAX
from _dependency.ising import Ising

#==========================================================================
# Define: Anisotropic 2D Ising Model
#==========================================================================

class IsingAnisotropic:
    def __init__(self, n, C):
        self.n = n
        self.C = C
        self.size = n * C
        self.ising = IsingJAX(self.size)

    def set_ferromagnet(self, l, temp, periodic=True, anti=False, isotropic=True):
        self.ising.set_ferromagnet(l, temp, periodic, anti, isotropic)

    def sample(self, key, param, num_sample=1, num_iters=10000):
        self.set_ferromagnet([self.n, self.C], param, periodic=True, anti=False, isotropic=False)
        return self.ising.sample(key, num_iters, num_samples=num_sample)

    def stat_m(self, X):
        batch_size = X.shape[0]
        n, C = self.n, self.C
        X_reshaped = X.reshape((batch_size, n, C))
        h_stat = jnp.zeros_like(X_reshaped)
        h_stat = h_stat.at[:, :, 1:].add(X_reshaped[:, :, :-1])
        h_stat = h_stat.at[:, :, :-1].add(X_reshaped[:, :, 1:])
        h_stat = -2 * X_reshaped * h_stat
        v_stat = jnp.zeros_like(X_reshaped)
        v_stat = v_stat.at[:, 1:, :].add(X_reshaped[:, :-1, :])
        v_stat = v_stat.at[:, :-1, :].add(X_reshaped[:, 1:, :])
        v_stat = -2 * X_reshaped * v_stat
        h_sum = jnp.sum(h_stat.reshape((batch_size, -1)), axis=1)
        v_sum = jnp.sum(v_stat.reshape((batch_size, -1)), axis=1)
        return jnp.stack([h_sum, v_sum], axis=1)

    def stat_p(self, X):
        batch_size = X.shape[0]
        n, C = self.n, self.C
        X_reshaped = X.reshape((batch_size, n, C))
        h_stat = jnp.zeros_like(X_reshaped)
        h_stat = h_stat.at[:, :, 1:].add(X_reshaped[:, :, :-1])
        h_stat = h_stat.at[:, :, :-1].add(X_reshaped[:, :, 1:])
        h_stat = 2 * X_reshaped * h_stat
        v_stat = jnp.zeros_like(X_reshaped)
        v_stat = v_stat.at[:, 1:, :].add(X_reshaped[:, :-1, :])
        v_stat = v_stat.at[:, :-1, :].add(X_reshaped[:, 1:, :])
        v_stat = 2 * X_reshaped * v_stat
        h_sum = jnp.sum(h_stat.reshape((batch_size, -1)), axis=1)
        v_sum = jnp.sum(v_stat.reshape((batch_size, -1)), axis=1)
        return jnp.stack([h_sum, v_sum], axis=1)

    def _compute_interactions(self, X, param):
        batch_size = X.shape[0]
        n, C = self.n, self.C
        X_reshaped = X.reshape((batch_size, n, C))
        gamma = param[0]
        beta_c = param[1:]
        # Horizontal interactions
        h_interactions = jnp.zeros_like(X_reshaped)
        for i in range(n):
            for c in range(C):
                if c > 0:
                    h_interactions = h_interactions.at[:, i, c].add(X_reshaped[:, i, c-1] * gamma)
                if c < C - 1:
                    h_interactions = h_interactions.at[:, i, c].add(X_reshaped[:, i, c+1] * gamma)
        # Vertical interactions
        v_interactions = jnp.zeros_like(X_reshaped)
        for c in range(C):
            for i in range(n):
                if i > 0:
                    v_interactions = v_interactions.at[:, i, c].add(X_reshaped[:, i-1, c] * beta_c[c])
                if i < n - 1:
                    v_interactions = v_interactions.at[:, i, c].add(X_reshaped[:, i+1, c] * beta_c[c])
        total_interactions = h_interactions + v_interactions
        return total_interactions.reshape(batch_size, -1)

    def ratio_m(self, param, SX_m):
        gamma = param[0]
        beta_c = param[1:]
        h_term = jnp.exp(SX_m[:, 0] * gamma)
        v_terms = jnp.stack([jnp.exp(SX_m[:, 1] * beta_c[c]) for c in range(self.C)], axis=1)
        v_term = jnp.prod(v_terms, axis=1)
        return h_term * v_term

    def ratio_p(self, param, SX_p):
        gamma = param[0]
        beta_c = param[1:]
        h_term = jnp.exp(SX_p[:, 0] / gamma)
        v_terms = jnp.stack([jnp.exp(SX_p[:, 1] * beta_c[c]) for c in range(self.C)], axis=1)
        v_term = jnp.prod(v_terms, axis=1)
        return h_term * v_term

    def uloglikelihood(self, param, X):
        interactions = self._compute_interactions(X, param)
        return jnp.sum(X * interactions)

    def shift_p(self, X, state=None, diff=2.0):
        # JAX version of shift_p
        if state is None:
            state = jnp.array([-1.0, 1.0])
        X_shift = (X[..., None] + jnp.eye(X.shape[1]) * diff).transpose(1, 2, 0)
        X_shift = X_shift.transpose(2, 0, 1)  # match PyTorch's transpose(1,2).transpose(0,1)
        return jnp.where((X_shift == state[-1] + diff), state[0], X_shift)

    def pseudologlikelihood(self, param, X):
        interactions = self._compute_interactions(X, param)
        tmp_a = jnp.exp(-interactions)
        tmp_b = jnp.exp(interactions)
        tmp_Xa = tmp_a * (jnp.abs(X - 1) / 2)
        tmp_Xb = tmp_b * (jnp.abs(X + 1) / 2)
        tmp = (tmp_Xa + tmp_Xb) / (tmp_a + tmp_b)
        return jnp.sum(jnp.log(tmp), axis=1)
    
    