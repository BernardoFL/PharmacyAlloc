import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from functools import partial  # Change this import
import tensorflow_probability.substrates.jax.distributions as tfd

#######################################
# Kernels and Core Functions
#######################################

@jit
def compute_rbf_kernel(coords1: jnp.ndarray, 
                      coords2: jnp.ndarray, 
                      lengthscale: float, 
                      variance: float) -> jnp.ndarray:
    """JIT-compiled RBF kernel computation"""
    diff = jnp.expand_dims(coords1, 1) - jnp.expand_dims(coords2, 0)
    dist_sq = jnp.sum(diff * diff, axis=-1)
    return variance * jnp.exp(-0.5 * dist_sq / (lengthscale ** 2))

@jit
def condition_similarity_vectorized(atc1: jnp.ndarray, 
                                  atc2: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled condition similarity computation"""
    n1, n2 = atc1.shape[0], atc2.shape[0]
    matches = jnp.zeros((n1, n2, 4))
    
    def compute_level_match(level):
        return jnp.all(
            atc1[:, None, :level+1] == atc2[None, :, :level+1], 
            axis=-1
        ) * (level + 1)
    
    matches = vmap(compute_level_match)(jnp.arange(4))
    return jnp.max(matches, axis=0)

class VariationalGP:
    def __init__(self, inducing_points: jnp.ndarray, 
                 condition_list: List, 
                 num_inducing: int):
        self.inducing_points = inducing_points
        self.condition_list = condition_list
        self.num_inducing = num_inducing
        
        # Initialize variational parameters
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        
        self.q_mu = random.normal(key1, (self.num_inducing, 1))
        self.q_sqrt = jnp.eye(self.num_inducing)
        
        # Initialize kernel parameters
        self.log_lengthscale = jnp.zeros(1)
        self.log_variance = jnp.zeros(1)
        self.log_scale = jnp.zeros(1)
        
        # Load precomputed kernel matrix
        kernel_data = np.load("Data/condition_kernel_matrix.npz")
        self.condition_kernel_matrix = jnp.array(kernel_data['kernel_matrix'])
    
    def get_params(self):
        """Get all trainable parameters as a dictionary"""
        return {
            'q_mu': self.q_mu,
            'q_sqrt': self.q_sqrt,
            'log_lengthscale': self.log_lengthscale,
            'log_variance': self.log_variance,
            'log_scale': self.log_scale
        }
    
    def set_params(self, params):
        """Set parameters from dictionary"""
        self.q_mu = params['q_mu']
        self.q_sqrt = params['q_sqrt']
        self.log_lengthscale = params['log_lengthscale']
        self.log_variance = params['log_variance']
        self.log_scale = params['log_scale']

    @partial(jit, static_argnums=(0,))  # Make self static
    def elbo(self, params, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute ELBO with given parameters"""
        # Extract parameters
        q_mu = params['q_mu']
        q_sqrt = params['q_sqrt']
        log_lengthscale = params['log_lengthscale']
        log_variance = params['log_variance']
        log_scale = params['log_scale']
        
        # Compute kernel matrices
        K_mm = self.compute_Kmm(log_lengthscale, log_variance, log_scale)
        K_nm = self.compute_Knm(X, log_lengthscale, log_variance, log_scale)
        
        # Compute variational distribution
        q_dist = tfd.MultivariateNormalTriL(
            loc=q_mu.flatten(),
            scale_tril=q_sqrt
        )
        
        # Compute KL divergence
        p_dist = tfd.MultivariateNormalTriL(
            loc=jnp.zeros(self.num_inducing),
            scale_tril=jnp.eye(self.num_inducing)
        )
        kl = tfd.kl_divergence(q_dist, p_dist)
        
        # Compute log likelihood
        f_mean = K_nm @ jnp.linalg.solve(K_mm, q_mu)
        log_lik = -0.5 * jnp.sum((y - f_mean.flatten()) ** 2)
        
        return log_lik - kl

    @partial(jit, static_argnums=(0,))
    def compute_Kmm(self, log_lengthscale, log_variance, log_scale):
        """Compute kernel matrix between inducing points"""
        lengthscale = jnp.exp(log_lengthscale)
        variance = jnp.exp(log_variance)
        scale = jnp.exp(log_scale)
        
        K = compute_rbf_kernel(
            self.inducing_points[:, 1:3],
            self.inducing_points[:, 1:3],
            lengthscale,
            variance
        )
        return scale * K + 1e-6 * jnp.eye(self.num_inducing)
    
    @partial(jit, static_argnums=(0,))
    def compute_Knm(self, X: jnp.ndarray, log_lengthscale, log_variance, log_scale):
        """Compute kernel matrix between inputs and inducing points"""
        lengthscale = jnp.exp(log_lengthscale)
        variance = jnp.exp(log_variance)
        scale = jnp.exp(log_scale)
        
        K = compute_rbf_kernel(
            X[:, 1:3],
            self.inducing_points[:, 1:3],
            lengthscale,
            variance
        )
        return scale * K
    
    @jit
    def compute_kernel_matrix(self, X1, X2):
        """Use precomputed condition kernel"""
        cond1 = X1[:, 0].astype(jnp.int32)
        cond2 = X2[:, 0].astype(jnp.int32)
        
        # Use precomputed condition kernel
        K_cond = self.condition_kernel_matrix[cond1][:, cond2]
        
        # Compute RBF kernel for spatial coordinates
        K_rbf = compute_rbf_kernel(
            X1[:, 1:3],
            X2[:, 1:3],
            jnp.exp(self.log_lengthscale),
            jnp.exp(self.log_variance)
        )
        
        return jnp.exp(self.log_scale) * K_cond * K_rbf