"""
Numpyro distributions for IsingAnisotropic model with FDBayes.
"""

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import torch

# Import the required modules
import sys
import os
sys.path.append('./Source')
sys.path.append('./_dependency')

from Models import IsingAnisotropic
from Posteriors import FDBayes


class IsingAnisotropicDistribution(dist.Distribution):
    """
    A Numpyro distribution that samples binary matrices from the IsingAnisotropic model.
    
    This distribution generates binary matrices X ∈ {-1, 1}^{n×C} where the joint
    probability is defined by the Ising model energy function and represented
    through the FDBayes generalized likelihood.
    
    The model uses:
    - γ (gamma) for horizontal interactions within rows
    - β_c (beta_c) for vertical interactions within column c
    """
    
    def __init__(self, n, C, prior_params=None, validate_args=None):
        """
        Initialize the IsingAnisotropic distribution.
        
        Parameters:
        -----------
        n : int
            Number of rows in the binary matrix
        C : int
            Number of columns in the binary matrix
        prior_params : dict, optional
            Parameters for the prior distribution on Ising parameters
        validate_args : bool, optional
            Whether to validate arguments
        """
        self.n = n
        self.C = C
        self.prior_params = prior_params or {
            'gamma_mean': 0.0,
            'gamma_std': 1.0,
            'beta_mean': 0.0,
            'beta_std': 1.0
        }
        
        # Initialize the Ising model
        self.ising_model = IsingAnisotropic(n, C)
        
        # Event shape is the binary matrix shape
        event_shape = (n, C)
        super().__init__(event_shape=event_shape, validate_args=validate_args)
        
        # Add required attributes for Numpyro compatibility
        self._is_discrete = True  # Binary matrices are discrete
        self._support = dist.constraints.integer_interval(-1, 1)

    def sample(self, key, sample_shape=()):
        """
        Sample binary matrices from the Ising model.
        
        Parameters:
        -----------
        key : jax.random.PRNGKey
            Random key for sampling
        sample_shape : tuple
            Shape of samples to generate
        
        Returns:
        --------
        samples : jax.numpy.ndarray
            Sampled binary matrices of shape sample_shape + (n, C) with values in {-1, 1}
        """
        # Sample Ising parameters first
        ising_params = self._sample_ising_params(key, sample_shape)
        
        # Convert to torch for the Ising model
        if len(sample_shape) > 0:
            # Handle multiple samples
            samples = []
            for i in range(sample_shape[0]):
                param_torch = torch.tensor(np.array(ising_params[i]), dtype=torch.float32)
                # Sample one binary matrix using the Ising model
                sample_torch = self.ising_model.sample(param_torch, num_sample=1)
                # Reshape to (n, C) and convert to JAX
                sample_reshaped = sample_torch.view(self.n, self.C)
                samples.append(jnp.array(sample_reshaped.numpy()))
            return jnp.stack(samples)
        else:
            # Single sample
            param_torch = torch.tensor(np.array(ising_params), dtype=torch.float32)
            sample_torch = self.ising_model.sample(param_torch, num_sample=1)
            sample_reshaped = sample_torch.view(self.n, self.C)
            return jnp.array(sample_reshaped.numpy())

    def _sample_ising_params(self, key, sample_shape):
        """
        Sample Ising parameters (gamma, beta_1, ..., beta_C) from priors.
        """
        gamma_key, beta_key = random.split(key)
        
        # Sample gamma (horizontal interaction parameter)
        gamma = dist.Normal(
            loc=self.prior_params['gamma_mean'],
            scale=self.prior_params['gamma_std']
        ).sample(gamma_key, sample_shape)
        
        # Sample beta parameters (vertical interaction parameters)
        beta_mean = self.prior_params['beta_mean']
        beta_std = self.prior_params['beta_std']
        
        # Handle beta sampling - beta_std might be a vector or scalar
        if hasattr(beta_std, 'shape') and len(beta_std.shape) > 0:
            # beta_std is a vector, sample each beta with its own std
            beta_samples = []
            for c in range(self.C):
                beta_c = dist.Normal(
                    loc=beta_mean,
                    scale=beta_std[c]
                ).sample(beta_key, sample_shape)
                beta_samples.append(beta_c)
            beta = jnp.stack(beta_samples, axis=-1)
        else:
            # beta_std is a scalar, sample all betas with the same std
            beta = dist.Normal(
                loc=beta_mean,
                scale=beta_std
            ).sample(beta_key, sample_shape + (self.C,))
        
        # Ensure gamma has the right shape for concatenation
        if len(sample_shape) > 0:
            gamma_expanded = gamma[..., None]
        else:
            gamma_expanded = gamma[None] if gamma.ndim == 0 else gamma[..., None]
        
        # Concatenate gamma and beta parameters
        params = jnp.concatenate([gamma_expanded, beta], axis=-1)
        return params

    def log_prob(self, value):
        """
        Compute the log probability for the given binary matrix using FDBayes.
        
        This computes the FDBayes generalized likelihood for the observed binary matrix.
        
        Parameters:
        -----------
        value : jax.numpy.ndarray
            Binary matrix of shape (n, C) with values in {-1, 1}
        
        Returns:
        --------
        log_prob : float
            Log probability based on FDBayes generalized likelihood
        """
        # Convert to torch for the Ising model
        value_torch = torch.tensor(np.array(value), dtype=torch.float32)
        
        # Reshape to (1, n*C) for the Ising model which expects batch dimension
        # value_torch shape: (n, C) -> reshape to (1, n*C)
        value_reshaped = value_torch.view(1, -1)  # Add batch dimension
        
        # Sample parameters from the prior for computing the likelihood
        # In practice, you might want to pass the parameters explicitly
        key = random.PRNGKey(0)  # Use fixed key for consistency
        params = self._sample_ising_params(key, ())
        param_torch = torch.tensor(np.array(params), dtype=torch.float32)
        
        # Set up FDBayes posterior
        log_prior = lambda p: -0.5 * torch.sum(p**2)  # Simple Gaussian prior
        posterior = FDBayes(
            self.ising_model.ratio_m,
            self.ising_model.ratio_p,
            self.ising_model.stat_m,
            self.ising_model.stat_p,
            log_prior
        )
        
        # Set the data in the posterior
        posterior.set_X(value_reshaped)
        
        # Compute the FDBayes loss (negative log likelihood)
        loss = posterior.loss(param_torch)
        
        # Return negative loss as log probability
        return -loss
    
    @property
    def is_discrete(self):
        """Return whether the distribution is discrete."""
        return self._is_discrete
    
    @property
    def support(self):
        """Return the support of the distribution."""
        return self._support
    
    @property
    def mean(self):
        """Return the mean of the distribution."""
        # For binary matrices, return zeros (neutral mean)
        return jnp.zeros((self.n, self.C))
    
    def enumerate_support(self, expand=True):
        """
        Enumerate support of the distribution.
        
        For binary matrices, this would be computationally expensive,
        so we return a simple approximation.
        """
        # Return a simple approximation - just zeros matrix
        # In practice, for large matrices this is not feasible to enumerate
        support = jnp.zeros((1, self.n, self.C))
        if expand:
            return support
        else:
            return support[0]
    
    def log_prob_with_params(self, value, params):
        """
        Compute log probability with explicit parameters.
        
        Parameters:
        -----------
        value : jax.numpy.ndarray
            Binary matrix of shape (n, C) with values in {-1, 1}
        params : jax.numpy.ndarray
            Ising parameters [gamma, beta_1, ..., beta_C]
        
        Returns:
        --------
        log_prob : float
            Log probability based on FDBayes generalized likelihood
        """
        # Convert to torch
        value_torch = torch.tensor(np.array(value), dtype=torch.float32)
        param_torch = torch.tensor(np.array(params), dtype=torch.float32)
        
        # Reshape to (1, n*C) for the Ising model which expects batch dimension
        # value_torch shape: (n, C) -> reshape to (1, n*C)
        value_reshaped = value_torch.view(1, -1)  # Add batch dimension
        
        # Set up FDBayes posterior
        log_prior = lambda p: -0.5 * torch.sum(p**2)
        posterior = FDBayes(
            self.ising_model.ratio_m,
            self.ising_model.ratio_p,
            self.ising_model.stat_m,
            self.ising_model.stat_p,
            log_prior
        )
        
        # Set the data and compute loss
        posterior.set_X(value_reshaped)
        loss = posterior.loss(param_torch)
        
        return -loss 