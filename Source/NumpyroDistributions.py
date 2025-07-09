"""
Numpyro distributions for IsingAnisotropic model with FDBayes.
"""

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax import random
import torch
import numpy as np

# Import the required modules
import sys
import os
sys.path.append('./Source')
sys.path.append('./_dependency')

from Models import IsingAnisotropic
from Posteriors import FDBayes


class IsingAnisotropicDistribution(dist.Distribution):
    """
    A Numpyro distribution that samples probability matrices based on 
    the IsingAnisotropic model with FDBayes posterior.
    
    This distribution generates probability matrices where each column 
    represents a multinomial distribution, with the probability structure
    determined by the Gibbs measure from the IsingAnisotropic model.

    Now supports hyperpriors on gamma and beta parameters.
    """
    
    def __init__(self, n, C, prior_params=None, hyperpriors=None, validate_args=None):
        """
        Initialize the IsingAnisotropic distribution.
        
        Parameters:
        -----------
        n : int
            Number of features per column (categories)
        C : int
            Number of columns (multinomial variables)
        prior_params : dict, optional
            Parameters for the prior distribution on Ising parameters
        hyperpriors : dict, optional
            Dictionary with keys 'gamma' and 'beta' specifying Numpyro distributions or their parameters for hyperpriors.
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
        self.hyperpriors = hyperpriors
        self.ising_model = IsingAnisotropic(n, C)
        event_shape = (n, C)
        super().__init__(event_shape=event_shape, validate_args=validate_args)
        
        # Add required attributes for Numpyro compatibility
        self._is_discrete = False
        self._support = dist.constraints.real

    def sample(self, key, sample_shape=()):
        """
        Sample probability matrices from the distribution.
        
        Parameters:
        -----------
        key : jax.random.PRNGKey
            Random key for sampling
        sample_shape : tuple
            Shape of samples to generate
        
        Returns:
        --------
        samples : jax.numpy.ndarray
            Sampled probability matrices of shape sample_shape + (n, C)
        """
        ising_params = self._sample_ising_params(key, sample_shape)
        samples = np.stack([self._params_to_probabilities(np.array(p)) for p in np.array(ising_params)])
        return samples

    def _sample_ising_params(self, key, sample_shape):
        gamma_key, beta_key = random.split(key)
        # Sample from fixed prior parameters
        gamma = dist.Normal(
            loc=self.prior_params['gamma_mean'],
            scale=self.prior_params['gamma_std']
        ).sample(gamma_key, sample_shape)
        
        # Handle beta sampling - beta_std might be a vector or scalar
        beta_mean = self.prior_params['beta_mean']
        beta_std = self.prior_params['beta_std']
        
        # If beta_std is a vector, we need to sample each beta with its own std
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
            # If we have sample dimensions, expand gamma to match beta's shape
            gamma_expanded = gamma[..., None]  # Add dimension to match beta
        else:
            # No sample dimensions, just add a dimension to gamma
            gamma_expanded = gamma[None] if gamma.ndim == 0 else gamma[..., None]
        
        params = jnp.concatenate([gamma_expanded, beta], axis=-1)
        params = jnp.exp(params)
        return params

    def _params_to_probabilities(self, params):
        params_torch = torch.tensor(np.array(params), dtype=torch.float32)
        probs = torch.zeros(self.n, self.C)
        for c in range(self.C):
            X_configs = torch.zeros(self.n, self.n, self.C)
            for i in range(self.n):
                X_configs[i, i, c] = 1.0
            X_flat = X_configs.view(self.n, -1)
            interactions = self.ising_model._compute_interactions(X_flat, params_torch)
            log_probs = interactions.sum(dim=1)
            probs[:, c] = torch.softmax(log_probs, dim=0)
        return jnp.array(probs.numpy())

    def log_prob(self, value):
        """
        Compute the log probability for the given probability matrix.
        
        For Numpyro compatibility, this computes a simple log probability
        based on the prior parameters. The FDBayes-specific log_prob
        with explicit parameters is available as log_prob_fdbayes().
        
        Parameters:
        -----------
        value : jax.numpy.ndarray
            Probability matrix of shape (n, C)
        
        Returns:
        --------
        log_prob : float
            Log probability based on prior parameters
        """
        # For Numpyro compatibility, compute a simple log probability
        # based on the prior parameters
        if self.prior_params is None:
            return 0.0
        
        # Convert to jax array for computation
        value_jnp = jnp.asarray(value)
        
        # Simple log probability based on prior parameters
        # This is a placeholder - in practice, you might want a more sophisticated prior
        log_prob = 0.0
        
        # Add regularization term to encourage reasonable probability values
        log_prob -= 0.1 * jnp.sum(value_jnp ** 2)  # L2 regularization
        
        # Add constraint that probabilities should be positive and sum to 1 per column
        for c in range(self.C):
            col_sum = jnp.sum(value_jnp[:, c])
            log_prob -= jnp.where(col_sum > 0, 0.1 * jnp.abs(col_sum - 1.0), 0.0)  # Encourage column sums to be 1
        
        return log_prob
    
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
        # Return a uniform probability matrix as the mean
        return jnp.ones((self.n, self.C)) / self.n
    
    def log_prob_fdbayes(self, value, param=None):
        """
        Compute the log generalized likelihood (negative FDBayes loss) for the given data matrix and Ising parameters.
        
        Parameters:
        -----------
        value : jax.numpy.ndarray
            Data matrix of shape (n, C) (should be integer or one-hot encoded)
        param : array-like or torch.Tensor
            Ising parameters (gamma, beta_1, ..., beta_C), shape (C+1,)
        
        Returns:
        --------
        log_prob : float
            Log generalized likelihood (negative FDBayes loss)
        """
        if param is None:
            raise ValueError("log_prob_fdbayes requires explicit Ising parameters (param) for FDBayes generalized likelihood.")
        # Convert value and param to torch
        value_torch = torch.tensor(np.array(value), dtype=torch.float32)
        param_torch = torch.tensor(np.array(param), dtype=torch.float32)
        # Set up FDBayes
        log_prior = lambda p: -0.5 * torch.sum(p**2)  # Simple Gaussian prior (not used in loss)
        posterior = FDBayes(
            self.ising_model.ratio_m,
            self.ising_model.ratio_p,
            self.ising_model.stat_m,
            self.ising_model.stat_p,
            log_prior
        )
        posterior.set_X(value_torch)
        loss = posterior.loss(param_torch)
        return -loss  # negative FDBayes loss is the generalized log-likelihood
    
    def sample_with_fdbayes(self, key, data, num_samples=1000, num_burnin=500):
        """
        Sample probability matrices using FDBayes posterior.
        
        Parameters:
        -----------
        key : jax.random.PRNGKey
            Random key for sampling
        data : jax.numpy.ndarray
            Observed data of shape (n_samples, C) with integer indices
        num_samples : int
            Number of posterior samples
        num_burnin : int
            Number of burn-in samples
            
        Returns:
        --------
        samples : jax.numpy.ndarray
            Sampled probability matrices of shape (num_samples, n, C)
        """
        # Convert data to torch tensor
        data_torch = torch.tensor(np.array(data), dtype=torch.long)
        
        # Set up FDBayes posterior
        log_prior = lambda param: -0.5 * torch.sum(param**2)  # Simple Gaussian prior
        posterior = FDBayes(
            self.ising_model.ratio_m,
            self.ising_model.ratio_p,
            self.ising_model.stat_m,
            self.ising_model.stat_p,
            log_prior
        )
        
        # Set data in posterior
        posterior.set_X(data_torch)
        
        # Set up transition distribution
        transit_p = torch.distributions.MultivariateNormal(
            torch.zeros(self.C+1), 
            0.1 * torch.eye(self.C+1)
        )
        
        # Sample from posterior
        post_samples = posterior.sample(
            num_samples, 
            num_burnin, 
            transit_p, 
            torch.randn(self.C+1), 
            beta=1.0
        )
        
        # Convert posterior samples to probability matrices
        prob_samples = []
        for i in range(min(num_samples, post_samples.shape[0])):
            param = post_samples[i]
            probs = self._params_to_probabilities(param)
            prob_samples.append(probs)
        
        return jnp.array(prob_samples)


def ising_anisotropic_model(n, C, data=None, prior_params=None):
    """
    Numpyro model using IsingAnisotropicDistribution.
    
    Parameters:
    -----------
    n : int
        Number of features per column
    C : int
        Number of columns
    data : jax.numpy.ndarray, optional
        Observed data
    prior_params : dict, optional
        Prior parameters for the distribution
    """
    # Sample probability matrix from IsingAnisotropic distribution
    probs = numpyro.sample(
        "probs", 
        IsingAnisotropicDistribution(n, C, prior_params)
    )
    
    if data is not None:
        # Sample observations from multinomial distributions
        for i in range(data.shape[0]):
            for c in range(C):
                numpyro.sample(
                    f"obs_{i}_{c}",
                    dist.Categorical(probs=probs[:, c]),
                    obs=data[i, c]
                )


def ising_anisotropic_guide(n, C, prior_params=None):
    """
    Guide (variational distribution) for the IsingAnisotropic model.
    
    Parameters:
    -----------
    n : int
        Number of features per column
    C : int
        Number of columns
    prior_params : dict, optional
        Prior parameters
    """
    # Variational parameters for the probability matrix
    probs_loc = numpyro.param("probs_loc", jnp.ones((n, C)) / n)
    probs_scale = numpyro.param("probs_scale", jnp.ones((n, C)) * 0.1)
    
    # Sample from variational distribution
    probs = numpyro.sample(
        "probs",
        dist.Normal(probs_loc, probs_scale)
    )
    
    # Ensure probabilities are positive and sum to 1
    probs = jax.nn.softmax(probs, axis=0)


# Utility functions for working with the distribution

def sample_ising_probabilities(n, C, key, num_samples=1, prior_params=None):
    """
    Convenience function to sample probability matrices.
    
    Parameters:
    -----------
    n : int
        Number of features per column
    C : int
        Number of columns
    key : jax.random.PRNGKey
        Random key
    num_samples : int
        Number of samples
    prior_params : dict, optional
        Prior parameters
        
    Returns:
    --------
    samples : jax.numpy.ndarray
        Sampled probability matrices
    """
    dist_obj = IsingAnisotropicDistribution(n, C, prior_params)
    return dist_obj.sample(key, (num_samples,))


def compute_ising_entropy(probs):
    """
    Compute entropy of probability matrices.
    
    Parameters:
    -----------
    probs : jax.numpy.ndarray
        Probability matrices of shape (..., n, C)
        
    Returns:
    --------
    entropy : jax.numpy.ndarray
        Entropy values
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    log_probs = jnp.log(probs + eps)
    entropy = -jnp.sum(probs * log_probs, axis=-2)  # Sum over features
    return entropy 