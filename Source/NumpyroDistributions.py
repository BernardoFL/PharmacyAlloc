"""
Numpyro distributions for IsingAnisotropic model with FDBayes.
"""

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax import random

# Import the required modules
import sys
import os
sys.path.append('./Source')
sys.path.append('./_dependency')

from Models import IsingAnisotropic
from JAXFDBayes import compute_ising_statistics_m, compute_ising_statistics_p, fd_bayes_log_likelihood


class IsingAnisotropicDistribution(dist.Distribution):
    """
    A Numpyro distribution that samples binary matrices from the IsingAnisotropic model (JAX-native).
    """
    def __init__(self, n, C, prior_params=None, validate_args=None):
        self.n = n
        self.C = C
        self.prior_params = prior_params or {
            'gamma_mean': 0.0,
            'gamma_std': 1.0,
            'beta_mean': 0.0,
            'beta_std': 1.0
        }
        self.ising_model = IsingAnisotropic(n, C)
        event_shape = (n, C)
        super().__init__(event_shape=event_shape, validate_args=validate_args)
        self._is_discrete = True
        self._support = dist.constraints.integer_interval(-1, 1)

    def sample(self, key, sample_shape=()):
        """
        Snowball sampling: start at a random point and propagate updates to neighbors.
        Each site's probability is updated using sigmoid of local field computed from current neighbor probabilities.
        """
        shape = sample_shape if sample_shape is not None else ()
        gamma_key, beta_key, init_key, start_key = random.split(key, 4)
        
        # Sample Ising parameters
        gamma = dist.Normal(
            loc=self.prior_params['gamma_mean'],
            scale=self.prior_params['gamma_std']
        ).sample(gamma_key, shape)
        beta_mean = self.prior_params['beta_mean']
        beta_std = self.prior_params['beta_std']
        if hasattr(beta_std, 'shape') and len(jnp.shape(beta_std)) > 0:
            beta = jnp.stack([
                dist.Normal(loc=beta_mean, scale=beta_std[c]).sample(beta_key, shape)
                for c in range(self.C)
            ], axis=-1)
        else:
            beta = dist.Normal(loc=beta_mean, scale=beta_std).sample(beta_key, shape + (self.C,))
        
        # Initialize random probabilities
        prob_surface = random.uniform(init_key, shape + (self.n, self.C))
        
        def update_single_sample(prob_grid, gamma_val, beta_val, start_key):
            n, C = self.n, self.C
            visited = jnp.zeros((n, C), dtype=bool)
            
            # Start at random point
            start_i = random.randint(start_key, (), 0, n)
            start_c = random.randint(random.split(start_key)[0], (), 0, C)
            
            # BFS queue simulation using scan
            def snowball_step(state, _):
                prob_grid, visited, queue_pos = state
                
                # Get current position (simulate queue with position counter)
                total_sites = n * C
                i, c = divmod(queue_pos % total_sites, C)
                
                # Skip if already visited
                already_visited = visited[i, c]
                
                def update_site():
                    # Compute local field from current neighbor probabilities
                    local_field = 0.0
                    # Horizontal neighbors (left)
                    local_field += jax.lax.cond(
                        c > 0,
                        lambda _: gamma_val * (2 * prob_grid[i, c-1] - 1),
                        lambda _: 0.0,
                        operand=None
                    )
                    # Horizontal neighbors (right)
                    local_field += jax.lax.cond(
                        c < C - 1,
                        lambda _: gamma_val * (2 * prob_grid[i, c+1] - 1),
                        lambda _: 0.0,
                        operand=None
                    )
                    # Vertical neighbors (up)
                    local_field += jax.lax.cond(
                        i > 0,
                        lambda _: beta_val[c] * (2 * prob_grid[i-1, c] - 1),
                        lambda _: 0.0,
                        operand=None
                    )
                    # Vertical neighbors (down)
                    local_field += jax.lax.cond(
                        i < n - 1,
                        lambda _: beta_val[c] * (2 * prob_grid[i+1, c] - 1),
                        lambda _: 0.0,
                        operand=None
                    )
                    # Update probability using sigmoid
                    new_prob = jax.nn.sigmoid(local_field)
                    new_prob_grid = prob_grid.at[i, c].set(new_prob)
                    new_visited = visited.at[i, c].set(True)
                    return new_prob_grid, new_visited
                
                def skip_site():
                    return prob_grid, visited
                
                new_prob_grid, new_visited = jax.lax.cond(
                    already_visited,
                    skip_site,
                    update_site
                )
                
                return (new_prob_grid, new_visited, queue_pos + 1), None
            
            # Run snowball for all sites
            initial_state = (prob_grid, visited, 0)
            (final_prob_grid, _, _), _ = jax.lax.scan(
                snowball_step, 
                initial_state, 
                jnp.arange(n * C * 2)  # Extra iterations to ensure coverage
            )
            
            return final_prob_grid
        
        if isinstance(shape, tuple) and len(shape) > 0:
            # Handle multiple samples
            start_keys = random.split(start_key, int(jnp.prod(jnp.array(shape))))
            gamma_flat = gamma.reshape(-1)
            beta_flat = beta.reshape(-1, self.C)
            prob_flat = prob_surface.reshape(-1, self.n, self.C)
            
            def process_sample(i):
                return update_single_sample(
                    prob_flat[i], 
                    gamma_flat[i], 
                    beta_flat[i], 
                    start_keys[i]
                )
            
            updated_probs = jax.vmap(process_sample)(jnp.arange(len(prob_flat)))
            return updated_probs.reshape(shape + (self.n, self.C))
        else:
            # Single sample
            return update_single_sample(prob_surface, gamma, beta, start_key)

    def log_prob(self, value):
        # value: (n, C) or (batch_size, n, C)
        # For now, assume single matrix
        # Use FDBayes JAX log likelihood
        # Sample parameters from the prior for log_prob (or pass as argument in a more advanced version)
        # Here, just use prior mean for demonstration
        gamma = self.prior_params['gamma_mean']
        beta = jnp.ones(self.C) * self.prior_params['beta_mean']
        param = jnp.concatenate([jnp.array([gamma]), beta], axis=-1)
        # Compute statistics
        X = value.reshape(1, self.n, self.C) if value.ndim == 2 else value
        SX_m = compute_ising_statistics_m(X)
        SX_p = compute_ising_statistics_p(X)
        logp = fd_bayes_log_likelihood(param, SX_m, SX_p)
        return logp

    @property
    def is_discrete(self):
        return self._is_discrete

    @property
    def support(self):
        return self._support

    @property
    def mean(self):
        return jnp.zeros((self.n, self.C))

    def enumerate_support(self, expand=True):
        support = jnp.zeros((1, self.n, self.C))
        if expand:
            return support
        else:
            return support[0] 


class GMRFDistribution(dist.Distribution):
    """
    A Numpyro distribution that implements a Gaussian Markov Random Field (GMRF) 
    for a 2D grid of probabilities with anisotropic interactions.
    
    The model consists of:
    1. A latent grid x of shape (I, C) with GMRF prior
    2. Probabilities p = sigmoid(x) 
    3. Anisotropic interactions: vertical (gamma) and horizontal (beta)
    """
    
    def __init__(self, I, C, sigma=1.0, gamma=0.0, beta=None, validate_args=None):
        """
        Initialize GMRF distribution.
        
        Parameters
        ----------
        I : int
            Number of rows (patients)
        C : int
            Number of columns (conditions)
        sigma : float, default=1.0
            Scale parameter for the potential term
        gamma : float, default=0.0
            Vertical coupling strength
        beta : array-like, optional
            Horizontal coupling strengths of shape (C-1,). If None, uses zeros.
        validate_args : bool, optional
            Whether to validate arguments
        """
        self.I = I
        self.C = C
        self.sigma = sigma
        self.gamma = gamma
        self.beta = jnp.zeros(C - 1) if beta is None else jnp.array(beta)
        
        # Event shape is the probability grid shape
        event_shape = (I, C)
        super().__init__(event_shape=event_shape, validate_args=validate_args)
        self._is_discrete = False
        self._support = dist.constraints.interval(0, 1)

    def sample(self, key, sample_shape=()):
        """
        Sample from the GMRF distribution.
        
        This samples the latent grid x from the GMRF model with fixed parameters
        and then transforms it to probabilities using sigmoid.
        """
        shape = sample_shape if sample_shape is not None else ()
        
        # Sample latent grid x from independent Normal (potential term)
        # This efficiently implements the potential term -x^2/(2*sigma^2)
        x = dist.Normal(0, self.sigma).sample(key, shape + (self.I, self.C))
        
        # Transform to probabilities using sigmoid with clipping
        p_raw = jax.nn.sigmoid(x)
        p = jnp.clip(p_raw, 1e-7, 1.0 - 1e-7)
        
        return p

    def log_prob(self, value):
        """
        Compute log probability of the GMRF distribution.
        
        Parameters
        ----------
        value : jax.numpy.ndarray
            Probability grid of shape (I, C) or (batch_size, I, C)
            
        Returns
        -------
        jax.numpy.ndarray
            Log probability values
        """
        # Clip probabilities to prevent under/overflow in logit transformation
        # Use small epsilon to avoid exact 0 or 1
        epsilon = 1e-7
        value_clipped = jnp.clip(value, epsilon, 1.0 - epsilon)
        
        # Transform probabilities back to latent space
        x = jax.scipy.special.logit(value_clipped)
        
        # Clip x values to prevent extreme values
        x_clipped = jnp.clip(x, -10.0, 10.0)
        
        # Use the fixed parameters of the distribution
        sigma = self.sigma
        gamma = self.gamma
        beta = self.beta
        
        # Compute log probability components
        # 1. Potential term: -x^2/(2*sigma^2)
        potential_term = -0.5 * jnp.sum(x_clipped**2) / (sigma**2)
        
        # 2. Vertical interactions: gamma * sum(x[i,c] * x[i+1,c])
        v_interactions = gamma * jnp.sum(x_clipped[:-1, :] * x_clipped[1:, :])
        
        # 3. Horizontal interactions: sum(beta[c] * sum(x[i,c] * x[i,c+1]))
        # Fix the broadcasting issue by computing interactions properly
        h_interactions = 0.0
        for c in range(self.C - 1):
            h_interactions += beta[c] * jnp.sum(x_clipped[:, c] * x_clipped[:, c+1])
        
        # Total log probability
        log_prob = potential_term + v_interactions + h_interactions
        
        # Clip final log probability to prevent extreme values
        log_prob_clipped = jnp.clip(log_prob, -1e6, 1e6)
        
        return log_prob_clipped

    @property
    def is_discrete(self):
        return self._is_discrete

    @property
    def support(self):
        return self._support

    @property
    def mean(self):
        """Mean of the probability grid (0.5 for sigmoid of zero-mean latent)"""
        return 0.5 * jnp.ones((self.I, self.C))

    def enumerate_support(self, expand=True):
        """Enumerate support (not implemented for continuous distribution)"""
        raise NotImplementedError("enumerate_support not implemented for GMRF distribution") 