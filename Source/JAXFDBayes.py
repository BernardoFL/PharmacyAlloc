"""
JAX-compiled FDBayes likelihood computation for IsingAnisotropic model.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random


@jit
def compute_ising_statistics_m(X):
    """
    JIT-compiled computation of stat_m for IsingAnisotropic model.
    
    Computes horizontal and vertical statistics for ratio_m computation.
    Based on the IsingAnisotropic.stat_m method from Models.py.
    
    Parameters:
    -----------
    X : jax.numpy.ndarray
        Binary matrix of shape (batch_size, n, C) with values in {-1, 1}
        
    Returns:
    --------
    SX_m : jax.numpy.ndarray
        Statistics for ratio_m computation, shape (batch_size, 2)
    """
    batch_size, n, C = X.shape
    
    # Horizontal statistics (within rows)
    h_stat = jnp.zeros_like(X)
    # Left neighbors
    h_stat = h_stat.at[:, :, 1:].add(X[:, :, :-1])
    # Right neighbors  
    h_stat = h_stat.at[:, :, :-1].add(X[:, :, 1:])
    h_stat = -2 * X * h_stat
    h_sum = jnp.sum(h_stat, axis=(1, 2))
    
    # Vertical statistics (within columns)
    v_stat = jnp.zeros_like(X)
    # Up neighbors
    v_stat = v_stat.at[:, 1:, :].add(X[:, :-1, :])
    # Down neighbors
    v_stat = v_stat.at[:, :-1, :].add(X[:, 1:, :])
    v_stat = -2 * X * v_stat
    v_sum = jnp.sum(v_stat, axis=(1, 2))
    
    return jnp.stack([h_sum, v_sum], axis=1)


@jit
def compute_ising_statistics_p(X):
    """
    JIT-compiled computation of stat_p for IsingAnisotropic model.
    
    Computes horizontal and vertical statistics for ratio_p computation.
    Based on the IsingAnisotropic.stat_p method from Models.py.
    
    Parameters:
    -----------
    X : jax.numpy.ndarray
        Binary matrix of shape (batch_size, n, C) with values in {-1, 1}
        
    Returns:
    --------
    SX_p : jax.numpy.ndarray
        Statistics for ratio_p computation, shape (batch_size, 2)
    """
    batch_size, n, C = X.shape
    
    # Horizontal statistics (within rows)
    h_stat = jnp.zeros_like(X)
    # Left neighbors
    h_stat = h_stat.at[:, :, 1:].add(X[:, :, :-1])
    # Right neighbors
    h_stat = h_stat.at[:, :, :-1].add(X[:, :, 1:])
    h_stat = 2 * X * h_stat
    h_sum = jnp.sum(h_stat, axis=(1, 2))
    
    # Vertical statistics (within columns)
    v_stat = jnp.zeros_like(X)
    # Up neighbors
    v_stat = v_stat.at[:, 1:, :].add(X[:, :-1, :])
    # Down neighbors
    v_stat = v_stat.at[:, :-1, :].add(X[:, 1:, :])
    v_stat = 2 * X * v_stat
    v_sum = jnp.sum(v_stat, axis=(1, 2))
    
    return jnp.stack([h_sum, v_sum], axis=1)


@jit
def compute_ratio_m(param, SX_m):
    """
    JIT-compiled computation of ratio_m for IsingAnisotropic model.
    
    Based on the IsingAnisotropic.ratio_m method from Models.py.
    
    Parameters:
    -----------
    param : jax.numpy.ndarray
        Model parameters [gamma, beta_1, ..., beta_C]
    SX_m : jax.numpy.ndarray
        Statistics from stat_m, shape (batch_size, 2)
        
    Returns:
    --------
    ratio_m : jax.numpy.ndarray
        Ratio for FDBayes loss computation, shape (batch_size,)
    """
    gamma = param[0]
    beta = param[1:]
    C = beta.shape[0]
    
    # Add numerical stability
    gamma = jnp.clip(gamma, -10, 10)
    beta = jnp.clip(beta, -10, 10)
    
    # Horizontal term
    h_term = jnp.exp(jnp.clip(SX_m[:, 0] * gamma, -10, 10))
    
    # Vertical terms (one per column)
    v_terms = jnp.exp(jnp.clip(SX_m[:, 1:2] * beta, -10, 10))  # Shape: (batch_size, C)
    v_term = jnp.prod(v_terms, axis=1)  # Multiply across columns
    
    return h_term * v_term


@jit
def compute_ratio_p(param, SX_p):
    """
    JIT-compiled computation of ratio_p for IsingAnisotropic model.
    
    Based on the IsingAnisotropic.ratio_p method from Models.py.
    
    Parameters:
    -----------
    param : jax.numpy.ndarray
        Model parameters [gamma, beta_1, ..., beta_C]
    SX_p : jax.numpy.ndarray
        Statistics from stat_p, shape (batch_size, 2)
        
    Returns:
    --------
    ratio_p : jax.numpy.ndarray
        Ratio for FDBayes loss computation, shape (batch_size,)
    """
    gamma = param[0]
    beta = param[1:]
    C = beta.shape[0]
    
    # Add numerical stability
    gamma = jnp.clip(gamma, -10, 10)
    beta = jnp.clip(beta, -10, 10)
    
    # Horizontal term - both gamma and beta should multiply
    h_term = jnp.exp(jnp.clip(SX_p[:, 0] * gamma, -10, 10))
    
    # Vertical terms (one per column)
    v_terms = jnp.exp(jnp.clip(SX_p[:, 1:2] * beta, -10, 10))  # Shape: (batch_size, C)
    v_term = jnp.prod(v_terms, axis=1)  # Multiply across columns
    
    return h_term * v_term


@jit
def fd_bayes_loss(param, SX_m, SX_p):
    """
    JIT-compiled FDBayes loss computation for IsingAnisotropic model.
    
    Parameters:
    -----------
    param : jax.numpy.ndarray
        Model parameters [gamma, beta_1, ..., beta_C]
    SX_m : jax.numpy.ndarray
        Precomputed statistics for ratio_m
    SX_p : jax.numpy.ndarray
        Precomputed statistics for ratio_p
        
    Returns:
    --------
    loss : float
        FDBayes loss value
    """
    ratio_m = compute_ratio_m(param, SX_m)
    ratio_p = compute_ratio_p(param, SX_p)
    
    # FDBayes loss: (ratio_m^2 - 2*ratio_p).sum()
    loss = jnp.sum(ratio_m**2 - 2 * ratio_p)
    
    return loss


@jit
def fd_bayes_log_likelihood(param, SX_m, SX_p, log_prior_func=None):
    """
    JIT-compiled FDBayes log likelihood (negative loss + prior).
    
    Parameters:
    -----------
    param : jax.numpy.ndarray
        Model parameters [gamma, beta_1, ..., beta_C]
    SX_m : jax.numpy.ndarray
        Precomputed statistics for ratio_m
    SX_p : jax.numpy.ndarray
        Precomputed statistics for ratio_p
    log_prior_func : callable, optional
        Log prior function. If None, uses simple Gaussian prior.
        
    Returns:
    --------
    log_likelihood : float
        FDBayes log likelihood value
    """
    # Compute FDBayes loss
    loss = fd_bayes_loss(param, SX_m, SX_p)
    
    # Add prior contribution
    if log_prior_func is None:
        # Default Gaussian prior
        log_prior = -0.5 * jnp.sum(param**2)
    else:
        log_prior = log_prior_func(param)
    
    # Return negative loss + prior (this is the log likelihood)
    return -loss + log_prior


# Vectorized versions for batch processing
@jit
def fd_bayes_loss_batch(params, SX_m, SX_p):
    """
    JIT-compiled batch FDBayes loss computation.
    
    Parameters:
    -----------
    params : jax.numpy.ndarray
        Batch of model parameters, shape (batch_size, n_params)
    SX_m : jax.numpy.ndarray
        Precomputed statistics for ratio_m
    SX_p : jax.numpy.ndarray
        Precomputed statistics for ratio_p
        
    Returns:
    --------
    losses : jax.numpy.ndarray
        FDBayes loss values for each parameter set
    """
    return vmap(fd_bayes_loss, in_axes=(0, None, None))(params, SX_m, SX_p)


@jit
def fd_bayes_log_likelihood_batch(params, SX_m, SX_p, log_prior_func=None):
    """
    JIT-compiled batch FDBayes log likelihood computation.
    
    Parameters:
    -----------
    params : jax.numpy.ndarray
        Batch of model parameters, shape (batch_size, n_params)
    SX_m : jax.numpy.ndarray
        Precomputed statistics for ratio_m
    SX_p : jax.numpy.ndarray
        Precomputed statistics for ratio_p
    log_prior_func : callable, optional
        Log prior function
        
    Returns:
    --------
    log_likelihoods : jax.numpy.ndarray
        FDBayes log likelihood values for each parameter set
    """
    return vmap(fd_bayes_log_likelihood, in_axes=(0, None, None, None))(params, SX_m, SX_p, log_prior_func)


class JAXFDBayes:
    """
    JAX-compiled FDBayes implementation for IsingAnisotropic model.
    """
    
    def __init__(self, log_prior_func=None):
        """
        Initialize JAX FDBayes.
        
        Parameters:
        -----------
        log_prior_func : callable, optional
            Log prior function. If None, uses simple Gaussian prior.
        """
        self.log_prior_func = log_prior_func
        self.SX_m = None
        self.SX_p = None
    
    def set_data(self, X):
        """
        Precompute statistics for the data.
        
        Parameters:
        -----------
        X : jax.numpy.ndarray
            Binary data matrix of shape (batch_size, n, C) with values in {-1, 1}
        """
        self.SX_m = compute_ising_statistics_m(X)
        self.SX_p = compute_ising_statistics_p(X)
    
    def loss(self, param):
        """
        Compute FDBayes loss.
        
        Parameters:
        -----------
        param : jax.numpy.ndarray
            Model parameters [gamma, beta_1, ..., beta_C]
            
        Returns:
        --------
        loss : float
            FDBayes loss value
        """
        if self.SX_m is None or self.SX_p is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        return fd_bayes_loss(param, self.SX_m, self.SX_p)
    
    def log_likelihood(self, param):
        """
        Compute FDBayes log likelihood.
        
        Parameters:
        -----------
        param : jax.numpy.ndarray
            Model parameters [gamma, beta_1, ..., beta_C]
            
        Returns:
        --------
        log_likelihood : float
            FDBayes log likelihood value
        """
        if self.SX_m is None or self.SX_p is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        return fd_bayes_log_likelihood(param, self.SX_m, self.SX_p, self.log_prior_func)
    
    def gradient(self, param):
        """
        Compute gradient of FDBayes loss.
        
        Parameters:
        -----------
        param : jax.numpy.ndarray
            Model parameters [gamma, beta_1, ..., beta_C]
            
        Returns:
        --------
        gradient : jax.numpy.ndarray
            Gradient of FDBayes loss
        """
        if self.SX_m is None or self.SX_p is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        return grad(fd_bayes_loss)(param, self.SX_m, self.SX_p)
    
    def batch_loss(self, params):
        """
        Compute FDBayes loss for a batch of parameters.
        
        Parameters:
        -----------
        params : jax.numpy.ndarray
            Batch of model parameters, shape (batch_size, n_params)
            
        Returns:
        --------
        losses : jax.numpy.ndarray
            FDBayes loss values for each parameter set
        """
        if self.SX_m is None or self.SX_p is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        return fd_bayes_loss_batch(params, self.SX_m, self.SX_p)
    
    def batch_log_likelihood(self, params):
        """
        Compute FDBayes log likelihood for a batch of parameters.
        
        Parameters:
        -----------
        params : jax.numpy.ndarray
            Batch of model parameters, shape (batch_size, n_params)
            
        Returns:
        --------
        log_likelihoods : jax.numpy.ndarray
            FDBayes log likelihood values for each parameter set
        """
        if self.SX_m is None or self.SX_p is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        return fd_bayes_log_likelihood_batch(params, self.SX_m, self.SX_p, self.log_prior_func)


# Utility functions for integration with Numpyro
def create_fdbayes_numpyro_model(data, n_params):
    """
    Create a Numpyro model that uses JAX-compiled FDBayes likelihood.
    
    Parameters:
    -----------
    data : jax.numpy.ndarray
        Observed binary data of shape (batch_size, n, C) with values in {-1, 1}
    n_params : int
        Number of model parameters (gamma + C betas)
        
    Returns:
    --------
    model_func : callable
        Numpyro model function
    """
    import numpyro
    import numpyro.distributions as dist
    
    # Initialize JAX FDBayes
    fd_bayes = JAXFDBayes()
    fd_bayes.set_data(data)
    
    def model():
        # Sample parameters from prior
        param = numpyro.sample("param", dist.Normal(0, 1).expand([n_params]))
        
        # Use JAX-compiled FDBayes likelihood
        log_likelihood = fd_bayes.log_likelihood(param)
        
        # Add likelihood to the model
        numpyro.factor("fd_bayes_likelihood", log_likelihood)
    
    return model


# Example usage and testing
def test_jax_fdbayes():
    """
    Test the JAX-compiled FDBayes implementation.
    """
    print("Testing JAX-compiled FDBayes for IsingAnisotropic...")
    
    # Create test binary data
    batch_size, n, C = 10, 5, 4
    key = random.PRNGKey(42)
    X = random.choice(key, jnp.array([-1, 1]), shape=(batch_size, n, C))
    
    # Create test parameters
    n_params = C + 1  # gamma + C betas
    key, subkey = random.split(key)
    param = random.normal(subkey, (n_params,))
    
    # Initialize JAX FDBayes
    fd_bayes = JAXFDBayes()
    fd_bayes.set_data(X)
    
    # Test loss computation
    loss = fd_bayes.loss(param)
    print(f"FDBayes loss: {loss:.4f}")
    
    # Test log likelihood computation
    log_likelihood = fd_bayes.log_likelihood(param)
    print(f"FDBayes log likelihood: {log_likelihood:.4f}")
    
    # Test gradient computation
    gradient = fd_bayes.gradient(param)
    print(f"FDBayes gradient shape: {gradient.shape}")
    print(f"Gradient norm: {jnp.linalg.norm(gradient):.4f}")
    
    # Test batch computation
    batch_size_params = 5
    key, subkey = random.split(key)
    params_batch = random.normal(subkey, (batch_size_params, n_params))
    losses_batch = fd_bayes.batch_loss(params_batch)
    print(f"Batch losses shape: {losses_batch.shape}")
    print(f"Batch losses mean: {jnp.mean(losses_batch):.4f}")
    
    print("JAX FDBayes tests completed successfully!")
    
    return fd_bayes


if __name__ == "__main__":
    test_jax_fdbayes() 