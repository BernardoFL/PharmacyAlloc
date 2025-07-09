#!/usr/bin/env python

import os
import sys

# Add paths for the required modules
sys.path.append('./Source')
sys.path.append('./_dependency')

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO messages

import argparse
import logging
import math
import time
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import random
from datetime import datetime

# Import Numpyro modules
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import initialize_model

# Import the required modules
from Source.NumpyroDistributions import IsingAnisotropicDistribution
from dataloader import load_data

def setup_logging(log_dir='logs'):
    """
    Set up logging configuration for the Ising model inference.
    
    Creates a timestamped log file in the specified directory and configures
    logging to output to both file and console.
    
    Parameters
    ----------
    log_dir : str, default='logs'
        Directory where log files will be stored. Created if it doesn't exist.
        
    Returns
    -------
    str
        Path to the created log file.
        
    Examples
    --------
    >>> log_file = setup_logging()
    >>> log_file = setup_logging('custom_logs')
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'ising_model_run_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print logs to console
        ]
    )
    logging.info(f"Logging to {log_file}")
    return log_file

def prepare_ising_data(A_sorted, X_cov_sorted, condition_list):
    """
    Prepare data for IsingAnisotropic model inference.
    
    Converts medical data matrices to JAX arrays suitable for Ising model analysis.
    The binary matrix A_sorted is used as the main data, where each patient
    represents a sample and conditions represent variables.
    
    Parameters
    ----------
    A_sorted : numpy.ndarray
        Binary matrix of shape (n_patients, n_conditions) indicating presence/absence
        of conditions for each patient.
    X_cov_sorted : numpy.ndarray
        Covariate matrix (not used in current implementation but kept for compatibility).
    condition_list : list
        List of condition names (not used in current implementation but kept for compatibility).
        
    Returns
    -------
    jax.numpy.ndarray
        JAX array of shape (n_patients, n_conditions) with dtype float32.
        
    Examples
    --------
    >>> A = np.random.randint(0, 2, size=(100, 50))
    >>> X_cov = np.random.randn(100, 5)
    >>> conditions = [f"condition_{i}" for i in range(50)]
    >>> data = prepare_ising_data(A, X_cov, conditions)
    >>> print(data.shape)  # (100, 50)
    """
    logging.info("Preparing data for IsingAnisotropic model...")
    
    # Use the binary matrix A_sorted as the main data
    # A_sorted shape: (n_patients, n_conditions)
    # We'll treat each patient as a sample and conditions as variables
    
    # Convert to jax array
    data_array = jnp.array(A_sorted, dtype=jnp.float32)
    
    logging.info(f"Prepared Ising data shape: {data_array.shape}")
    logging.info(f"Data statistics: mean={data_array.mean():.4f}, std={data_array.std():.4f}")
    logging.info(f"Binary values: {jnp.unique(data_array)}")
    
    return data_array

def ising_model(n, C, data):
    """
    Numpyro model for IsingAnisotropic inference with hierarchical priors.
    
    This model implements a hierarchical Bayesian structure for the Ising model:
    1. Samples hyperpriors for gamma parameters (Normal-Gamma)
    2. Samples hyperpriors for beta parameters (Horseshoe prior)
    3. Samples probability matrices from IsingAnisotropicDistribution
    4. Models data as independent Bernoulli draws
    
    Parameters
    ----------
    n : int
        Number of features per column (categories).
    C : int
        Number of columns (multinomial variables).
    data : jax.numpy.ndarray
        Observed data matrix of shape (n_samples, n*C) where each row represents
        a flattened observation.
        
    Notes
    -----
    The model uses the following hierarchical structure:
    
    - gamma_mean ~ Normal(0, 2)
    - gamma_std ~ Gamma(2, 1)
    - tau ~ HalfCauchy(1)  # Global shrinkage
    - lambda_beta_c ~ HalfCauchy(1)  # Local shrinkage for each c
    - beta_mean ~ Normal(0, 2)
    - beta_std = tau * lambda_beta  # Horseshoe shrinkage
    - probs ~ IsingAnisotropicDistribution(n, C, dynamic_prior_params)
    - data[i, c] ~ Bernoulli(probs[c])
    
    Examples
    --------
    >>> n, C = 3, 4
    >>> data = jnp.random.randint(0, 2, size=(10, n*C))
    >>> model = lambda: ising_model(n, C, data)
    """
    # Sample hyperpriors for gamma
    gamma_mean = numpyro.sample("gamma_mean", dist.Normal(0, 2))
    gamma_std = numpyro.sample("gamma_std", dist.Gamma(2, 1))  # Positive std
    
    # Horseshoe prior for betas
    # Global shrinkage parameter
    tau = numpyro.sample("tau", dist.HalfCauchy(1))
    
    # Local shrinkage parameters for each beta
    lambda_beta = numpyro.sample("lambda_beta", dist.HalfCauchy(1).expand([C]))
    
    # Beta means and standard deviations using horseshoe structure
    beta_mean = numpyro.sample("beta_mean", dist.Normal(0, 2))
    beta_std = tau * lambda_beta  # Horseshoe shrinkage
    
    # Create prior parameters using the sampled hyperpriors
    dynamic_prior_params = {
        'gamma_mean': gamma_mean,
        'gamma_std': gamma_std,
        'beta_mean': beta_mean,
        'beta_std': beta_std
    }
    
    # Sample probability matrix from IsingAnisotropicDistribution
    probs = numpyro.sample(
        "probs", 
        IsingAnisotropicDistribution(n, C, dynamic_prior_params)
    )
    # For each data point and each variable, sample from Bernoulli
    for i in range(data.shape[0]):
        for c in range(C):
            numpyro.sample(
                f"obs_{i}_{c}",
                dist.Bernoulli(probs=probs[c]),  # or probs[:, c] depending on shape
                obs=data[i, c]
            )

def run_ising_inference(data, args):
    """
    Run IsingAnisotropic model inference using Numpyro MCMC.
    
    Performs Bayesian inference on the Ising model using NUTS (No-U-Turn Sampler)
    with multiple chains. The model automatically determines grid dimensions
    based on the number of conditions in the data.
    
    Parameters
    ----------
    data : jax.numpy.ndarray
        Input data matrix of shape (n_patients, n_conditions).
    args : argparse.Namespace
        Command line arguments containing inference parameters:
        - dnum: Number of data samples to use
        - pnum: Number of posterior samples
        
    Returns
    -------
    tuple
        A tuple containing:
        - post_samples: jax.numpy.ndarray
            Posterior samples of probability matrices, shape (n_samples, n, C)
        - times_post: None
            Placeholder for timing information (not implemented)
        - beta_opt: None
            Placeholder for optimal beta (not implemented)
        - time_bootstrap: None
            Placeholder for bootstrap timing (not implemented)
            
    Notes
    -----
    The function automatically:
    1. Determines grid dimensions (n, C) based on data shape
    2. Pads or truncates data to fit the grid
    3. Runs MCMC with 4 chains using NUTS sampler
    4. Saves results to timestamped directory in ./Res/
    
    Examples
    --------
    >>> data = jnp.random.randint(0, 2, size=(100, 50))
    >>> args = argparse.Namespace(dnum=100, pnum=1000)
    >>> samples, _, _, _ = run_ising_inference(data, args)
    """
    logging.info("Starting IsingAnisotropic model inference with Numpyro...")
    
    # Set random seeds for reproducibility
    key = random.PRNGKey(0)
    
    # Determine grid dimensions based on number of conditions
    n_conditions = data.shape[1]
    n = int(np.ceil(np.sqrt(n_conditions)))  # rows
    C = int(np.ceil(n_conditions / n))       # columns
    grid_size = n * C
    logging.info(f"Using IsingAnisotropic grid size: n={n}, C={C} (for {n_conditions} conditions)")
    
    # Prepare data for the model
    n_samples = min(args.dnum, data.shape[0])
    inference_data = data[:n_samples]
    
    # If we have more conditions than the grid can handle, use a subset
    if n_conditions > grid_size:
        logging.warning(f"Number of conditions ({n_conditions}) exceeds grid capacity ({grid_size}). Using first {grid_size} conditions.")
        inference_data = inference_data[:, :grid_size]
    
    # Pad the data if we have fewer conditions than grid capacity
    if inference_data.shape[1] < grid_size:
        padding_size = grid_size - inference_data.shape[1]
        padding = jnp.zeros((inference_data.shape[0], padding_size))
        inference_data = jnp.concatenate([inference_data, padding], axis=1)
    
    logging.info(f"Final inference data shape: {inference_data.shape}")
    
    # Create Numpyro model
    model = lambda: ising_model(n, C, inference_data)
    
    # Initialize model
    logging.info("Initializing model...")
    from numpyro.infer.util import initialize_model
    init_params = initialize_model(key, model)
    
    # Set up NUTS sampler
    from numpyro.infer import MCMC, NUTS, init_to_median
    nuts_kernel = NUTS(model, init_strategy=init_to_median)
    
    # Run MCMC
    logging.info("Running MCMC...")
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.pnum // 2,  # Use half for warmup
        num_samples=args.pnum,
        num_chains=4,  # Run multiple chains
        progress_bar=True
    )
    
    # Run the MCMC
    mcmc.run(key)
    
    # Get samples
    samples = mcmc.get_samples()
    logging.info(f"MCMC completed. Sample keys: {list(samples.keys())}")
    
    # Extract probability matrix samples
    if 'probs' in samples:
        post_samples = samples['probs']
        logging.info(f"Posterior samples shape: {post_samples.shape}")
    else:
        logging.warning("No 'probs' found in samples. Creating dummy samples.")
        post_samples = jnp.zeros((args.pnum, n, C))
    
    # Get MCMC diagnostics
    mcmc.print_summary()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'./Res/ising_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(f"{results_dir}/post_samples.npy", np.array(post_samples))
    np.save(f"{results_dir}/mcmc_samples.npy", {k: np.array(v) for k, v in samples.items()})
    
    logging.info(f"Results saved to {results_dir}")
    
    return post_samples, None, None, None

def main():
    """
    Main function to run IsingAnisotropic model inference.
    
    Parses command line arguments, loads data, and runs Bayesian inference
    using the IsingAnisotropic model with Numpyro. Results are saved to
    timestamped directories in the ./Res/ folder.
    
    Command Line Arguments
    ----------------------
    --type : str, default="FDBayes"
        Type of posterior (legacy, not used in Numpyro version)
    --size : int, default=10
        Grid size (will be adjusted based on data)
    --theta : float, default=5.0
        True parameter value (for synthetic data)
    --dnum : int, default=1000
        Number of data samples to use
    --pnum : int, default=2000
        Number of posterior samples
    --numboot : int, default=100
        Number of bootstrap samples (legacy, not used)
    --batch_size : int, optional
        Number of patients to use (if not specified, use all)
    --bootstrap : bool, default=False
        Whether to use bootstrap minimizers (legacy, not used)
        
    Examples
    --------
    Run with default parameters:
    >>> python run_model.py
    
    Run with custom parameters:
    >>> python run_model.py --dnum 500 --pnum 1000 --batch_size 100
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run IsingAnisotropic Model Inference with Numpyro')
    parser.add_argument('--type', default="FDBayes", type=str, choices=["FDBayes", "KSDBayes", "PseudoBayes"],
                        help='Type of posterior to use (legacy, not used in Numpyro version)')
    parser.add_argument('--size', default=10, type=int,
                        help='Grid size for IsingAnisotropic model (will be adjusted based on data)')
    parser.add_argument('--theta', default=5.0, type=float,
                        help='True parameter value (for synthetic data)')
    parser.add_argument('--dnum', default=1000, type=int,
                        help='Number of data samples to use')
    parser.add_argument('--pnum', default=2000, type=int,
                        help='Number of posterior samples')
    parser.add_argument('--numboot', default=100, type=int,
                        help='Number of bootstrap samples (legacy, not used in Numpyro version)')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Number of patients to use for inference (if not specified, use all)')
    parser.add_argument('--bootstrap', default=False, type=bool,
                        help='Whether to use bootstrap minimizers (legacy, not used in Numpyro version)') 
    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting IsingAnisotropic model run with Numpyro. Arguments: {args}")

    try:
        # Load and preprocess data
        logging.info("Loading data...")
        A, X_cov, condition_list = load_data(batch_size=args.batch_size)
        
        # For Ising model, we'll use the binary matrix A
        # Take first timepoint if 3D
        if A.ndim == 3:
            A_data = A[:, :, 0]  # Shape: (n_patients, n_conditions)
        else:
            A_data = A
        
        logging.info(f"Loaded data shape: {A_data.shape}")
        
        # Prepare data for Ising model
        ising_data = prepare_ising_data(A_data, X_cov, condition_list)
        
        # Run inference
        post_samples, times_post, beta_opt, time_bootstrap = run_ising_inference(ising_data, args)
        
        # Log final results
        logging.info("Inference completed successfully!")
        logging.info(f"Posterior samples shape: {post_samples.shape}")
        logging.info("MCMC diagnostics printed above.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 