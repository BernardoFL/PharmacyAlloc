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
from Source.JAXFDBayes import JAXFDBayes

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



def gmrf_model(I, C, data=None):
    """
    Numpyro model for Gaussian Markov Random Field (GMRF) inference.
    
    The model implements:
    1. Hyperpriors for sigma, gamma, and beta parameters
    2. Latent grid x with GMRF prior
    3. Probability grid p = sigmoid(x)
    4. Likelihood for observed binary data
    
    Parameters
    ----------
    I : int
        Number of rows (patients)
    C : int
        Number of columns (conditions)
    data : jax.numpy.ndarray, optional
        Observed binary data of shape (I, C) with values in {0, 1}
    """
    # Define hyperpriors
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    
    # Normal-InverseGamma hyperprior for gamma
    gamma_mean = numpyro.sample("gamma_mean", dist.Normal(0, 5))
    gamma_std2 = numpyro.sample("gamma_std2", dist.InverseGamma(2, 2))
    gamma_std = jnp.sqrt(gamma_std2)
    gamma = numpyro.sample("gamma", dist.Normal(gamma_mean, gamma_std))
    
    # Horseshoe prior for betas
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    lambda_betas = numpyro.sample("lambda_betas", dist.HalfCauchy(jnp.ones(C-1)))
    beta = numpyro.sample("beta", dist.Normal(0, tau * lambda_betas))
    
    # Sample the latent grid x from independent Normal (potential term)
    # This efficiently implements the potential term -x^2/(2*sigma^2)
    x = numpyro.sample("x", dist.Normal(0, sigma).expand([I, C]))
    
    # Add GMRF interaction terms using numpyro.factor
    
    # Vertical interactions (between rows): gamma * sum(x[i,c] * x[i+1,c])
    v_potential = gamma * jnp.sum(x[:-1, :] * x[1:, :])
    numpyro.factor("v_interact", v_potential)
    
    # Horizontal interactions (between columns): sum(beta[c] * sum(x[i,c] * x[i,c+1]))
    h_potential = jnp.sum(beta * jnp.sum(x[:, :-1] * x[:, 1:], axis=0))
    numpyro.factor("h_interact", h_potential)
    
    # Transform to probabilities using sigmoid with clipping to prevent extreme values
    p_raw = jax.nn.sigmoid(x)
    p = jnp.clip(p_raw, 1e-7, 1.0 - 1e-7)
    
    # Add likelihood for observed data if provided
    if data is not None:
        # Use binomial likelihood for binary data
        numpyro.sample("obs", dist.Binomial(total_count=1, probs=p), obs=data)
    
    return p


def run_gmrf_inference(data, args):
    """
    Run Gaussian Markov Random Field (GMRF) model inference using Numpyro MCMC.
    
    Performs Bayesian inference on the GMRF model using NUTS (No-U-Turn Sampler)
    with multiple chains. The model samples probability matrices and uses binomial likelihood.
    
    Parameters:
    -----------
    data : jax.numpy.ndarray
        Input binary data matrix of shape (n_patients, n_conditions) with values in {0, 1}
    args : argparse.Namespace
        Command line arguments containing inference parameters
        
    Returns:
    -------
    tuple
        A tuple containing posterior samples and timing information
    """
    logging.info("Starting GMRF model inference with Numpyro...")
    
    # Set random seeds for reproducibility
    key = random.PRNGKey(0)
    
    # Ensure data is binary {0, 1}
    binary_data = jnp.where(data > 0.5, 1, 0)
    
    # Determine grid dimensions
    I, C = binary_data.shape  # I = patients (rows), C = conditions (columns)
    
    logging.info(f"Using GMRF grid size: I={I} (patients), C={C} (conditions)")
    logging.info(f"Grid size: {I} × {C} = {I * C}")
    
    # Prepare data for the model
    inference_data = binary_data
    
    logging.info(f"Data shape: {inference_data.shape}")
    logging.info(f"Binary values: {jnp.unique(inference_data)}")
    logging.info(f"Data statistics: mean={inference_data.mean():.4f}, std={inference_data.std():.4f}")

    # Create Numpyro model
    model = lambda: gmrf_model(I, C, inference_data)
    
    # Set up NUTS sampler with memory-efficient settings
    from numpyro.infer import MCMC, NUTS, init_to_median
    nuts_kernel = NUTS(
        model, 
        init_strategy=init_to_median,
        max_tree_depth=8,  # Reduce tree depth to save memory
        target_accept_prob=0.8  # Slightly lower acceptance rate for efficiency
    )
    
    # Run MCMC
    logging.info("Running MCMC for GMRF model...")
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.pnum // 4,  # Use quarter for warmup to save memory
        num_samples=args.pnum,
        num_chains=1,  # Use single chain to reduce memory
        progress_bar=True
    )
    
    # Run the MCMC
    mcmc.run(key)
    
    # Get samples
    samples = mcmc.get_samples()
    logging.info(f"MCMC completed. Sample keys: {list(samples.keys())}")
    
    # Extract parameter samples
    x_samples = samples.get('x', jnp.zeros((args.pnum, I, C)))
    p_raw = jax.nn.sigmoid(x_samples)
    p_samples = jnp.clip(p_raw, 1e-7, 1.0 - 1e-7)
    
    post_samples = {
        'sigma': samples.get('sigma', jnp.zeros(args.pnum)),
        'gamma': samples.get('gamma', jnp.zeros(args.pnum)),
        'gamma_mean': samples.get('gamma_mean', jnp.zeros(args.pnum)),
        'gamma_std2': samples.get('gamma_std2', jnp.zeros(args.pnum)),
        'beta': samples.get('beta', jnp.zeros((args.pnum, C-1))),
        'tau': samples.get('tau', jnp.zeros(args.pnum)),
        'lambda_betas': samples.get('lambda_betas', jnp.zeros((args.pnum, C-1))),
        'x': x_samples,
        'p': p_samples
    }
    
    # Get MCMC diagnostics
    mcmc.print_summary()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'./Res/gmrf_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(f"{results_dir}/post_samples.npy", {k: np.array(v) for k, v in post_samples.items()})
    np.save(f"{results_dir}/mcmc_samples.npy", {k: np.array(v) for k, v in samples.items()})
    
    logging.info(f"Results saved to {results_dir}")
    
    return post_samples, None, None, None

def main():
    """
    Main function to run Bayesian model inference (Ising or GMRF).
    
    Parses command line arguments, loads data, and runs Bayesian inference
    using either the IsingAnisotropic model or Gaussian Markov Random Field (GMRF)
    model with Numpyro. Results are saved to timestamped directories in the ./Res/ folder.
    
    Command Line Arguments
    ----------------------
    --type : str, default="FDBayes"
        Type of posterior (legacy, not used in Numpyro version)
    --theta : float, default=5.0
        True parameter value (for synthetic data)
    --pnum : int, default=500
        Number of posterior samples
    --numboot : int, default=100
        Number of bootstrap samples (legacy, not used)
    --batch_size : int, default=20
        Number of patients to use for inference (controls both data loading and inference size)
    --bootstrap : bool, default=False
        Whether to use bootstrap minimizers (legacy, not used)
        
    Examples
    --------
    Run model with default parameters:
    >>> python run_model.py
    
    Run  model with custom parameters:
    >>> python run_model.py --pnum 1000 --batch_size 100
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Bayesian Model Inference with Numpyro')
    parser.add_argument('--type', default="FDBayes", type=str, choices=["FDBayes", "KSDBayes", "PseudoBayes"],
                        help='Type of posterior to use (legacy, not used in Numpyro version)')

    parser.add_argument('--theta', default=5.0, type=float,
                        help='True parameter value (for synthetic data)')

    parser.add_argument('--pnum', default=500, type=int,
                        help='Number of posterior samples')
    parser.add_argument('--numboot', default=100, type=int,
                        help='Number of bootstrap samples (legacy, not used in Numpyro version)')
    parser.add_argument('--batch_size', default=20, type=int,
                        help='Number of patients to use for inference (if not specified, use 20)')
    parser.add_argument('--bootstrap', default=False, type=bool,
                        help='Whether to use bootstrap minimizers (legacy, not used in Numpyro version)') 
    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting model run with Numpyro. Arguments: {args}")

    try:
        # Load and preprocess data
        logging.info("Loading data...")
        A, X_cov, condition_list = load_data(batch_size=args.batch_size)
        
        # Take first timepoint if 3D
        if A.ndim == 3:
            A_data = A[:, :, 0]  # Shape: (n_patients, n_conditions)
        else:
            A_data = A
        
        logging.info(f"Loaded data shape: {A_data.shape}")
                    
        # Prepare data for GMRF model (binary format)
        gmrf_data = jnp.array(A_data, dtype=jnp.float32)
            
        # Run GMRF inference
        post_samples, times_post, beta_opt, time_bootstrap = run_gmrf_inference(gmrf_data, args)
        
        # Log final results
        logging.info("Inference completed successfully!")
        logging.info(f"Posterior samples keys: {list(post_samples.keys())}")
        logging.info("MCMC diagnostics printed above.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 