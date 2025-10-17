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
import scipy.sparse as sp
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax import random
from datetime import datetime
from collections import defaultdict

# Import Numpyro modules
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMCECS, init_to_median
from numpyro.infer.util import initialize_model

# Import the required modules
from Source.NumpyroDistributions import IsingAnisotropicDistribution
from dataloader import load_data
from Source.JAXFDBayes import JAXFDBayes
from knn_utils import load_patient_knn

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

def bym_model(L_pat, y):
    """
    Besag, York, Mollié (BYM) model for spatial smoothing.
    
    Parameters
    ----------
    L_pat : jax.experimental.sparse.BCOO
        Sparse graph Laplacian of the patient k-NN graph.
    y : jax.numpy.ndarray
        Observed binary data matrix.
    """
    I, C = y.shape

    # --- Define Hyper-priors ---
    
    # Precision for the structured spatial component (ICAR)
    tau_s = numpyro.sample("tau_s", dist.HalfCauchy(2.0))
    
    # Precision for the unstructured component (i.i.d. noise)
    tau_u = numpyro.sample("tau_u", dist.HalfCauchy(2.0))
    
    # Simple column effects for each condition
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfCauchy(1.0))
    delta = numpyro.sample("delta", dist.Normal(0, sigma_delta).expand([C]))

    # --- Define Latent Patient Field phi using BYM factor approach ---
    
    # Base distribution for the patient field phi
    phi = numpyro.sample("phi", dist.Normal(0, 1.0).expand([I]))

    # Add structured energy (ICAR component)
    # This corresponds to the prior: phi_structured ~ N(0, (tau_s * L_pat)^-1)
    U_structured = tau_s * (phi.T @ L_pat @ phi)
    numpyro.factor("structured_effect", -0.5 * U_structured)

    # Add unstructured energy (i.i.d. component)
    # This corresponds to the prior: phi_unstructured ~ N(0, (tau_u * I)^-1)
    U_unstructured = tau_u * jnp.sum(phi**2)
    numpyro.factor("unstructured_effect", -0.5 * U_unstructured)

    # --- Construct Final Latent Field Lambda ---
    
    # Combine patient and column effects
    Lambda = phi[:, None] + delta[None, :]

    # --- Likelihood ---
    
    # Connect the latent field to the observed binary data
    numpyro.sample("obs", dist.Bernoulli(logits=Lambda), obs=y)

def run_bym_inference(data, args):
    """
    Run BYM model inference using a sparse graph Laplacian.
    """
    logging.info("Starting BYM model inference with sparse Laplacian...")
    key = random.PRNGKey(0)
    
    binary_data = jnp.where(data > 0.5, 1, 0)
    I, C = binary_data.shape
    logging.info(f"Using BYM model: I={I} (patients), C={C} (conditions)")

    # --- Build Patient Graph Laplacian L_pat ---
    logging.info("Building patient graph Laplacian...")
    
    patient_knn = load_patient_knn(lazy_load=True)
    # Use 10 nearest neighbors, exclude self (1st column)
    knn_indices = patient_knn['indices'][:I, 1:11] 
    
    n_patients = I
    rows = np.arange(n_patients).repeat(knn_indices.shape[1])
    cols = knn_indices.flatten()
    data = np.ones_like(rows, dtype=int)
    
    W_p = sp.coo_matrix((data, (rows, cols)), shape=(n_patients, n_patients))
    # Symmetrize the adjacency matrix
    W_p = (W_p + W_p.T).astype(bool).astype(int)
    
    # Compute degree matrix and graph Laplacian
    D_p = sp.diags(W_p.sum(axis=1).A1)
    L_pat = D_p - W_p
    
    # Convert to JAX BCOO sparse format
    L_pat_coo = L_pat.tocoo()
    L_pat_jax = BCOO((jnp.array(L_pat_coo.data), 
                      jnp.array(np.vstack((L_pat_coo.row, L_pat_coo.col)).T)), 
                     shape=L_pat_coo.shape)
    
    logging.info(f"Constructed sparse patient Laplacian L_pat with shape {L_pat_jax.shape}")
    
    # --- NumPyro Model Inference ---
    model = lambda: bym_model(L_pat_jax, binary_data)
    
    kernel = NUTS(model)
    
    logging.info("Running MCMC for BYM model with NUTS kernel...")
    mcmc = MCMC(
        kernel,
        num_warmup=10000,
        num_samples=args.pnum,
        num_chains=4,
        progress_bar=True,
        jit_model_args=True,
        chain_method='parallel'
    )
    mcmc.run(key)
    
    samples = mcmc.get_samples()
    logging.info(f"MCMC completed. Sample keys: {list(samples.keys())}")
    
    mcmc.print_summary()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shard_suffix = f"_shard_{args.shard_id}" if args.shard_id is not None else ""
    results_dir = f'./Res/bym_{timestamp}{shard_suffix}'
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(f"{results_dir}/mcmc_samples.npy", {k: np.array(v) for k, v in samples.items()})
    
    logging.info(f"Results saved to {results_dir}")
    
    # To maintain function signature from main
    post_samples = {k: np.array(v) for k, v in samples.items()}
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
    parser.add_argument('--start_idx', default=None, type=int,
                        help='Starting index for patient data slicing.')
    parser.add_argument('--end_idx', default=None, type=int,
                        help='Ending index for patient data slicing.')
    parser.add_argument('--shard_id', default=None, type=int,
                        help='Identifier for the current shard.')
    args = parser.parse_args()

    numpyro.set_host_device_count(4)
    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting model run with Numpyro. Arguments: {args}")

    try:
        # Load and preprocess data
        logging.info("Loading data...")
        A, X_cov, condition_list = load_data(
            patient_start_idx=args.start_idx, 
            patient_end_idx=args.end_idx
        )
        
        # Take first timepoint if 3D
        if A.ndim == 3:
            A_data = A[:, :, 0]  # Shape: (n_patients, n_conditions)
        else:
            A_data = A
        
        logging.info(f"Loaded data shape: {A_data.shape}")
                    
        # Prepare data for GMRF model (binary format)
        gmrf_data = jnp.array(A_data, dtype=jnp.float32)
            
        # Run GMRF inference
        post_samples, times_post, beta_opt, time_bootstrap = run_bym_inference(gmrf_data, args)
        
        # Log final results
        logging.info("Inference completed successfully!")
        logging.info(f"Posterior samples keys: {list(post_samples.keys())}")
        logging.info("MCMC diagnostics printed above.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 