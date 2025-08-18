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
from knn_utils import load_patient_knn, load_condition_knn

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

def create_neighbor_hash_tables(knn_indices, max_neighbors=10):
    """
    Create hash tables for caching nearest neighbors with k ≤ 10 restriction.
    
    Parameters
    ----------
    knn_indices : numpy.ndarray
        Array of shape (n_items, k_neighbors) with indices of nearest neighbors.
        It is assumed that the first column is the item itself and should be ignored.
    max_neighbors : int, default=10
        Maximum number of neighbors to cache per item (k ≤ 10 restriction).
        
    Returns
    -------
    dict
        A dictionary mapping item indices to lists of neighbor indices (max 10 per item).
    """
    if max_neighbors > 10:
        raise ValueError(f"max_neighbors must be ≤ 10, got {max_neighbors}")
    
    neighbor_tables = {}
    n_items = knn_indices.shape[0]
    
    for i in range(n_items):
        # Get neighbors (excluding self, which is typically the first column)
        neighbors = knn_indices[i, 1:].tolist()
        # Apply k ≤ 10 restriction
        neighbors = neighbors[:max_neighbors]
        neighbor_tables[i] = neighbors
    
    return neighbor_tables

def compute_neighbor_energy(Lambda, neighbor_table, beta_values=None):
    """
    Compute the energy contribution from neighbor interactions using hash table lookup.
    
    Parameters
    ----------
    Lambda : jax.numpy.ndarray
        Latent field matrix of shape (n_items, n_features) or (n_features, n_items).
    neighbor_table : dict
        Hash table mapping item indices to neighbor indices.
    beta_values : jax.numpy.ndarray, optional
        Beta values for scaling interactions. If None, uses uniform scaling.
        
    Returns
    -------
    float
        Total energy contribution from neighbor interactions.
    """
    total_energy = 0.0
    
    for item_idx, neighbors in neighbor_table.items():
        if len(neighbors) == 0:
            continue
            
        # Convert neighbors list to JAX array for proper indexing
        neighbors_array = jnp.array(neighbors)
        
        # Get the latent values for current item and its neighbors
        item_latent = Lambda[item_idx]
        neighbor_latents = Lambda[neighbors_array]
        
        # Compute squared differences with neighbors
        if beta_values is not None:
            # Use per-item beta values for scaling
            beta = beta_values[item_idx] if item_idx < len(beta_values) else 1.0
        else:
            beta = 1.0
            
        # Sum of squared differences with all neighbors
        # Handle both 1D and 2D cases - compute squared differences element-wise
        if item_latent.ndim == 1:
            # For 1D case (single row/column), compute squared differences directly
            squared_diffs = (item_latent - neighbor_latents) ** 2
        else:
            # For 2D case, sum across the feature dimension
            squared_diffs = jnp.sum((item_latent - neighbor_latents) ** 2, axis=1)
        
        energy_contribution = beta * jnp.sum(squared_diffs)
        total_energy += energy_contribution
    
    return total_energy

def gmrf_model_hash_tables(patient_neighbors, condition_neighbors, y):
    """
    Gaussian Markov Random Field model using hash tables for O(n) neighbor computation.
    
    Parameters
    ----------
    patient_neighbors : dict
        Hash table mapping patient indices to neighbor indices (max 10 per patient).
    condition_neighbors : dict
        Hash table mapping condition indices to neighbor indices (max 10 per condition).
    y : jax.numpy.ndarray
        Observed binary data matrix.
    """
    I, C = y.shape  # I = patients (rows), C = conditions (columns)

    # --- Define Priors ---

    # Patient/Row term with a more constrained Gaussian prior
    beta_pat = numpyro.sample("beta_pat", dist.Normal(0.0, 1.0))

    # Condition/Column term with a Horseshoe prior (non-centered parameterization)
    # Global shrinkage parameter (tau)
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    # Local shrinkage parameters (lambda_i)
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(1.0).expand([C]))
    # Sample beta parameter for patients from a standard Normal
    # Scale them to get the final beta parameters
    beta_cond = numpyro.deterministic("beta_cond",  tau * lambdas)

    # Latent Field
    Lambda = numpyro.sample("Lambda", dist.Normal(0, 1.0).expand([I, C]))

    # --- Impose GMRF Structure using Hash Tables ---
    
    # Vertical Energy (Row/Patient Interactions) - Governed by a single beta_pat
    # Compute energy for each column of Lambda using patient neighbor tables
    def vertical_energy_per_col(col_idx):
        col_latent = Lambda[:, col_idx]
        return compute_neighbor_energy(col_latent, patient_neighbors, None)
    
    # Use vmap to compute energy for all columns
    U_vertical_per_col = jax.vmap(vertical_energy_per_col)(jnp.arange(C))
    U_vertical = beta_pat * jnp.sum(U_vertical_per_col)
    numpyro.factor("v_interact", -0.5 * U_vertical)

    # Horizontal Energy (Column/Condition Interactions) - Governed by per-condition betas
    # Scale Lambda by condition betas
    Lambda_scaled = Lambda * beta_cond[None, :]  # Broadcasting beta across all patients
    
    def horizontal_energy_per_row(row_idx):
        row_latent = Lambda_scaled[row_idx, :]
        return compute_neighbor_energy(row_latent, condition_neighbors, None)
    
    # Use vmap to compute energy for all rows
    U_horizontal_per_row = jax.vmap(horizontal_energy_per_row)(jnp.arange(I))
    U_horizontal = jnp.sum(U_horizontal_per_row)
    numpyro.factor("h_interact", -0.5 * U_horizontal)

    # Likelihood
    numpyro.sample("obs", dist.Bernoulli(logits=Lambda), obs=y)

def run_gmrf_inference(data, args):
    """
    Run GMRF model inference using hash tables for O(n) neighbor computation.

    Performs pre-computation of neighbor hash tables and then
    runs Bayesian inference using NUTS.
    """
    logging.info("Starting GMRF model inference with hash table optimization...")
    key = random.PRNGKey(0)
    
    binary_data = jnp.where(data > 0.5, 1, 0)
    N, C = binary_data.shape
    logging.info(f"Using GMRF model: N={N} (patients), C={C} (conditions)")

    # --- Pre-computation Steps ---
    logging.info("Performing pre-computation for GMRF with hash tables...")
    
    # 1. Load KNN data
    patient_knn = load_patient_knn(lazy_load=True)
    condition_knn = load_condition_knn(lazy_load=True)

    # 2. Create neighbor hash tables with k ≤ 10 restriction
    logging.info("Creating neighbor hash tables with k ≤ 10 restriction...")
    patient_neighbors_full = create_neighbor_hash_tables(patient_knn['indices'], max_neighbors=10)
    condition_neighbors_full = create_neighbor_hash_tables(condition_knn['indices'], max_neighbors=10)
    
    # 3. Subset neighbor tables to match the data batch
    logging.info(f"Subsetting neighbor tables for batch size: N={N}, C={C}")
    patient_neighbors = {i: patient_neighbors_full[i] for i in range(N) if i in patient_neighbors_full}
    condition_neighbors = {i: condition_neighbors_full[i] for i in range(C) if i in condition_neighbors_full}

    logging.info(f"Patient neighbor table size: {len(patient_neighbors)}")
    logging.info(f"Condition neighbor table size: {len(condition_neighbors)}")
    
    # Log some statistics about neighbor counts
    patient_neighbor_counts = [len(neighbors) for neighbors in patient_neighbors.values()]
    condition_neighbor_counts = [len(neighbors) for neighbors in condition_neighbors.values()]
    logging.info(f"Patient neighbors per item - min: {min(patient_neighbor_counts)}, max: {max(patient_neighbor_counts)}, mean: {np.mean(patient_neighbor_counts):.2f}")
    logging.info(f"Condition neighbors per item - min: {min(condition_neighbor_counts)}, max: {max(condition_neighbor_counts)}, mean: {np.mean(condition_neighbor_counts):.2f}")

    # --- NumPyro Model Inference ---
    model = lambda: gmrf_model_hash_tables(patient_neighbors, condition_neighbors, binary_data)
    
    kernel = HMCECS(NUTS(model), num_blocks=10)
    
    logging.info("Running MCMC for GMRF model with HMCECS kernel...")
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
    results_dir = f'./Res/gmrf_{timestamp}{shard_suffix}'
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