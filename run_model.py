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

def create_padded_neighbor_arrays(neighbor_lists, num_items, max_k=10):
    """
    Creates padded, dense JAX arrays for neighbors and masks from a dictionary.
    
    Parameters
    ----------
    neighbor_lists : dict
        Dictionary mapping item indices to lists of neighbors.
    num_items : int
        The total number of items (e.g., N patients or C conditions).
    max_k : int
        The maximum number of neighbors to pad to.
        
    Returns
    -------
    (jax.numpy.ndarray, jax.numpy.ndarray)
        A tuple containing the padded neighbor array and the boolean mask array.
    """
    # Initialize padded array with -1 as a sentinel value
    neighbors_padded = np.full((num_items, max_k), -1, dtype=np.int32)
    neighbors_mask = np.zeros((num_items, max_k), dtype=bool)

    for i, neighbors in neighbor_lists.items():
        num_neighbors = len(neighbors)
        if num_neighbors > 0:
            neighbors_padded[i, :num_neighbors] = neighbors
            neighbors_mask[i, :num_neighbors] = True
            
    return jnp.array(neighbors_padded), jnp.array(neighbors_mask)

def compute_neighbor_energy_vectorized(Lambda, neighbors_padded, neighbors_mask, beta_values=None):
    """
    Computes the energy contribution from neighbor interactions in a fully vectorized manner.
    
    Parameters
    ----------
    Lambda : jax.numpy.ndarray
        Latent field vector of shape (num_items,).
    neighbors_padded : jax.numpy.ndarray
        Padded neighbor indices, shape (num_items, max_k).
    neighbors_mask : jax.numpy.ndarray
        Boolean mask for real neighbors, shape (num_items, max_k).
    beta_values : jax.numpy.ndarray, optional
        Beta values for scaling interactions. If None, uses uniform scaling.
        
    Returns
    -------
    float
        Total energy contribution from neighbor interactions.
    """
    # Get latent values for all items. Shape: (num_items, 1)
    item_latents = Lambda[:, None]
    
    # Handle the padding index (-1). Replace with 0; the mask will negate its contribution.
    valid_neighbors = jnp.where(neighbors_padded == -1, 0, neighbors_padded)
    # Get latent values for all neighbors. Shape: (num_items, max_k)
    neighbor_latents = Lambda[valid_neighbors]

    # Compute squared differences. Shape: (num_items, max_k)
    squared_diffs = (item_latents - neighbor_latents) ** 2
    
    # Apply mask to zero out padded neighbors
    masked_diffs = squared_diffs * neighbors_mask
    
    # Sum over the neighbors dimension. Shape: (num_items,)
    energy_per_item = jnp.sum(masked_diffs, axis=1)
    
    # Apply beta scaling if provided
    if beta_values is not None:
        scaled_energy_per_item = energy_per_item * beta_values
    else:
        scaled_energy_per_item = energy_per_item
        
    # Return the total sum
    return jnp.sum(scaled_energy_per_item)

def gmrf_model_vectorized(patient_padded, patient_mask, condition_padded, condition_mask, y):
    """
    Gaussian Markov Random Field model using fully vectorized neighbor computation.
    
    Parameters
    ----------
    patient_padded : jax.numpy.ndarray
        Padded neighbor indices for patients.
    patient_mask : jax.numpy.ndarray
        Boolean mask for patient neighbors.
    condition_padded : jax.numpy.ndarray
        Padded neighbor indices for conditions.
    condition_mask : jax.numpy.ndarray
        Boolean mask for condition neighbors.
    y : jax.numpy.ndarray
        Observed binary data matrix.
    """
    I, C = y.shape  # I = patients (rows), C = conditions (columns)

    # --- Define Priors ---
    beta_pat = numpyro.sample("beta_pat", dist.Normal(0.0, 1.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(1.0).expand([C]))
    beta_cond = numpyro.deterministic("beta_cond",  tau * lambdas)
    Lambda = numpyro.sample("Lambda", dist.Normal(0, 1.0).expand([I, C]))

    # --- Impose GMRF Structure using Vectorized Computation ---
    
    # Vertical Energy (Patient Interactions)
    # vmap over columns of Lambda. `col_latent` has shape (I,).
    def vertical_energy_per_col(col_latent):
        return compute_neighbor_energy_vectorized(col_latent, patient_padded, patient_mask, None)

    U_vertical_per_col = jax.vmap(vertical_energy_per_col, in_axes=1, out_axes=0)(Lambda)
    U_vertical = beta_pat * jnp.sum(U_vertical_per_col)
    numpyro.factor("v_interact", -0.5 * U_vertical)

    # Horizontal Energy (Condition Interactions)
    # Pre-scale Lambda by condition-specific betas
    Lambda_scaled = Lambda * beta_cond[None, :]
    
    # vmap over rows of Lambda_scaled. `row_latent` has shape (C,).
    def horizontal_energy_per_row(row_latent):
        return compute_neighbor_energy_vectorized(row_latent, condition_padded, condition_mask, None)

    U_horizontal_per_row = jax.vmap(horizontal_energy_per_row, in_axes=0, out_axes=0)(Lambda_scaled)
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

    # 2. Create neighbor hash tables from the required subset to save memory
    logging.info("Creating neighbor hash tables for the data subset (k <= 10)...")
    
    # Process patients: slice the required data, create the table, and then free memory
    patient_indices_subset = patient_knn['indices'][:N]
    del patient_knn  # Free memory
    patient_neighbors_list = create_neighbor_hash_tables(patient_indices_subset, max_neighbors=10)
    del patient_indices_subset  # Free memory
    
    # Process conditions: slice the required data, create the table, and then free memory
    condition_indices_subset = condition_knn['indices'][:C]
    del condition_knn  # Free memory
    condition_neighbors_list = create_neighbor_hash_tables(condition_indices_subset, max_neighbors=10)
    del condition_indices_subset  # Free memory

    # Pre-convert neighbor lists to padded JAX arrays for vectorization
    patient_padded, patient_mask = create_padded_neighbor_arrays(patient_neighbors_list, N)
    condition_padded, condition_mask = create_padded_neighbor_arrays(condition_neighbors_list, C)

    logging.info(f"Patient padded array shape: {patient_padded.shape}")
    logging.info(f"Condition padded array shape: {condition_padded.shape}")
    
    # Log some statistics about neighbor counts
    patient_neighbor_counts = jnp.sum(patient_mask, axis=1)
    condition_neighbor_counts = jnp.sum(condition_mask, axis=1)
    logging.info(f"Patient neighbors per item - min: {jnp.min(patient_neighbor_counts)}, max: {jnp.max(patient_neighbor_counts)}, mean: {jnp.mean(patient_neighbor_counts):.2f}")
    logging.info(f"Condition neighbors per item - min: {jnp.min(condition_neighbor_counts)}, max: {jnp.max(condition_neighbor_counts)}, mean: {jnp.mean(condition_neighbor_counts):.2f}")

    # --- NumPyro Model Inference ---
    model = lambda: gmrf_model_vectorized(patient_padded, patient_mask, condition_padded, condition_mask, binary_data)
    
    kernel = NUTS(model)
    
    logging.info("Running MCMC for GMRF model with NUTS kernel...")
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