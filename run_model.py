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

def create_adjacency_matrix(knn_indices):
    """
    Create a sparse, symmetric, binary adjacency matrix from KNN indices.
    
    Parameters
    ----------
    knn_indices : numpy.ndarray
        Array of shape (n_items, k_neighbors) with indices of nearest neighbors.
        It is assumed that the first column is the item itself and should be ignored.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse, symmetric, binary adjacency matrix (n_items, n_items).
    """
    n_items = knn_indices.shape[0]
    rows = np.repeat(np.arange(n_items), knn_indices.shape[1] - 1)
    cols = knn_indices[:, 1:].flatten()
    data = np.ones_like(rows)
    
    W = sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.int32)
    
    # Make the matrix symmetric
    W = W + W.T
    W[W > 1] = 1
    
    return W

def compute_graph_laplacian(W):
    """
    Compute the graph Laplacian from a sparse adjacency matrix.
    L = D - W, where D is the diagonal degree matrix of W.
    """
    D = sp.diags(W.sum(axis=1).A1, format='csr')
    L = D - W
    return L

def gmrf_model(L_rows, L_cols, y):
    """
    Gaussian Markov Random Field model with separable row and column dependencies.
    
    Parameters
    ----------
    L_rows : jax.experimental.sparse.BCOO
        Graph Laplacian for rows (patients).
    L_cols : jax.experimental.sparse.BCOO
        Graph Laplacian for columns (conditions).
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

    # --- Impose GMRF Structure ---
    
    # Define a function for the quadratic form using efficient sparse matrix-vector products
    def sparse_quadratic_form(x, L):
        """Computes x^T L x for a vector x and sparse matrix L."""
        return x.T @ (L @ x)

    # Vertical Energy (Row/Patient Interactions) - Governed by a single J_v
    # Use vmap to compute the quadratic form for each column of Lambda
    U_vertical_per_col = jax.vmap(sparse_quadratic_form, in_axes=(1, None), out_axes=0)(Lambda, L_rows)
    U_vertical = beta_pat * jnp.sum(U_vertical_per_col)
    numpyro.factor("v_interact", -0.5 * U_vertical)

    # Horizontal Energy (Column/Condition Interactions) - Governed by per-condition betas
    Lambda_scaled = Lambda * beta_cond[None, :]  # Broadcasting beta across all patients
    # Use vmap to compute the quadratic form for each row of Lambda_scaled
    U_horizontal_per_row = jax.vmap(sparse_quadratic_form, in_axes=(0, None), out_axes=0)(Lambda_scaled, L_cols)
    U_horizontal = jnp.sum(U_horizontal_per_row)
    numpyro.factor("h_interact", -0.5 * U_horizontal)

    # Likelihood
    numpyro.sample("obs", dist.Bernoulli(logits=Lambda), obs=y)

def run_gmrf_inference(data, args):
    """
    Run GMRF model inference.

    Performs pre-computation of adjacency and Laplacian matrices, and then
    runs Bayesian inference using NUTS.
    """
    logging.info("Starting GMRF model inference with Numpyro...")
    key = random.PRNGKey(0)
    
    binary_data = jnp.where(data > 0.5, 1, 0)
    N, C = binary_data.shape
    logging.info(f"Using GMRF model: N={N} (patients), C={C} (conditions)")

    # --- Pre-computation Steps ---
    logging.info("Performing pre-computation for GMRF...")
    
    # 1. Load KNN data
    patient_knn = load_patient_knn(lazy_load=True)
    condition_knn = load_condition_knn(lazy_load=True)

    # 2. Create Adjacency Matrices from the full dataset
    logging.info("Creating full adjacency matrices from KNN data...")
    W_rows_full = create_adjacency_matrix(patient_knn['indices'])
    W_cols_full = create_adjacency_matrix(condition_knn['indices'])
    
    # 3. Subset adjacency matrices to match the data batch
    # This assumes that the `data` (A_data from main) corresponds to the first N patients
    # and C conditions from the original dataset.
    logging.info(f"Subsetting adjacency matrices for batch size: N={N}, C={C}")
    W_rows = W_rows_full[:N, :N]
    W_cols = W_cols_full[:C, :C]

    # 4. Compute Graph Laplacians for the batch
    logging.info("Computing graph Laplacians for the batch...")
    L_rows_sp = compute_graph_laplacian(W_rows)
    L_cols_sp = compute_graph_laplacian(W_cols)
    
    # Convert to JAX BCOO sparse arrays
    L_rows = BCOO.from_scipy_sparse(L_rows_sp)
    L_cols = BCOO.from_scipy_sparse(L_cols_sp)

    logging.info(f"Row Laplacian shape: {L_rows.shape}")
    logging.info(f"Column Laplacian shape: {L_cols.shape}")

    # --- NumPyro Model Inference ---
    model = lambda: gmrf_model(L_rows, L_cols, binary_data)
    
    kernel = HMCECS(model)
    
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