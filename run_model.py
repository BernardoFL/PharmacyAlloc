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

def extract_batch_distance_matrices(patient_distances, condition_distances, batch_size, actual_conditions):
    """
    Extract the relevant subset of distance matrices for a given batch size.
    
    Parameters
    ----------
    patient_distances : numpy.ndarray
        Full N x N patient distance matrix
    condition_distances : numpy.ndarray
        Full C x C condition distance matrix
    batch_size : int
        Number of patients to use (first batch_size patients)
    actual_conditions : int
        Number of conditions in the actual data
        
    Returns
    -------
    tuple
        (batch_patient_distances, batch_condition_distances) - subset of distance matrices
    """
    if batch_size is None or batch_size >= patient_distances.shape[0]:
        # Use full matrices for patients, but extract conditions based on actual data
        batch_patient_distances = patient_distances
    else:
        # Extract subset for the first batch_size patients
        batch_patient_distances = patient_distances[:batch_size, :batch_size]
    
    # Extract subset for the actual number of conditions in the data
    if actual_conditions <= condition_distances.shape[0]:
        batch_condition_distances = condition_distances[:actual_conditions, :actual_conditions]
    else:
        logging.warning(f"Actual conditions ({actual_conditions}) exceeds condition distance matrix size ({condition_distances.shape[0]}). Using full matrix.")
        batch_condition_distances = condition_distances
    
    logging.info(f"Extracted batch distance matrices: patients {batch_patient_distances.shape}, conditions {batch_condition_distances.shape}")
    
    return batch_patient_distances, batch_condition_distances

def load_precomputed_distance_matrices():
    """
    Load precomputed patient and condition distance matrices.
    
    Returns
    -------
    tuple
        (patient_distances, condition_distances) - both as numpy arrays
    """
    try:
        # Load patient distance matrix
        patient_file = "Data/patient_knn_distances.npy"
        if os.path.exists(patient_file):
            patient_distances = np.load(patient_file)
            logging.info(f"Loaded precomputed patient distance matrix: {patient_distances.shape}")
        else:
            logging.warning(f"Patient distance file not found: {patient_file}")
            return None, None
        
        # Load condition distance matrix
        condition_file = "Data/condition_knn_distances.npy"
        if os.path.exists(condition_file):
            condition_distances = np.load(condition_file)
            logging.info(f"Loaded precomputed condition distance matrix: {condition_distances.shape}")
        else:
            logging.warning(f"Condition distance file not found: {condition_file}")
            return None, None
            
        return patient_distances, condition_distances
        
    except Exception as e:
        logging.warning(f"Error loading precomputed distance matrices: {str(e)}")
        return None, None

def load_or_create_product_distance_matrix(patient_distances, condition_distances):
    """
    Load precomputed product distance matrix or create it if not available.
    
    Parameters
    ----------
    patient_distances : numpy.ndarray
        N x N distance matrix between patients
    condition_distances : numpy.ndarray
        C x C distance matrix between conditions
        
    Returns
    -------
    jax.numpy.ndarray
        (N*C) x (N*C) product distance matrix
    """
    N = patient_distances.shape[0]
    C = condition_distances.shape[0]
    
    # Try to load precomputed matrix (check both .npy and .h5 formats)
    product_matrix_file_npy = "Data/product_distance_matrix.npy"
    product_matrix_file_h5 = "Data/product_distance_matrix.h5"
    
    # Try HDF5 format first (more memory efficient)
    if os.path.exists(product_matrix_file_h5):
        try:
            import h5py
            with h5py.File(product_matrix_file_h5, 'r') as f:
                product_distances = f['product_distance_matrix'][:]
                logging.info(f"Loaded precomputed product distance matrix (HDF5): {product_distances.shape}")
                
                # Check if the loaded matrix matches our current dimensions
                expected_shape = (N*C, N*C)
                if product_distances.shape == expected_shape:
                    logging.info("Precomputed matrix dimensions match. Using cached version.")
                    return jnp.array(product_distances, dtype=jnp.float32)
                else:
                    logging.warning(f"Precomputed matrix shape {product_distances.shape} doesn't match expected {expected_shape}. Recomputing...")
        except Exception as e:
            logging.warning(f"Error loading precomputed HDF5 matrix: {str(e)}. Trying NPY format...")
    
    # Try NPY format as fallback
    if os.path.exists(product_matrix_file_npy):
        try:
            product_distances = np.load(product_matrix_file_npy)
            logging.info(f"Loaded precomputed product distance matrix (NPY): {product_distances.shape}")
            
            # Check if the loaded matrix matches our current dimensions
            expected_shape = (N*C, N*C)
            if product_distances.shape == expected_shape:
                logging.info("Precomputed matrix dimensions match. Using cached version.")
                return jnp.array(product_distances, dtype=jnp.float32)
            else:
                logging.warning(f"Precomputed matrix shape {product_distances.shape} doesn't match expected {expected_shape}. Recomputing...")
        except Exception as e:
            logging.warning(f"Error loading precomputed NPY matrix: {str(e)}. Recomputing...")
    
    # If we get here, we need to create the matrix
    logging.info(f"Creating product distance matrix: {N} patients × {C} conditions = {N*C} total dimensions")
    
    # Initialize product distance matrix
    product_distances = np.zeros((N*C, N*C))
    
    # For each pair of patient-condition combinations (i,c) and (j,c')
    for i in range(N):
        for c in range(C):
            for j in range(N):
                for c_prime in range(C):
                    # Linear indices for (i,c) and (j,c')
                    idx1 = i * C + c
                    idx2 = j * C + c_prime
                    
                    # Product metric: d = d_1^2 + d_2^2
                    d1_squared = patient_distances[i, j] ** 2
                    d2_squared = condition_distances[c, c_prime] ** 2
                    product_distances[idx1, idx2] = jnp.sqrt(d1_squared + d2_squared)
    
    # Add small jitter to ensure positive definiteness for GP kernel
    jitter = 1e-6
    product_distances = product_distances + jitter * np.eye(N*C)
    
    logging.info(f"Product distance matrix created: {product_distances.shape}")
    logging.info(f"Distance statistics: min={product_distances.min():.6f}, max={product_distances.max():.6f}, mean={product_distances.mean():.6f}")
    
    return jnp.array(product_distances, dtype=jnp.float32)


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



def hierarchical_gp_model(D_product, y):
    """
    Hierarchical Bayesian model with Gaussian Process prior using product distance metric.
    
    The model implements:
    1. GP hyperparameters for the product space (eta, ell, sigma_noise)
    2. GP prior on the flattened patient-condition space using product distance matrix
    3. Bernoulli likelihood for binary outcomes
    
    Parameters
    ----------
    D_product : jax.numpy.ndarray
        (N*C) x (N*C) product distance matrix between (i,c) and (j,c')
    y : jax.numpy.ndarray
        N x C binary outcome matrix for N patients under C conditions
    """
    # Infer dimensions from inputs
    N, C = y.shape
    total_dim = N * C
    
    # Define GP hyperparameters for the product space
    eta = numpyro.sample("eta", dist.HalfCauchy(2.0))  # Amplitude
    ell = numpyro.sample("ell", dist.HalfCauchy(1.0))  # Length-scale
    sigma_noise = numpyro.sample("sigma_noise", dist.HalfCauchy(0.5))  # GP noise
    
    # Construct GP kernel using product distance matrix
    # K[i,j] = eta^2 * exp(-0.5 * (D_product[i,j] / ell)^2)
    K = eta**2 * jnp.exp(-0.5 * (D_product / ell)**2)
    
    # Add noise term and jitter to ensure positive definiteness
    K = K + jnp.eye(total_dim) * (sigma_noise**2 + 1e-6)
    
    # Sample latent field from GP prior
    f = numpyro.sample("f", dist.MultivariateNormal(
        loc=jnp.zeros(total_dim), 
        covariance_matrix=K
    ))
    
    # Reshape f back to N x C for the likelihood
    f_reshaped = f.reshape(N, C)
    
    # Transform to probabilities using sigmoid
    p = jax.nn.sigmoid(f_reshaped)
    
    # Bernoulli likelihood for observed binary data
    numpyro.sample("obs", dist.Bernoulli(probs=p), obs=y)
    
    return p


def run_hierarchical_gp_inference(data, args):
    """
    Run hierarchical Bayesian model with Gaussian Process prior on patient effects.
    
    Performs Bayesian inference using NUTS (No-U-Turn Sampler) with multiple chains.
    The model uses a GP prior on patient effects based on patient distance matrix.
    
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
    logging.info("Starting hierarchical GP model inference with Numpyro...")
    
    # Set random seeds for reproducibility
    key = random.PRNGKey(0)
    
    # Ensure data is binary {0, 1}
    binary_data = jnp.where(data > 0.5, 1, 0)
    
    # Determine dimensions
    N, C = binary_data.shape  # N = patients (rows), C = conditions (columns)
    
    logging.info(f"Using hierarchical GP model: N={N} (patients), C={C} (conditions)")
    logging.info(f"Data matrix size: {N} × {C} = {N * C}")
    
    # Load precomputed distance matrices
    full_patient_distances, full_condition_distances = load_precomputed_distance_matrices()
    
    if full_patient_distances is None or full_condition_distances is None:
        logging.error("Precomputed distance matrices not found or could not be loaded. Exiting.")
        return None, None, None, None

    # Extract batch-specific distance matrices
    patient_distances, condition_distances = extract_batch_distance_matrices(
        full_patient_distances, full_condition_distances, args.batch_size, C
    )

    # Load or create product distance matrix
    D_p = load_or_create_product_distance_matrix(patient_distances, condition_distances)
    
    logging.info(f"Product distance matrix shape: {D_p.shape}")
    logging.info(f"Distance statistics: min={D_p.min():.4f}, max={D_p.max():.4f}, mean={D_p.mean():.4f}")
    
    # Prepare data for the model
    inference_data = binary_data
    
    logging.info(f"Data shape: {inference_data.shape}")
    logging.info(f"Binary values: {jnp.unique(inference_data)}")
    logging.info(f"Data statistics: mean={inference_data.mean():.4f}, std={inference_data.std():.4f}")

    # Create Numpyro model
    model = lambda: hierarchical_gp_model(D_p, inference_data)
    
    # Set up NUTS sampler with better initialization for GP models
    from numpyro.infer import MCMC, NUTS, init_to_uniform
    nuts_kernel = NUTS(
        model, 
        init_strategy=init_to_uniform,
        max_tree_depth=8,  # Moderate tree depth
        target_accept_prob=0.8,  # Standard acceptance rate
        step_size=0.05  # Even smaller step size for stability
    )
    
    # Run MCMC
    logging.info("Running MCMC for hierarchical GP model with 4 parallel chains...")
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.pnum // 2,  # Use half for warmup for better adaptation
        num_samples=args.pnum,
        num_chains=4,  # Use 4 chains in parallel
        progress_bar=True,
        chain_method='parallel'  # Enable parallel execution
    )
    
    # Run the MCMC
    mcmc.run(key)
    
    # Get samples
    samples = mcmc.get_samples()
    logging.info(f"MCMC completed. Sample keys: {list(samples.keys())}")
    
    # Extract parameter samples
    f_samples = samples.get('f', jnp.zeros((args.pnum * 4, N*C)))  # 4 chains
    
    # Reshape f samples back to N x C for each sample
    f_reshaped_samples = f_samples.reshape(args.pnum * 4, N, C)
    
    # Compute probability samples from latent field
    p_samples = jax.nn.sigmoid(f_reshaped_samples)
    
    post_samples = {
        'eta': samples.get('eta', jnp.zeros(args.pnum * 4)),
        'ell': samples.get('ell', jnp.zeros(args.pnum * 4)),
        'sigma_noise': samples.get('sigma_noise', jnp.zeros(args.pnum * 4)),
        'f': f_samples,
        'f_reshaped': f_reshaped_samples,
        'p': p_samples
    }
    
    # Get MCMC diagnostics
    mcmc.print_summary()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'./Res/hierarchical_gp_{timestamp}'
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
    parser.add_argument('--start_index', default=0, type=int,
                        help='Starting index for patient selection (for sharding)')
    parser.add_argument('--bootstrap', default=False, type=bool,
                        help='Whether to use bootstrap minimizers (legacy, not used in Numpyro version)') 
    args = parser.parse_args()

    numpyro.set_host_device_count(4)
    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting model run with Numpyro. Arguments: {args}")

    try:
        # Load and preprocess data
        logging.info("Loading data...")
        A, X_cov, condition_list = load_data(batch_size=args.batch_size, start_index=args.start_index)
        
        # Take first timepoint if 3D
        if A.ndim == 3:
            A_data = A[:, :, 0]  # Shape: (n_patients, n_conditions)
        else:
            A_data = A
        
        logging.info(f"Loaded data shape: {A_data.shape}")
                    
        # Prepare data for hierarchical GP model (binary format)
        hierarchical_gp_data = jnp.array(A_data, dtype=jnp.float32)
            
        # Run hierarchical GP inference
        post_samples, times_post, beta_opt, time_bootstrap = run_hierarchical_gp_inference(hierarchical_gp_data, args)
        
        # Log final results
        logging.info("Inference completed successfully!")
        logging.info(f"Posterior samples keys: {list(post_samples.keys())}")
        logging.info("MCMC diagnostics printed above.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 