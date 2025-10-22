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
from numpyro import handlers
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
    Besag, York, Mollié (BYM) model for spatial smoothing with correlated delta effects.
    
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
    
    # Hyperprior for delta covariance matrix with shared variance and structured correlation
    # Single variance parameter for all conditions
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(1.0))
    # Global correlation strength
    rho = numpyro.sample("rho", dist.Uniform(-1.0, 1.0))
    # Condition-specific random effects (variance 0.01 => std 0.1)
    xi = numpyro.sample("xi", dist.Normal(0.0, jnp.sqrt(0.01)).expand([C]))
    
    # Build raw correlation matrix: R_ij = rho * xi_i * xi_j for i != j; set diagonal to 1
    outer_xi = jnp.outer(xi, xi)
    R_raw = rho * outer_xi
    R_raw = R_raw.at[jnp.diag_indices(C)].set(1.0)
    # Clip raw entries to [-1, 1]
    R_raw = jnp.clip(R_raw, -1.0, 1.0)
    # Ensure symmetry
    R = 0.5 * (R_raw + R_raw.T)
    # Set diagonal to exactly 1.0
    R = R.at[jnp.diag_indices(C)].set(1.0)

    # Expose correlation matrix in samples
    R = numpyro.deterministic("delta_corr", R)
    # Construct covariance: add diagonal regularization for numerical stability
    # This ensures PD: cov = sigma^2 * R + ridge * I where ridge dominates if R is near singular
    ridge = 1e-4  # Strong enough to guarantee PD without eigen-decomposition
    delta_cov = numpyro.deterministic("delta_cov", (sigma_delta ** 2) * R + ridge * jnp.eye(C))

    # --- Define Latent Patient Field phi using BYM factor approach ---
    
    # Base distribution for the patient field phi
    phi = numpyro.sample("phi", dist.Normal(0, 1.0).expand([I]))

    # Add structured energy (ICAR component) using NumPyro deterministic
    # This corresponds to the prior: phi_structured ~ N(0, (tau_s * L_pat)^-1)
    U_structured = numpyro.deterministic("U_structured", tau_s * (phi.T @ L_pat @ phi))
    numpyro.factor("structured_effect", -0.5 * U_structured)

    # Add unstructured energy (i.i.d. component) using NumPyro deterministic
    # This corresponds to the prior: phi_unstructured ~ N(0, (tau_u * I)^-1)
    U_unstructured = numpyro.deterministic("U_unstructured", tau_u * jnp.sum(phi**2))
    numpyro.factor("unstructured_effect", -0.5 * U_unstructured)

    # --- Correlated Delta Effects ---
    
    # Sample delta from multivariate normal with covariance structure
    delta = numpyro.sample("delta", dist.MultivariateNormal(jnp.zeros(C), delta_cov))

    # --- Construct Final Latent Field Lambda ---
    
    # Combine patient and column effects using NumPyro deterministic
    Lambda = numpyro.deterministic("Lambda", phi[:, None] + delta[None, :])

    # --- Likelihood ---
    
    # Connect the latent field to the observed binary data
    numpyro.sample("obs", dist.Bernoulli(logits=Lambda), obs=y)

def bym_ou_model(L_pat, y_3d, obs_mask, dt_matrix):
    """
    Temporal BYM model with OU dynamics on patient field phi only.
    y_3d: shape (I, C, T)
    obs_mask: shape (I, T) boolean mask for available visits
    dt_matrix: shape (I, T) integer deltas between successive viscounts (dt[:,0] used but typically 1)
    """
    I, C, T = y_3d.shape

    # --- Define Hyper-priors (same as BYM) ---
    tau_s = numpyro.sample("tau_s", dist.HalfCauchy(2.0))
    tau_u = numpyro.sample("tau_u", dist.HalfCauchy(2.0))

    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(1.0))
    rho = numpyro.sample("rho", dist.Uniform(-1.0, 1.0))
    xi = numpyro.sample("xi", dist.Normal(0.0, jnp.sqrt(0.01)).expand([C]))

    outer_xi = jnp.outer(xi, xi)
    R_raw = rho * outer_xi
    R_raw = R_raw.at[jnp.diag_indices(C)].set(1.0)
    R_raw = jnp.clip(R_raw, -1.0, 1.0)
    R = 0.5 * (R_raw + R_raw.T)
    R = R.at[jnp.diag_indices(C)].set(1.0)
    R = numpyro.deterministic("delta_corr", R)
    ridge = 1e-4
    delta_cov = numpyro.deterministic("delta_cov", (sigma_delta ** 2) * R + ridge * jnp.eye(C))
    delta = numpyro.sample("delta", dist.MultivariateNormal(jnp.zeros(C), delta_cov))

    # OU parameters for phi_t
    rho_ou = numpyro.sample("rho_ou", dist.Beta(2, 2))
    sigma_ou = numpyro.sample("sigma_ou", dist.HalfNormal(1.0))

    # t=0: base patient field
    phi_t = numpyro.sample("phi_0", dist.Normal(0.0, 1.0).expand([I]))
    U_structured_0 = numpyro.deterministic("U_structured_t0", tau_s * (phi_t.T @ L_pat @ phi_t))
    numpyro.factor("structured_effect_t0", -0.5 * U_structured_0)
    U_unstructured_0 = numpyro.deterministic("U_unstructured_t0", tau_u * jnp.sum(phi_t**2))
    numpyro.factor("unstructured_effect_t0", -0.5 * U_unstructured_0)
    Lambda_t = numpyro.deterministic("Lambda_0", phi_t[:, None] + delta[None, :])
    with handlers.mask(mask=obs_mask[:, 0][:, None]):
        numpyro.sample("obs_0", dist.Bernoulli(logits=Lambda_t), obs=y_3d[:, :, 0])

    # t>0 transitions on phi only
    for t in range(1, T):
        dt = dt_matrix[:, t]
        # ensure dt >= 1
        dt = jnp.maximum(dt, 1)
        rho_dt = rho_ou ** dt
        sigma_dt = sigma_ou * jnp.sqrt(1.0 - (rho_ou ** (2.0 * dt)))
        mean_phi = rho_dt * phi_t
        phi_t = numpyro.sample(f"phi_{t}", dist.Normal(mean_phi, sigma_dt).to_event(0))
        U_s = numpyro.deterministic(f"U_structured_t{t}", tau_s * (phi_t.T @ L_pat @ phi_t))
        numpyro.factor(f"structured_effect_t{t}", -0.5 * U_s)
        U_u = numpyro.deterministic(f"U_unstructured_t{t}", tau_u * jnp.sum(phi_t**2))
        numpyro.factor(f"unstructured_effect_t{t}", -0.5 * U_u)
        Lambda_t = numpyro.deterministic(f"Lambda_{t}", phi_t[:, None] + delta[None, :])
        with handlers.mask(mask=obs_mask[:, t][:, None]):
            numpyro.sample(f"obs_{t}", dist.Bernoulli(logits=Lambda_t), obs=y_3d[:, :, t])

def run_bym_ou_inference(data_3d, visit_mask, visit_times, args):
    """Run temporal BYM-OU inference with sparse Laplacian and OU on phi."""
    logging.info("Starting BYM-OU model inference with sparse Laplacian and OU on phi...")
    key = random.PRNGKey(0)

    # binarize data
    binary_data = jnp.where(data_3d > 0.5, 1, 0)
    I, C, T = binary_data.shape
    logging.info(f"Using BYM-OU model: I={I}, C={C}, T={T}")

    # Build patient Laplacian from KNN (same as BYM) using all I patients
    logging.info("Building patient graph Laplacian...")
    patient_knn = load_patient_knn(lazy_load=True)
    # If a mapping of filtered patients to global indices exists in visit_times' metadata,
    # we cannot access it here; instead, ensure indices we use are valid for the filtered set by remapping.
    # Build a local KNN by selecting neighbors among the first I indices of the global KNN and clipping to range.
    global_indices = patient_knn['indices']
    # Build a mask of valid neighbor indices (< I). Exclude self (first column) later.
    trimmed = np.where(global_indices < I, global_indices, 0)
    knn_indices = trimmed[:I, 1:11]
    n_patients = I
    rows = np.arange(n_patients).repeat(knn_indices.shape[1])
    cols = knn_indices.flatten()
    data = np.ones_like(rows, dtype=int)
    W_p = sp.coo_matrix((data, (rows, cols)), shape=(n_patients, n_patients))
    W_p = (W_p + W_p.T).astype(bool).astype(int)
    D_p = sp.diags(W_p.sum(axis=1).A1)
    L_pat = D_p - W_p
    L_pat_coo = L_pat.tocoo()
    L_pat_jax = BCOO((jnp.array(L_pat_coo.data), jnp.array(np.vstack((L_pat_coo.row, L_pat_coo.col)).T)), shape=L_pat_coo.shape)
    logging.info(f"Constructed sparse patient Laplacian L_pat with shape {L_pat_jax.shape}")

    # Build dt matrix from visit_times; ensure shape (I, T)
    vt = jnp.array(visit_times)
    mask = jnp.array(visit_mask).astype(bool)
    # compute dt as difference of successive times; for missing, propagate previous valid time
    vt_filled = jnp.where(mask, vt, jnp.nan)
    # forward-fill NaNs per patient
    def ffill(row):
        vals = row
        last = jnp.nan
        def body(carry, x):
            last = carry
            val = jnp.where(jnp.isnan(x), last, x)
            return val, val
        _, filled = jax.lax.scan(body, jnp.nan, vals)
        return filled
    vt_ff = jax.vmap(ffill)(vt_filled)
    # set first dt = 1, others = max(1, diff)
    diffs = jnp.concatenate([jnp.ones((I,1)), jnp.diff(jnp.nan_to_num(vt_ff, nan=0.0), axis=1)], axis=1)
    dt_matrix = jnp.maximum(diffs.astype(jnp.int32), 1)

    model = lambda: bym_ou_model(L_pat_jax, binary_data, mask, dt_matrix)
    kernel = NUTS(model)
    logging.info("Running MCMC for BYM-OU model with NUTS kernel...")
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shard_suffix = f"_shard_{args.shard_id}" if args.shard_id is not None else ""
    results_dir = f'./Res/bym_ou_{timestamp}{shard_suffix}'
    os.makedirs(results_dir, exist_ok=True)
    np.save(f"{results_dir}/mcmc_samples.npy", {k: np.array(v) for k, v in samples.items()})
    logging.info(f"Results saved to {results_dir}")
    post_samples = {k: np.array(v) for k, v in samples.items()}
    return post_samples, None, None, None
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
    parser.add_argument('--model', default='bym', type=str, choices=['bym','bym_ou'],
                        help='Choose model: bym (static) or bym_ou (temporal OU on phi).')
    args = parser.parse_args()

    numpyro.set_host_device_count(4)
    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting model run with Numpyro. Arguments: {args}")

    try:
        if args.model == 'bym':
            logging.info("Loading data for BYM (static) model...")
            A, X_cov, condition_list = load_data(
                patient_start_idx=args.start_idx,
                patient_end_idx=args.end_idx
            )
            # Take first timepoint if 3D
            if A.ndim == 3:
                A_data = A[:, :, 0]
            else:
                A_data = A
            logging.info(f"Loaded data shape: {A_data.shape}")
            gmrf_data = jnp.array(A_data, dtype=jnp.float32)
            post_samples, times_post, beta_opt, time_bootstrap = run_bym_inference(gmrf_data, args)
        else:
            logging.info("Loading data for BYM-OU (temporal) model with visit metadata...")
            A, X_cov, condition_list, visit_mask, visit_times, original_indices = load_data(
                patient_start_idx=args.start_idx,
                patient_end_idx=args.end_idx,
                return_time_meta=True,
                min_visits=2,
                return_index_map=True,
                top_k_by_visits=100,
                max_visits=10
            )
            if A.ndim != 3:
                raise ValueError("Temporal model requires 3D A matrix (I, C, T)")
            logging.info(f"Loaded temporal data shape: {A.shape}")
            post_samples, times_post, beta_opt, time_bootstrap = run_bym_ou_inference(A, visit_mask, visit_times, args)
        
        # Log final results
        logging.info("Inference completed successfully!")
        logging.info(f"Posterior samples keys: {list(post_samples.keys())}")
        logging.info("MCMC diagnostics printed above.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 