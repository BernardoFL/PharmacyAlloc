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
import torch
import torch.autograd as autograd
from datetime import datetime

# Import the required modules
from Posteriors import FDBayes, KSDBayes, PseudoBayes
from Models import IsingAnisotropic
from dataloader import load_data

def setup_logging(log_dir='logs'):
    """Set up logging configuration"""
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
    Convert the medical data to a format suitable for Ising model analysis.
    """
    logging.info("Preparing data for IsingAnisotropic model...")
    
    # Use the binary matrix A_sorted as the main data
    # A_sorted shape: (n_patients, n_conditions)
    # We'll treat each patient as a sample and conditions as variables
    
    # Convert to torch tensor
    data_tensor = torch.tensor(A_sorted, dtype=torch.float32)
    
    logging.info(f"Prepared Ising data shape: {data_tensor.shape}")
    logging.info(f"Data statistics: mean={data_tensor.mean():.4f}, std={data_tensor.std():.4f}")
    logging.info(f"Binary values: {torch.unique(data_tensor)}")
    
    return data_tensor

def run_ising_inference(data, args):
    """
    Run IsingAnisotropic model inference using FDBayes posterior.
    """
    logging.info("Starting IsingAnisotropic model inference...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Instantiate the IsingAnisotropic model
    # Use the size based on number of conditions (variables)
    n_conditions = data.shape[1]
    # For IsingAnisotropic, we need n (rows) and C (columns) parameters
    # We'll create a rectangular grid that best fits the number of conditions
    n = int(np.ceil(np.sqrt(n_conditions)))  # rows
    C = int(np.ceil(n_conditions / n))       # columns
    grid_size = n * C
    logging.info(f"Using IsingAnisotropic grid size: n={n}, C={C} (for {n_conditions} conditions)")
    
    model = IsingAnisotropic(n, C)
    
    # Set up prior and transition distribution
    prior = torch.distributions.MultivariateNormal(torch.zeros(C+1), 3*torch.eye(C+1))
    log_prior = lambda param: prior.log_prob(param).sum()
    transit_p = torch.distributions.MultivariateNormal(torch.zeros(C+1), 0.1 * torch.eye(C+1))
    
    # Create posterior
    if args.type == "FDBayes":
        posterior = FDBayes(model.ratio_m, model.ratio_p, model.stat_m, model.stat_p, log_prior)
    elif args.type == "KSDBayes":
        posterior = KSDBayes(model.ratio_m, model.stat_m, model.shift_p, log_prior)
    elif args.type == "PseudoBayes":
        posterior = PseudoBayes(model.pseudologlikelihood, log_prior)
    else:
        raise ValueError(f"Unknown posterior type: {args.type}")
    
    logging.info(f"Created {args.type} posterior")
    
    # Prepare data for the model
    # We'll use a subset of the data for inference
    n_samples = min(args.dnum, data.shape[0])
    inference_data = data[:n_samples]
    
    # If we have more conditions than the grid can handle, we'll use a subset
    if n_conditions > grid_size:
        logging.warning(f"Number of conditions ({n_conditions}) exceeds grid capacity ({grid_size}). Using first {grid_size} conditions.")
        inference_data = inference_data[:, :grid_size]
    
    # Pad the data if we have fewer conditions than grid capacity
    if inference_data.shape[1] < grid_size:
        padding_size = grid_size - inference_data.shape[1]
        padding = torch.zeros(inference_data.shape[0], padding_size)
        inference_data = torch.cat([inference_data, padding], dim=1)
    
    logging.info(f"Final inference data shape: {inference_data.shape}")
    
    # Initialize posterior with data
    posterior.set_X(inference_data)
    
    # Initial parameter
    p0 = prior.sample()
    
    # Minimize to get initial parameter
    logging.info("Running initial minimization...")
    p_init, _ = posterior.minimise(posterior.loss, p0, ite=5000, lr=0.01, loss_thin=100, progress=False)
    
    # Bootstrap minimizers and find optimal beta
    if args.bootstrap:
        logging.info("Running bootstrap minimizers...")
        time_start_1 = time.time()
        boot_minimisers, _ = posterior.bootstrap_minimisers(inference_data, args.numboot, lambda: p_init, lr=0.01)
        posterior.set_X(inference_data)
        beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
        time_end_1 = time.time()
        time0_beta = time_end_1 - time_start_1
        
        logging.info(f"Optimal beta: {beta_opt:.4f}")
        logging.info(f"Bootstrap time: {time0_beta:.2f}s")
    else:
        beta_opt = 1.0
        time0_beta = None
    
    # Sample from posterior
    logging.info("Sampling from posterior...")
    D = C + 1  # number of parameters
    post_samples = torch.zeros(10, args.pnum, D)
    times_post = torch.zeros(10)

    for i in range(10):
        time_start_2 = time.time()
        post_sample = posterior.sample_nuts(args.pnum, args.pnum, prior.sample(), beta=beta_opt)
        time_end_2 = time.time()
        
        post_samples[i] = post_sample
        times_post[i] = time_end_2 - time_start_2
        
        if (i + 1) % 2 == 0:
            logging.info(f"Completed {i + 1}/10 posterior sampling runs")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'./Res/ising_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(f"{results_dir}/post_samples.npy", post_samples.numpy())
    np.save(f"{results_dir}/times_post.npy", times_post.numpy())
    if args.bootstrap:
        np.save(f"{results_dir}/beta_opt.npy", beta_opt.numpy())
        np.save(f"{results_dir}/time_bootstrap.npy", time0_beta)
    
    logging.info(f"Results saved to {results_dir}")
    
    return post_samples, times_post, beta_opt, time0_beta

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run IsingAnisotropic Model Inference with FDBayes')
    parser.add_argument('--type', default="FDBayes", type=str, choices=["FDBayes", "KSDBayes", "PseudoBayes"],
                        help='Type of posterior to use')
    parser.add_argument('--size', default=10, type=int,
                        help='Grid size for IsingAnisotropic model (will be adjusted based on data)')
    parser.add_argument('--theta', default=5.0, type=float,
                        help='True parameter value (for synthetic data)')
    parser.add_argument('--dnum', default=1000, type=int,
                        help='Number of data samples to use')
    parser.add_argument('--pnum', default=2000, type=int,
                        help='Number of posterior samples')
    parser.add_argument('--numboot', default=100, type=int,
                        help='Number of bootstrap samples')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Number of patients to use for inference (if not specified, use all)')
    parser.add_argument('--bootstrap', default=False, type=bool,
                        help='Whether to use bootstrap minimizers') 
    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting IsingAnisotropic model run with arguments: {args}")

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
        logging.info(f"Average sampling time: {times_post.mean():.2f}s")
        if args.bootstrap:
            logging.info(f"Optimal beta: {beta_opt:.4f}")
            logging.info(f"Bootstrap time: {time_bootstrap:.2f}s")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 