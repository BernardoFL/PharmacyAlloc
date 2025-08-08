#!/usr/bin/env python

import os
import sys
import subprocess
import argparse
import logging
import numpy as np
import jax
import jax.numpy as jnp
from datetime import datetime
import pickle
from pathlib import Path
import ot  # Python Optimal Transport library

# Add paths for the required modules
sys.path.append('./Source')
sys.path.append('./_dependency')

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO messages

from dataloader import load_data

def setup_logging(log_dir='logs'):
    """
    Set up logging configuration for the sharded model inference.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'sharded_model_run_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")
    return log_file

def get_total_patients():
    """
    Get the total number of patients in the dataset.
    """
    try:
        # Load a small batch to get the total count
        A, X_cov, condition_list = load_data(batch_size=1)
        # Load full data to get total count
        A_full, X_cov_full, condition_list_full = load_data(batch_size=None)
        total_patients = A_full.shape[0]
        logging.info(f"Total patients in dataset: {total_patients}")
        return total_patients
    except Exception as e:
        logging.error(f"Error getting total patients: {str(e)}")
        raise

def run_shard(shard_start, shard_size, args, shard_id):
    """
    Run the model on a specific shard of patients.
    
    Parameters:
    -----------
    shard_start : int
        Starting index for this shard
    shard_size : int
        Number of patients in this shard
    args : argparse.Namespace
        Command line arguments
    shard_id : int
        Identifier for this shard
        
    Returns:
    --------
    str
        Path to the results directory for this shard
    """
    logging.info(f"Running shard {shard_id}: patients {shard_start} to {shard_start + shard_size - 1}")
    
    # Create modified arguments for this shard
    shard_args = [
        'python', 'run_model.py',
        '--pnum', str(args.pnum),
        '--batch_size', str(shard_size),
        '--start_index', str(shard_start)
    ]
    
    # Set environment variable to indicate shard start
    env = os.environ.copy()
    env['SHARD_START'] = str(shard_start)
    env['SHARD_ID'] = str(shard_id)
    
    # Run the model for this shard
    try:
        logging.info(f"Starting shard {shard_id} with command: {' '.join(shard_args)}")
        
        # Run subprocess with real-time output display
        process = subprocess.Popen(
            shard_args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Display output in real-time
        for line in process.stdout:
            print(f"[Shard {shard_id}] {line.rstrip()}")
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            logging.info(f"Shard {shard_id} completed successfully")
        else:
            raise subprocess.CalledProcessError(return_code, shard_args)
        
        # Find the results directory (look for the most recent hierarchical_gp directory)
        results_dir = None
        for item in os.listdir('./Res'):
            if item.startswith('hierarchical_gp_') and os.path.isdir(os.path.join('./Res', item)):
                if results_dir is None or item > results_dir:
                    results_dir = item
        
        if results_dir:
            return os.path.join('./Res', results_dir)
        else:
            raise FileNotFoundError("Could not find results directory for shard")
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Shard {shard_id} failed with return code: {e.returncode}")
        raise
    except Exception as e:
        logging.error(f"Shard {shard_id} failed with exception: {str(e)}")
        raise

def compute_wasserstein_barycenter(mcmc_samples, weights=None):
    """
    Compute the Wasserstein barycenter between multiple empirical distributions
    given by MCMC samples.
    
    Parameters:
    -----------
    mcmc_samples : list of np.ndarray
        List of arrays where each array contains MCMC samples from a distribution
    weights : np.ndarray, optional
        Weights for each distribution in the barycenter computation
        
    Returns:
    --------
    np.ndarray
        Samples from the Wasserstein barycenter distribution
    """
    n_distributions = len(mcmc_samples)
    
    if weights is None:
        weights = np.ones(n_distributions) / n_distributions
    
    # Process each set of MCMC samples
    all_samples = []
    all_weights = []
    
    for samples in mcmc_samples:
        # Flatten if multidimensional while preserving sample dimension
        flat_samples = samples.reshape(len(samples), -1)
        # Use uniform weights for empirical distribution
        sample_weights = np.ones(len(samples)) / len(samples)
        
        all_samples.append(flat_samples)
        all_weights.append(sample_weights)
    
    # Compute Wasserstein barycenter using empirical samples directly
    barycenter = ot.lp.free_support_barycenter(all_samples, all_weights, weights)
    
    # Reshape back to original dimensions if needed
    original_shape = mcmc_samples[0].shape
    if len(original_shape) > 2:
        barycenter = barycenter.reshape(-1, *original_shape[1:])
    
    return barycenter

def combine_shard_results(shard_results, num_shards, args):
    """
    Combine results from all shards using Wasserstein barycenters.
    
    This approach computes the Wasserstein barycenter between the chain trajectories
    from different shards, which provides a more principled way to combine the
    posterior distributions.
    
    Parameters:
    -----------
    shard_results : list
        List of paths to shard result directories
    num_shards : int
        Number of shards used
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Combined posterior samples using Wasserstein barycenters
    """
    logging.info(f"Combining results from {num_shards} shards using Wasserstein barycenters")
    
    # Load results from each shard
    all_samples = []
    for i, shard_dir in enumerate(shard_results):
        try:
            samples_file = os.path.join(shard_dir, 'post_samples.npy')
            shard_samples = np.load(samples_file, allow_pickle=True).item()
            all_samples.append(shard_samples)
            logging.info(f"Loaded shard {i+1} samples with keys: {list(shard_samples.keys())}")
        except Exception as e:
            logging.error(f"Error loading shard {i+1} results: {str(e)}")
            raise
    
    # Initialize combined samples dictionary
    combined_samples = {}
    
    # For GP hyperparameters (eta, ell, sigma_noise), compute Wasserstein barycenter
    for param in ['eta', 'ell', 'sigma_noise']:
        if param in all_samples[0]:
            param_samples = []
            for shard_sample in all_samples:
                param_samples.append(shard_sample[param])
            
            # Compute Wasserstein barycenter for the parameter
            combined_samples[param] = compute_wasserstein_barycenter(param_samples)
            logging.info(f"Combined {param} using Wasserstein barycenter: shape {combined_samples[param].shape}")
    
    # For latent field f, compute Wasserstein barycenter
    if 'f' in all_samples[0]:
        f_samples = []
        for shard_sample in all_samples:
            f_samples.append(shard_sample['f'])
        
        # Compute Wasserstein barycenter for the latent field
        combined_samples['f'] = compute_wasserstein_barycenter(f_samples)
        combined_samples['f_original'] = f_samples  # Keep original for comparison
        logging.info(f"Combined f using Wasserstein barycenter: shape {combined_samples['f'].shape}")
    
    # For probabilities p, compute Wasserstein barycenter in logit space
    if 'p' in all_samples[0]:
        p_samples = []
        for shard_sample in all_samples:
            # Convert to logits for better interpolation
            logits = np.log(shard_sample['p'] / (1 - shard_sample['p'] + 1e-8))
            p_samples.append(logits)
        
        # Compute Wasserstein barycenter in logit space
        logits_combined = compute_wasserstein_barycenter(p_samples)
        
        # Convert back to probabilities
        combined_samples['p'] = 1 / (1 + np.exp(-logits_combined))
        combined_samples['p_original'] = [s['p'] for s in all_samples]  # Keep original for comparison
        logging.info(f"Combined p using Wasserstein barycenter: shape {combined_samples['p'].shape}")
    
    # Add metadata about the combination
    combined_samples['num_shards'] = num_shards
    combined_samples['shard_results_dirs'] = shard_results
    combined_samples['combination_method'] = 'wasserstein_barycenter'
    combined_samples['total_samples'] = len(all_samples[0]['f']) * num_shards if 'f' in all_samples[0] else 0
    
    return combined_samples

def save_combined_results(combined_samples, args, failed_shards=None):
    """
    Save the combined results to a timestamped directory.
    
    Parameters:
    -----------
    combined_samples : dict
        Combined posterior samples
    args : argparse.Namespace
        Command line arguments
    failed_shards : list, optional
        List of failed shard IDs
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'./Res/sharded_hierarchical_gp_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save combined samples
    np.save(f"{results_dir}/combined_post_samples.npy", combined_samples)
    
    # Save metadata
    metadata = {
        'num_shards': combined_samples['num_shards'],
        'shard_results_dirs': combined_samples['shard_results_dirs'],
        'args': vars(args),
        'timestamp': timestamp,
        'failed_shards': failed_shards or []
    }
    
    with open(f"{results_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    # Create summary report
    create_summary_report(results_dir, combined_samples, args, failed_shards)
    
    logging.info(f"Combined results saved to {results_dir}")
    return results_dir

def create_summary_report(results_dir, combined_samples, args, failed_shards=None):
    """
    Create a summary report of the sharded run.
    
    Parameters:
    -----------
    results_dir : str
        Directory to save the report
    combined_samples : dict
        Combined posterior samples
    args : argparse.Namespace
        Command line arguments
    failed_shards : list, optional
        List of failed shard IDs
    """
    report_file = os.path.join(results_dir, 'summary_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("SHARDED HIERARCHICAL GP MODEL RUN SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total shards: {combined_samples['num_shards']}\n")
        f.write(f"Shard size: {args.shard_size}\n")
        f.write(f"Posterior samples per shard: {args.pnum}\n")
        f.write(f"Total posterior samples: {combined_samples.get('total_samples', 'N/A')}\n")
        f.write(f"Combination method: Wasserstein barycenter\n\n")
        
        f.write("Wasserstein Barycenter Details:\n")
        f.write("  - Using empirical distributions from MCMC samples\n")
        f.write("  - Equal weights for all shards\n")
        f.write("  - Logit transform used for probability parameters\n")
        f.write("  - Direct computation without density estimation\n\n")
        
        if failed_shards:
            f.write(f"Failed shards: {failed_shards}\n")
            f.write(f"Successful shards: {combined_samples['num_shards'] - len(failed_shards)}\n\n")
        
        f.write("Combined samples keys:\n")
        for key in combined_samples.keys():
            if key not in ['num_shards', 'shard_results_dirs', 'combination_method', 'total_samples']:
                if isinstance(combined_samples[key], np.ndarray):
                    f.write(f"  {key}: {combined_samples[key].shape}\n")
                    if key in ['f', 'p']:
                        f.write(f"  {key}_original: stored separately for comparison\n")
                else:
                    f.write(f"  {key}: {type(combined_samples[key])}\n")
        
        f.write(f"\nShard result directories:\n")
        for i, shard_dir in enumerate(combined_samples['shard_results_dirs']):
            f.write(f"  Shard {i+1}: {shard_dir}\n")
        
        f.write("\nNotes:\n")
        f.write("- Wasserstein barycenter provides a principled way to combine posterior distributions\n")
        f.write("- Original chain trajectories are preserved for each shard\n")
        f.write("- Direct use of empirical distributions from MCMC samples\n")
        f.write("- Special handling for bounded parameters (probabilities) via logit transform\n")
    
    logging.info(f"Summary report saved to {report_file}")

def main():
    """
    Main function to run sharded Bayesian model inference.
    
    Divides the data into shards of 1,000 patients and runs the hierarchical GP model
    on each shard. Then combines the results with likelihood elevated to the power
    of the number of shards.
    """
    parser = argparse.ArgumentParser(description='Run Sharded Bayesian Model Inference')
    parser.add_argument('--pnum', default=500, type=int,
                        help='Number of posterior samples per shard')
    parser.add_argument('--shard_size', default=1000, type=int,
                        help='Number of patients per shard')
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting sharded model run. Arguments: {args}")
    
    try:
        # Get total number of patients
        total_patients = get_total_patients()
        
        # Calculate number of shards
        num_shards = (total_patients + args.shard_size - 1) // args.shard_size
        logging.info(f"Total patients: {total_patients}")
        logging.info(f"Shard size: {args.shard_size}")
        logging.info(f"Number of shards: {num_shards}")
        
        # Run each shard
        shard_results = []
        failed_shards = []
        
        for shard_id in range(num_shards):
            shard_start = shard_id * args.shard_size
            shard_size = min(args.shard_size, total_patients - shard_start)
            
            logging.info(f"Processing shard {shard_id + 1}/{num_shards}")
            logging.info(f"Shard {shard_id + 1}: patients {shard_start} to {shard_start + shard_size - 1}")
            
            try:
                # Run the shard
                shard_result_dir = run_shard(shard_start, shard_size, args, shard_id + 1)
                shard_results.append(shard_result_dir)
                logging.info(f"Shard {shard_id + 1} completed successfully")
            except Exception as e:
                logging.error(f"Shard {shard_id + 1} failed: {str(e)}")
                failed_shards.append(shard_id + 1)
                continue
        
        if failed_shards:
            logging.warning(f"Failed shards: {failed_shards}")
            if len(failed_shards) == num_shards:
                raise RuntimeError("All shards failed. Cannot proceed with combination.")
            logging.info(f"Proceeding with {len(shard_results)} successful shards out of {num_shards}")
        
        # Combine results from all shards
        logging.info("All shards completed. Combining results...")
        combined_samples = combine_shard_results(shard_results, len(shard_results), args)
        
        # Save combined results
        final_results_dir = save_combined_results(combined_samples, args, failed_shards)
        
        logging.info("Sharded inference completed successfully!")
        logging.info(f"Final results saved to: {final_results_dir}")
        logging.info(f"Combined samples keys: {list(combined_samples.keys())}")
        
    except Exception as e:
        logging.error(f"An error occurred during sharded execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
