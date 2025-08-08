#!/usr/bin/env python

import os
import sys
import argparse
import logging
import numpy as np
import jax
import jax.numpy as jnp
from datetime import datetime
import time

# Add paths for the required modules
sys.path.append('./Source')
sys.path.append('./_dependency')

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO messages

def setup_logging(log_dir='logs'):
    """
    Set up logging configuration for the product distance matrix precomputation.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'product_distance_precomputation_{timestamp}.log')

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

def create_product_distance_matrix_optimized(patient_distances, condition_distances, chunk_size=1000):
    """
    Create product distance matrix between (i,c) and (j,c') using d = sqrt(d_1^2 + d_2^2).
    Memory-efficient version that processes in chunks to avoid OOM.
    
    Parameters
    ----------
    patient_distances : numpy.ndarray
        N x N distance matrix between patients
    condition_distances : numpy.ndarray
        C x C distance matrix between conditions
    chunk_size : int
        Size of chunks for processing to manage memory
        
    Returns
    -------
    numpy.ndarray
        (N*C) x (N*C) product distance matrix
    """
    N = patient_distances.shape[0]
    C = condition_distances.shape[0]
    
    logging.info(f"Creating product distance matrix: {N} patients × {C} conditions = {N*C} total dimensions")
    logging.info(f"Using memory-efficient chunked computation with chunk_size={chunk_size}")
    
    # Convert to float32 for memory efficiency
    patient_distances = patient_distances.astype(np.float32)
    condition_distances = condition_distances.astype(np.float32)
    
    # Square the distance matrices
    patient_squared = patient_distances ** 2
    condition_squared = condition_distances ** 2
    
    # Initialize product distance matrix
    product_distances = np.zeros((N*C, N*C), dtype=np.float32)
    
    # Process in chunks to manage memory
    # We'll process the matrix row by row in chunks
    total_rows = N * C
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_count = (chunk_start // chunk_size) + 1
        
        logging.info(f"Processing chunk {chunk_count}/{total_chunks}: rows {chunk_start} to {chunk_end-1}")
        
        # For each row in this chunk
        for row_idx in range(chunk_start, chunk_end):
            # Convert row index back to patient and condition indices
            patient_i = row_idx // C
            condition_i = row_idx % C
            
            # Compute this entire row efficiently
            for col_idx in range(total_rows):
                # Convert column index back to patient and condition indices
                patient_j = col_idx // C
                condition_j = col_idx % C
                
                # Product metric: d = sqrt(d_1^2 + d_2^2)
                d1_squared = patient_squared[patient_i, patient_j]
                d2_squared = condition_squared[condition_i, condition_j]
                product_distances[row_idx, col_idx] = np.sqrt(d1_squared + d2_squared)
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
    
    # Add small jitter to ensure positive definiteness for GP kernel
    jitter = 1e-6
    product_distances = product_distances + jitter * np.eye(N*C, dtype=np.float32)
    
    logging.info(f"Product distance matrix created: {product_distances.shape}")
    logging.info(f"Distance statistics: min={product_distances.min():.6f}, max={product_distances.max():.6f}, mean={product_distances.mean():.6f}")
    
    return product_distances

def save_product_distance_matrix(product_distances, output_file):
    """
    Save the product distance matrix to a file.
    
    Parameters
    ----------
    product_distances : numpy.ndarray
        The product distance matrix to save
    output_file : str
        Path to the output file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the matrix
        np.save(output_file, product_distances)
        logging.info(f"Product distance matrix saved to: {output_file}")
        
        # Also save metadata
        metadata = {
            'shape': product_distances.shape,
            'dtype': str(product_distances.dtype),
            'min': float(product_distances.min()),
            'max': float(product_distances.max()),
            'mean': float(product_distances.mean()),
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = output_file.replace('.npy', '_metadata.npz')
        np.savez(metadata_file, **metadata)
        logging.info(f"Metadata saved to: {metadata_file}")
        
    except Exception as e:
        logging.error(f"Error saving product distance matrix: {str(e)}")
        raise

def main():
    """
    Main function to precompute the product distance matrix.
    """
    parser = argparse.ArgumentParser(description='Precompute Product Distance Matrix')
    parser.add_argument('--output_file', default='Data/product_distance_matrix.npy',
                        help='Output file path for the product distance matrix')
    parser.add_argument('--chunk_size', default=100, type=int,
                        help='Chunk size for processing (to manage memory, smaller = less memory)')
    parser.add_argument('--max_patients', default=None, type=int,
                        help='Maximum number of patients to use (for testing)')
    parser.add_argument('--max_conditions', default=None, type=int,
                        help='Maximum number of conditions to use (for testing)')
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting product distance matrix precomputation. Arguments: {args}")
    
    try:
        # Load precomputed distance matrices
        logging.info("Loading precomputed distance matrices...")
        patient_distances, condition_distances = load_precomputed_distance_matrices()
        
        if patient_distances is None or condition_distances is None:
            logging.error("Could not load precomputed distance matrices. Exiting.")
            return
        
        # Apply size limits if specified (for testing)
        if args.max_patients is not None:
            patient_distances = patient_distances[:args.max_patients, :args.max_patients]
            logging.info(f"Limited to {args.max_patients} patients")
        
        if args.max_conditions is not None:
            condition_distances = condition_distances[:args.max_conditions, :args.max_conditions]
            logging.info(f"Limited to {args.max_conditions} conditions")
        
        # Create product distance matrix
        start_time = time.time()
        product_distances = create_product_distance_matrix_optimized(
            patient_distances, condition_distances, args.chunk_size
        )
        end_time = time.time()
        
        logging.info(f"Product distance matrix computation completed in {end_time - start_time:.2f} seconds")
        
        # Save the matrix
        save_product_distance_matrix(product_distances, args.output_file)
        
        # Print summary
        total_size_gb = product_distances.nbytes / (1024**3)
        logging.info(f"Product distance matrix summary:")
        logging.info(f"  Shape: {product_distances.shape}")
        logging.info(f"  Memory usage: {total_size_gb:.2f} GB")
        logging.info(f"  Computation time: {end_time - start_time:.2f} seconds")
        logging.info(f"  Saved to: {args.output_file}")
        
    except Exception as e:
        logging.error(f"An error occurred during precomputation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
