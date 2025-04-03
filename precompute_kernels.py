import os
import argparse
from multiprocessing import Pool, cpu_count
import multiprocessing
from concurrent.futures import ProcessPoolExecutor  # Add this import
from tqdm import tqdm
import numpy as np

# Set start method to 'spawn' at the beginning of the script
multiprocessing.set_start_method('spawn', force=True)

# Configure JAX to use CPU and disable GPU/TPU warnings
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import jax
jax.config.update('jax_default_device', jax.devices('cpu')[0])
import jax.numpy as jnp
from jax import jit, vmap, lax, pmap
import numpy as np
from dataloader import load_data
import logging
from datetime import datetime
import os
from functools import partial
import time
from itertools import combinations_with_replacement
from string_kernel import compute_similarity_multisentence, sample_random_kmers
import jax.random as random

def setup_logging(log_dir='logs'):
    """Set up logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'kernel_precomputation_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

@partial(jit, static_argnums=(0,))
def compute_level_match(level: int, atc1: jnp.ndarray, atc2: jnp.ndarray) -> jnp.ndarray:
    """Compute matches for a specific ATC level with static shapes"""
    # Pre-slice arrays outside of JIT
    atc1_slice = atc1[:, :level+1]
    atc2_slice = atc2[:, :level+1]
    
    matches = jnp.all(
        atc1_slice[:, None, :] == atc2_slice[None, :, :],
        axis=-1
    )
    return matches * (level + 1)

@jit
def compute_similarity_matrix(atc1: jnp.ndarray, atc2: jnp.ndarray) -> jnp.ndarray:
    """Compute similarity matrix between two sets of ATC codes with static shapes"""
    n1, n2 = atc1.shape[0], atc2.shape[0]
    
    # Initialize array to store all level matches
    all_matches = jnp.zeros((4, n1, n2))
    
    for i in range(4):
        # Slice to current level for both arrays
        atc1_level = atc1[:, :i+1]
        atc2_level = atc2[:, :i+1]
        
        # Compute matches for this level
        matches = jnp.all(
            atc1_level[:, None, :] == atc2_level[None, :, :],
            axis=-1
        )
        # Store matches with level weight
        all_matches = all_matches.at[i].set(matches * (i + 1))
    
    # Take maximum across all levels
    return jnp.max(all_matches, axis=0)

@jit
def compute_atc_similarity(code1: jnp.ndarray, code2: jnp.ndarray) -> float:
    """Vectorized ATC code similarity computation"""
    matches = jnp.array([
        jnp.all(code1[:level+1] == code2[:level+1]) * (level + 1)
        for level in range(4)
    ])
    return jnp.max(matches)

def compute_full_kernel(atc_codes: jnp.ndarray) -> jnp.ndarray:
    """Compute the full kernel matrix without batching"""
    return compute_similarity_matrix(atc_codes, atc_codes)

def pad_atc_codes(atc_codes_list):
    """Pad ATC codes to have the same shape"""
    # Find maximum length
    max_len = max(code.shape[0] for code in atc_codes_list)
    
    # Pad each code to max length
    padded_codes = []
    for code in atc_codes_list:
        if len(code.shape) == 1:
            # Add feature dimension if needed
            code = code[None, :]
        pad_width = ((0, max_len - code.shape[0]), (0, 0))
        padded = np.pad(code, pad_width, mode='constant', constant_values=0)
        padded_codes.append(padded)
    
    return jnp.array(np.stack(padded_codes))

def validate_atc_codes(atc_codes_list):
    """Validate and print shapes of ATC codes"""
    shapes = [code.shape for code in atc_codes_list]
    unique_shapes = set(shapes)
    logging.info(f"Found {len(unique_shapes)} different shapes in ATC codes:")
    for shape in unique_shapes:
        count = sum(1 for s in shapes if s == shape)
        logging.info(f"Shape {shape}: {count} codes")
    
    if len(unique_shapes) > 1:
        raise ValueError(f"ATC codes have inconsistent shapes: {unique_shapes}")
    return shapes[0] if shapes else None

def convert_condition_to_array(condition, char2int):
    """Convert a condition's ATC codes to integer arrays"""
    drug_arrays = []
    for drug in condition.drugs:
        for atc in drug.atcs:
            if atc is not None:
                # Convert each level to integer
                drug_arrays.append(jnp.array([char2int[str(level)] for level in atc]))
    return drug_arrays

def compute_condition_similarity(cond1, cond2, random_kmers, char2int):
    """Compute similarity between two conditions using string kernel"""
    if not (cond1.drugs and cond2.drugs):
        return 0.0
    
    # Convert conditions to integer arrays
    arrays1 = convert_condition_to_array(cond1, char2int)
    arrays2 = convert_condition_to_array(cond2, char2int)
    
    if not (arrays1 and arrays2):
        return 0.0
    
    return compute_similarity_multisentence(arrays1, arrays2, random_kmers)

def process_condition_chunk(args):
    """Process a chunk of condition pairs with progress bar"""
    start_idx, end_idx, condition_list, random_kmers, char2int = args
    chunk_size = end_idx - start_idx
    chunk_matrix = np.zeros((chunk_size, len(condition_list)))
    
    # Create progress bar for this chunk
    pbar_desc = f"Chunk {start_idx}-{end_idx}"
    for i, idx1 in enumerate(tqdm(range(start_idx, end_idx), 
                                desc=pbar_desc, 
                                leave=False)):
        cond1 = condition_list[idx1]
        for idx2 in range(len(condition_list)):
            cond2 = condition_list[idx2]
            sim = compute_condition_similarity(cond1, cond2, random_kmers, char2int)
            chunk_matrix[i, idx2] = sim
            
    return chunk_matrix

def setup_argument_parser():
    """Set up command line argument parsing"""
    parser = argparse.ArgumentParser(description='Precompute condition kernel matrix')
    parser.add_argument('--num-cores', type=int, default=cpu_count(),
                      help='Number of CPU cores to use (default: all available)')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Batch size for processing (default: 100)')
    parser.add_argument('--num-features', type=int, default=100,
                       help='Number of random features for string kernel')
    parser.add_argument('--kmer-length', type=int, default=3,
                       help='Length of k-mers for string kernel')
    return parser

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging()
    start_time = time.time()
    
    try:
        # Initialize random features
        key = random.PRNGKey(0)
        random_kmers = sample_random_kmers(key, args.num_features, args.kmer_length)
        
        # Load data
        _, _, condition_list = load_data()
        n_conditions = len(condition_list)
        logging.info(f"Loaded {n_conditions} conditions")
        
        # Create character to integer mapping
        atc_chars = set()
        for condition in condition_list:
            for drug in condition.drugs:
                for atc in drug.atcs:
                    if atc is not None:
                        for level in atc:
                            atc_chars.add(str(level))
        
        alphabet = sorted(list(atc_chars))
        char2int = {c: i for i, c in enumerate(alphabet)}
        
        # Compute chunk size based on number of cores
        n_cores = min(args.num_cores, cpu_count())
        chunk_size = max(1, n_conditions // n_cores)
        chunks = [(i, min(i + chunk_size, n_conditions), condition_list, random_kmers, char2int) 
                 for i in range(0, n_conditions, chunk_size)]
        
        logging.info(f"Processing on {n_cores} cores with chunk size {chunk_size}")
        
        # Process chunks in parallel with overall progress bar
        total_pairs = n_conditions * n_conditions
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = list(executor.submit(process_condition_chunk, chunk) 
                         for chunk in chunks)
            
            results = []
            for future in tqdm(futures, 
                             desc="Processing chunks", 
                             total=len(futures)):
                results.append(future.result())
        
        # Combine results
        kernel_matrix = np.vstack(results)
        
        # Make symmetric
        kernel_matrix = np.maximum(kernel_matrix, kernel_matrix.T)
        
        # Save the precomputed kernel
        output_file = "Data/condition_kernel_matrix.npz"
        np.savez(output_file, 
                 kernel_matrix=kernel_matrix,
                 condition_list=[c.name for c in condition_list])
        
        logging.info(f"Kernel matrix saved to {output_file}")
        logging.info(f"Matrix shape: {kernel_matrix.shape}")
        logging.info(f"Total computation time: {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()