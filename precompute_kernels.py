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
from string_kernel import extract_kmers

# Add this line at the top of your file with other global variables/imports
alphabet = None

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

# Fix the function signature in compute_condition_similarity()
def compute_condition_similarity(arrays1, arrays2, custom_alphabet=None, m=100, threshold=0.05, batch_size=1000):
    """Faster k-mer comparison using hash sets with relaxed matching and ATC hierarchy"""
    # Access the global alphabet
    global alphabet
    
    # Use custom alphabet if provided
    alphabet_to_use = custom_alphabet if custom_alphabet is not None else alphabet
    
    k = 3  # k-mer length
    
    # Extract all k-mers from both conditions
    kmers1 = []
    kmers2 = []
    
    for arr in arrays1:
        if arr.shape[0] >= k:
            kmers1.append(extract_kmers(arr, k))
    
    for arr in arrays2:
        if arr.shape[0] >= k:
            kmers2.append(extract_kmers(arr, k))
    
    # If either condition has no valid k-mers, similarity is 0
    if not kmers1 or not kmers2:
        return 0.0
    
    # Concatenate all k-mers
    kmers1 = jnp.concatenate(kmers1, axis=0)
    kmers2 = jnp.concatenate(kmers2, axis=0)
    
    # Convert kmers2 to a hash set for O(1) lookups
    # Convert JAX arrays to tuples for hashing
    kmers2_set = {tuple(kmer.tolist()) for kmer in np.array(kmers2)}
    
    # If either condition has fewer than m k-mers, use all of them
    m1 = min(m, kmers1.shape[0])
    
    # Sample k-mers from condition 1
    key = jax.random.PRNGKey(0)
    indices1 = jax.random.choice(key, kmers1.shape[0], shape=(m1,), replace=False)
    sampled_kmers1 = kmers1[indices1]
    
    # For large k-mer sets, process in batches
    matches = 0
    partial_matches = 0
    
    # Convert to NumPy arrays for easier list comprehension
    np_sampled_kmers1 = np.array(sampled_kmers1)
    np_kmers2_list = [tuple(kmer) for kmer in np.array(kmers2)]
    
    for i in range(m1):
        kmer = tuple(np_sampled_kmers1[i])
        
        # Check for exact match
        if kmer in kmers2_set:
            matches += 1
            continue
            
        # Check for partial matches (2 out of 3 positions matching)
        for kmer2 in np_kmers2_list:
            if sum(a == b for a, b in zip(kmer, kmer2)) >= k-1:
                partial_matches += 1
                break
    
    # Calculate similarity with both exact and partial matches
    # Exact matches count as 1, partial as 0.5
    sim1to2 = (matches + 0.5 * partial_matches) / m1 if m1 > 0 else 0
    
    # If similarity is still 0, use ATC hierarchy knowledge
    if sim1to2 == 0:
        # Extract first characters/levels of k-mers for hierarchical matching
        first_chars1 = {kmer[0] for kmer in np_sampled_kmers1}
        first_chars2 = {kmer[0] for kmer in np.array(kmers2)}
        
        # Check for hierarchical overlap
        hierarchy_overlap = len(first_chars1.intersection(first_chars2))
        if hierarchy_overlap > 0:
            # Small non-zero similarity based on first level match
            return 0.1 * (hierarchy_overlap / len(first_chars1))
        
        # Try reversed direction
        kmers1_set = {tuple(kmer.tolist()) for kmer in np.array(kmers1)}
        
        m2 = min(m, kmers2.shape[0])
        # Sample k-mers from condition 2
        key = jax.random.PRNGKey(1)
        indices2 = jax.random.choice(key, kmers2.shape[0], shape=(m2,), replace=False)
        sampled_kmers2 = kmers2[indices2]
        
        matches = 0
        partial_matches = 0
        np_sampled_kmers2 = np.array(sampled_kmers2)
        np_kmers1_list = [tuple(kmer) for kmer in np.array(kmers1)]
        
        for i in range(m2):
            kmer = tuple(np_sampled_kmers2[i])
            
            # Check for exact match
            if kmer in kmers1_set:
                matches += 1
                continue
                
            # Check for partial matches (2 out of 3 positions matching)
            for kmer1 in np_kmers1_list:
                if sum(a == b for a, b in zip(kmer, kmer1)) >= k-1:
                    partial_matches += 1
                    break
        
        sim2to1 = (matches + 0.5 * partial_matches) / m2 if m2 > 0 else 0
        return sim2to1
    
    return sim1to2

def get_alphabet(condition_list):
    """Get alphabet from condition list"""
    atc_chars = set()
    for condition in condition_list:
        for drug in condition.drugs:
            for atc in drug.atcs:
                if atc is not None:
                    for level in atc:
                        atc_chars.add(str(level))
    return sorted(list(atc_chars))

def process_condition_chunk(args):
    """Process a chunk of condition pairs with progress bar"""
    try:
        start_idx, end_idx, condition_list, char2int = args
        chunk_size = end_idx - start_idx
        chunk_matrix = np.zeros((chunk_size, len(condition_list)))
        
        # Create progress bar for this chunk
        pbar_desc = f"Chunk {start_idx}-{end_idx}"
        for i, idx1 in enumerate(tqdm(range(start_idx, end_idx), 
                                    desc=pbar_desc, 
                                    leave=False)):
            cond1 = condition_list[idx1]
            arrays1 = convert_condition_to_array(cond1, char2int)
            for idx2 in range(len(condition_list)):
                cond2 = condition_list[idx2]
                arrays2 = convert_condition_to_array(cond2, char2int)
                sim = compute_condition_similarity(arrays1, arrays2)
                chunk_matrix[i, idx2] = sim
                
        return chunk_matrix
    except Exception as e:
        import traceback
        return f"Error in process_condition_chunk: {str(e)}\n{traceback.format_exc()}"

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
        global alphabet  # Tell Python you're modifying the global variable
        alphabet = get_alphabet(condition_list)
        char2int = {c: i for i, c in enumerate(alphabet)}
        
        # Process without multiprocessing
        logging.info("Processing without multiprocessing (single thread)")
        kernel_matrix = np.zeros((n_conditions, n_conditions))
        
        # Use tqdm for progress tracking
        for idx1 in tqdm(range(n_conditions), desc="Computing similarities"):
            cond1 = condition_list[idx1]
            arrays1 = convert_condition_to_array(cond1, char2int)
            for idx2 in range(n_conditions):
                cond2 = condition_list[idx2]
                arrays2 = convert_condition_to_array(cond2, char2int)
                sim = compute_condition_similarity(arrays1, arrays2, alphabet)
                kernel_matrix[idx1, idx2] = sim
        
        # Make symmetric (just to be safe)
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