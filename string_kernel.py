import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from dataloader import load_data

# Load data
_, _, condition_list = load_data()

# Collect all unique ATC code characters at each level
atc_chars = set()
for condition in condition_list:
    for drug in condition.drugs:
        for atc in drug.atcs:
            # Assuming atc is a list/array of 4 levels
            for level in atc:
                atc_chars.add(str(level))

# Convert to sorted list for consistent mapping
alphabet = sorted(list(atc_chars))
char2int = {c: i for i, c in enumerate(alphabet)}



def string_to_int_array(s, max_length=None):
    """Convert string to integer array with JIT compilation"""
    arr = jnp.array([char2int[c] for c in s if c in char2int])
    if max_length is not None:
        pad_length = max_length - arr.shape[0]
        if pad_length > 0:
            arr = jnp.pad(arr, (0, pad_length), constant_values=-1)
    return arr


def extract_kmers(arr, k):
    """Extract k-mers with static k parameter"""
    return jnp.stack([arr[i:i + k] for i in range(arr.shape[0] - k + 1)])


def random_feature_vector(arr, random_kmers):
    """Compute random feature vector with JIT"""
    k = random_kmers.shape[1]
    kmers = extract_kmers(arr, k)
    
    kmers_exp = jnp.expand_dims(kmers, axis=0)
    random_kmers_exp = jnp.expand_dims(random_kmers, axis=1)
    
    eq = (kmers_exp == random_kmers_exp)
    match = jnp.all(eq, axis=-1)
    counts = jnp.sum(match, axis=-1)
    
    norm = kmers.shape[0]
    return jnp.sqrt(counts.astype(jnp.float32) / norm)


def sample_random_kmers(key, D, k):
    """Sample random k-mers with static D and k parameters"""
    return jax.random.randint(key, (D, k), 0, len(alphabet))


def compute_random_features_for_multisentence_string(s, random_kmers, delimiter="."):
    """Compute features for multi-sentence string"""
    sentences = s.split(delimiter)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    k = random_kmers.shape[1]  # Get k dynamically from random_kmers shape
    feature_sum = jnp.zeros(random_kmers.shape[0])
    total_windows = 0

    for sentence in sentences:
        arr = string_to_int_array(sentence)
        if arr.shape[0] < k:
            continue
        kmers = extract_kmers(arr, k)
        kmers_exp = jnp.expand_dims(kmers, axis=0)
        random_kmers_exp = jnp.expand_dims(random_kmers, axis=1)
        eq = (kmers_exp == random_kmers_exp)
        match = jnp.all(eq, axis=-1)
        counts = jnp.sum(match, axis=-1)
        
        feature_sum += counts
        total_windows += kmers.shape[0]
    
    return jnp.where(total_windows > 0,
                    jnp.sqrt(feature_sum.astype(jnp.float32) / total_windows),
                    jnp.zeros(random_kmers.shape[0]))


def compute_similarity_multisentence(arrays1, arrays2, random_kmers):
    """Compute similarity between two sets of integer arrays"""
    phi1 = compute_random_features_multiarray(arrays1, random_kmers)
    phi2 = compute_random_features_multiarray(arrays2, random_kmers)
    return jnp.dot(phi1, phi2)


def compute_random_features_multiarray(arrays, random_kmers):
    """
    Compute normalized random feature vectors for multiple integer arrays with detailed logging.
    """
    k = random_kmers.shape[1]
    D = random_kmers.shape[0]
    
    # Pre-expand random_kmers once outside the loop
    random_kmers_exp = jnp.expand_dims(random_kmers, axis=1)
    feature_sum = jnp.zeros(D)
    total_windows = 0
    
    print(f"Arrays lengths: {[arr.shape[0] for arr in arrays]}")
    print(f"k-mer length: {k}")
    print(f"Random k-mers range: {jnp.min(random_kmers)} to {jnp.max(random_kmers)}")
    print(f"Alphabet size: {len(alphabet)}")
    
    # Convert random k-mers to readable format for debugging
    random_kmers_readable = []
    for i in range(min(5, random_kmers.shape[0])):
        chars = [alphabet[idx] for idx in random_kmers[i]]
        random_kmers_readable.append(''.join(chars))
    print(f"Sample random k-mers (readable): {random_kmers_readable}")
    
    match_found = False
    for arr_idx, arr in enumerate(arrays):
        print(f"\nProcessing array {arr_idx}, length: {arr.shape[0]}")
        
        # Skip short arrays immediately
        if arr.shape[0] < k:
            print(f"  Skipping array {arr_idx} (too short)")
            continue
            
        kmers = extract_kmers(arr, k)
        print(f"  Extracted {kmers.shape[0]} k-mers")
        
        # Convert some k-mers to readable format for debugging
        kmers_readable = []
        for i in range(min(5, kmers.shape[0])):
            chars = [alphabet[idx] for idx in kmers[i]]
            kmers_readable.append(''.join(chars))
        print(f"  Sample k-mers (readable): {kmers_readable}")
        
        # Compare with each random k-mer
        print("  Comparing k-mers...")
        for i in range(min(5, kmers.shape[0])):
            kmer = kmers[i]
            kmer_str = ''.join([alphabet[idx] for idx in kmer])
            print(f"    Array k-mer {i}: {kmer} ({kmer_str})")
            
            for j in range(min(5, random_kmers.shape[0])):
                random_kmer = random_kmers[j]
                random_kmer_str = ''.join([alphabet[idx] for idx in random_kmer])
                
                if jnp.all(kmer == random_kmer):
                    print(f"      MATCH with random k-mer {j}: {random_kmer} ({random_kmer_str})")
                    match_found = True
                else:
                    print(f"      No match with random k-mer {j}: {random_kmer} ({random_kmer_str})")
        
        # Regular processing continues
        kmers_exp = jnp.expand_dims(kmers, axis=0)
        match = jnp.all(kmers_exp == random_kmers_exp, axis=-1)
        counts = jnp.sum(match, axis=-1)
        
        print(f"  Total matches found: {jnp.sum(counts)}")
        feature_sum += counts
        total_windows += kmers.shape[0]
    
    if not match_found:
        print("\n!!! WARNING: NO MATCHES FOUND BETWEEN ANY K-MERS !!!")
        print("This explains why you're getting all zeros.")
        print("Consider one of these solutions:")
        print("1. Increase the number of random features (D)")
        print("2. Decrease the k-mer length (k)")
        print("3. Sample random k-mers from your actual data")
    
    # More efficient normalization
    result = jnp.where(total_windows > 0,
                      jnp.sqrt(feature_sum.astype(jnp.float32) / total_windows),
                      jnp.zeros(D))
    
    print(f"\nTotal windows: {total_windows}")
    print(f"Feature sum: {feature_sum[:5]}...")
    print(f"Result: {result[:5]}...")
    print(f"Non-zero elements: {jnp.sum(result > 0)} out of {D}")
    
    return result

