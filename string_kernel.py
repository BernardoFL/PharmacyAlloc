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


@jit
def string_to_int_array(s, max_length=None):
    """Convert string to integer array with JIT compilation"""
    arr = jnp.array([char2int[c] for c in s if c in char2int])
    if max_length is not None:
        pad_length = max_length - arr.shape[0]
        if pad_length > 0:
            arr = jnp.pad(arr, (0, pad_length), constant_values=-1)
    return arr

@partial(jit, static_argnums=(1,))
def extract_kmers(arr, k):
    """Extract k-mers with static k parameter"""
    return jnp.stack([arr[i:i + k] for i in range(arr.shape[0] - k + 1)])

@partial(jit, static_argnums=(1,))
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

@partial(jit, static_argnums=(1,2))
def sample_random_kmers(key, D, k):
    """Sample random k-mers with static D and k parameters"""
    return jax.random.randint(key, (D, k), 0, len(alphabet))

@jit
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

@jit
def compute_similarity_multisentence(arrays1, arrays2, random_kmers):
    """Compute similarity between two sets of integer arrays"""
    phi1 = compute_random_features_multiarray(arrays1, random_kmers)
    phi2 = compute_random_features_multiarray(arrays2, random_kmers)
    return jnp.dot(phi1, phi2)

@jit
def compute_random_features_multiarray(arrays, random_kmers):
    """Compute features for multiple integer arrays"""
    k = random_kmers.shape[1]
    feature_sum = jnp.zeros(random_kmers.shape[0])
    total_windows = 0
    
    for arr in arrays:
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

