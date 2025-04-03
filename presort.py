import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import combinations
from functools import partial
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from dataloader import load_data
from GPVarInf import condition_similarity_vectorized

def compute_similarity_chunk(args):
    """Helper function to compute similarities for a chunk of condition pairs"""
    condition_pairs, condition_list = args
    results = []
    for i, j in condition_pairs:
        sim = condition_similarity_vectorized(condition_list[i], condition_list[j])
        results.append((i, j, sim.item()))
    return results

def compute_condition_distance_matrix(condition_list):
    """
    Parallel computation of condition distances based on drug ATC codes.
    """
    n_conditions = len(condition_list)
    similarity_matrix = torch.zeros((n_conditions, n_conditions))
    
    # Generate all pairs of indices
    pairs = list(combinations(range(n_conditions), 2))
    
    # Split pairs into chunks for parallel processing
    n_cores = mp.cpu_count()
    chunk_size = max(1, len(pairs) // (n_cores * 4))
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    
    # Prepare arguments for parallel processing
    args = [(chunk, condition_list) for chunk in chunks]
    
    # Compute similarities in parallel
    with mp.Pool(n_cores) as pool:
        results = pool.map(compute_similarity_chunk, args)
    
    # Combine results
    for chunk_results in results:
        for i, j, sim in chunk_results:
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    # Set diagonal to maximum similarity (self-similarity)
    torch.diagonal(similarity_matrix).fill_(similarity_matrix.max())
    
    # Convert similarities to distances and ensure diagonal is 0
    max_sim = similarity_matrix.max()
    distance_matrix = max_sim - similarity_matrix
    torch.diagonal(distance_matrix).fill_(0.0)
    
    return distance_matrix

def get_ordering_indices():
    """
    Get sorting indices for conditions and patients based on similarity.
    Returns numpy arrays of indices using only first visit data.
    """
    # Load data using the existing load_data function
    A, X_cov, condition_list = load_data()
    
    # Filter matrix A to only include first visits
    # Assuming A is organized as [patients x conditions]
    patients_per_visit = len(condition_list)
    A_first_visit = A[:patients_per_visit]
    X_cov_first_visit = X_cov[:patients_per_visit]
    
    print(f"Number of unique conditions: {len(condition_list)}")
    print(f"Number of patients (first visit only): {A_first_visit.shape[0]}")
    
    # Compute condition distances and perform hierarchical clustering
    condition_dist_matrix = compute_condition_distance_matrix(condition_list)
    condition_linkage = linkage(squareform(condition_dist_matrix), method='ward')
    condition_order = np.array(dendrogram(condition_linkage, no_plot=True)['leaves'])
    
    # Compute patient distances using first visit covariate matrix
    patient_distances = pdist(X_cov_first_visit.numpy(), metric='euclidean')
    patient_linkage = linkage(patient_distances, method='ward')
    patient_order = np.array(dendrogram(patient_linkage, no_plot=True)['leaves'])
    
    return patient_order, condition_order

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    print("Starting presort computation...")
    
    # Get ordering indices
    print("Computing ordering indices...")
    patient_order, condition_order = get_ordering_indices()
    
    # Save the orderings
    print("Saving results...")
    np.save("Data/patient_order.npy", patient_order)
    np.save("Data/condition_order.npy", condition_order)
    
    # Print shapes using numpy array methods
    print(f"Patient ordering shape: {patient_order.shape}")
    print(f"Condition ordering shape: {condition_order.shape}")
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")