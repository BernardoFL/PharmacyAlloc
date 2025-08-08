import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import combinations
from functools import partial
from scipy.spatial.distance import pdist, squareform
from dataloader import load_data
import sys
sys.setrecursionlimit(100000) #don't like this but it's necessary for the dendrogram function

def similarity_to_distance(similarity_matrix):
    

    max_sim = similarity_matrix.max()
    distance_matrix = max_sim - similarity_matrix

    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix


def compute_patient_knn_distances(X_cov_first_visit, k=10):
    """
    Precompute patient KNN distance matrix using Tanimoto distance.
    
    Parameters
    ----------
    X_cov_first_visit : numpy.ndarray
        Covariate matrix for first visit data
    k : int
        Number of nearest neighbors to consider (default: 10)
        
    Returns
    -------
    numpy.ndarray
        KNN distance matrix
    """
    print(f"Computing patient KNN distance matrix for k={k}...")
    
    def tanimoto_distance(X):
        """Compute Tanimoto distance matrix for binary data (vectorized)."""
        X_binary = (X > 0).astype(float)
        # Intersection: dot product
        intersection = np.dot(X_binary, X_binary.T)
        # Union: |A| + |B| - |A & B|
        row_sums = X_binary.sum(axis=1)
        union = row_sums[:, None] + row_sums[None, :] - intersection
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            tanimoto_sim = np.where(union == 0, 1.0, intersection / union)
        tanimoto_dist = 1.0 - tanimoto_sim
        return tanimoto_dist
    
    # Compute base Tanimoto distance matrix
    base_distances = tanimoto_distance(X_cov_first_visit)
    N = base_distances.shape[0]
    
    # Adjust k if it's larger than N-1
    k_adj = min(k, N-1)
    
    # Initialize KNN distance matrix
    knn_distances = base_distances.copy()
    
    # For each patient, find k nearest neighbors and adjust distances
    for i in range(N):
        # Get distances from patient i to all others
        distances_from_i = base_distances[i, :]
        
        # Find k nearest neighbors (excluding self)
        nearest_indices = np.argsort(distances_from_i)[1:k_adj+1]
        
        # For patients that are not in the k-nearest neighbors, set distance to zero
        for j in range(N):
            if j != i and j not in nearest_indices:
                # Increase distance for non-neighbors
                knn_distances[i, j] = 0.0
    
    # Ensure symmetry
    knn_distances = np.maximum(knn_distances, knn_distances.T)
    
    # Add small diagonal term to ensure positive definiteness
    np.fill_diagonal(knn_distances, 0.0)
    
    # Add small jitter to ensure positive definiteness for GP kernel
    jitter = 1e-6
    knn_distances = knn_distances + jitter * np.eye(N)
    
    return knn_distances


def compute_condition_knn_distances(condition_kernel_matrix, k=10):
    """
    Precompute condition KNN distance matrix using condition kernel similarity.
    
    Parameters
    ----------
    condition_kernel_matrix : numpy.ndarray
        Condition kernel similarity matrix
    k : int
        Number of nearest neighbors to consider (default: 10)
        
    Returns
    -------
    numpy.ndarray
        KNN distance matrix
    """
    print(f"Computing condition KNN distance matrix for k={k}...")
    
    # Convert similarity to distance
    condition_distances = similarity_to_distance(condition_kernel_matrix)
    C = condition_distances.shape[0]
    
    # Adjust k if it's larger than C-1
    k_adj = min(k, C-1)
    
    # Initialize KNN distance matrix
    knn_distances = condition_distances.copy()
    
    # For each condition, find k nearest neighbors and adjust distances
    for i in range(C):
        # Get distances from condition i to all others
        distances_from_i = condition_distances[i, :]
        
        # Find k nearest neighbors (excluding self)
        nearest_indices = np.argsort(distances_from_i)[1:k_adj+1]
        
        # For conditions that are not in the k-nearest neighbors, increase distance
        for j in range(C):
            if j != i and j not in nearest_indices:
                # Increase distance for non-neighbors
                knn_distances[i, j] = condition_distances[i, j] * 1.5
    
    # Ensure symmetry
    knn_distances = np.maximum(knn_distances, knn_distances.T)
    
    # Add small diagonal term to ensure positive definiteness
    np.fill_diagonal(knn_distances, 0.0)
    
    # Add small jitter to ensure positive definiteness for GP kernel
    jitter = 1e-6
    knn_distances = knn_distances + jitter * np.eye(C)
    
    return knn_distances




if __name__ == "__main__":
    import time
    start_time = time.time()
    
    print("Starting presort computation...")
    
    # Load data
    print("Loading data...")
    A, X_cov, condition_list = load_data()  # Use smaller batch for testing
    A_first_visit = A[:,:,0]
    X_cov_first_visit = X_cov[:,:,0]
    
    # Load condition kernel matrix
    condition_kernel_matrix = np.load("Data/condition_kernel_matrix.npz")
    condition_kernel_matrix = condition_kernel_matrix['kernel_matrix']
    
    # Compute KNN distance matrices
    patient_knn_matrix = compute_patient_knn_distances(X_cov_first_visit, k=10)
    condition_knn_matrix = compute_condition_knn_distances(condition_kernel_matrix, k=10)
    
    # Save KNN distance matrices
    print("Saving results...")
    np.save("Data/patient_knn_distances.npy", patient_knn_matrix)
    np.save("Data/condition_knn_distances.npy", condition_knn_matrix)
    
    # Print shapes using numpy array methods
    print(f"Patient KNN matrix shape: {patient_knn_matrix.shape}")
    print(f"Condition KNN matrix shape: {condition_knn_matrix.shape}")
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")