import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from dataloader import load_data
import time

def similarity_to_distance(similarity_matrix):
    """Convert similarity matrix to distance matrix."""
    max_sim = similarity_matrix.max()
    distance_matrix = max_sim - similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix

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

def compute_knn_distances(X, n_neighbors=10, metric='precomputed'):
    """
    Compute KNN distance matrix using sklearn NearestNeighbors.
    
    Args:
        X: Distance matrix (n_samples, n_samples) or feature matrix
        n_neighbors: Number of neighbors to consider
        metric: Distance metric ('precomputed' for distance matrix, 'euclidean' for features)
    
    Returns:
        distances: KNN distances (n_samples, n_neighbors)
        indices: KNN indices (n_samples, n_neighbors)
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='auto')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances, indices

def precompute_knn_matrices():
    """
    Precompute KNN distance matrices for both patients and conditions.
    Uses the same metrics as presort.py:
    - Conditions: condition kernel matrix converted to distance
    - Patients: Tanimoto distance on first visit covariate matrix
    """
    print("Loading data...")
    A, X_cov, condition_list = load_data()
    
    # Filter to first visit only
    A_first_visit = A[:,:,0]
    X_cov_first_visit = X_cov[:,:,0]
    
    print(f"Number of unique conditions: {len(condition_list)}")
    print(f"Number of patients (first visit only): {A_first_visit.shape[0]}")
    
    # Load condition kernel matrix and convert to distance
    print("Loading condition kernel matrix...")
    condition_kernel_matrix = np.load("Data/condition_kernel_matrix.npz")
    condition_kernel_matrix = condition_kernel_matrix['kernel_matrix']
    condition_distance_matrix = similarity_to_distance(condition_kernel_matrix)
    
    # Compute patient distances using Tanimoto distance
    print("Computing patient Tanimoto distances...")
    patient_distance_matrix = tanimoto_distance(X_cov_first_visit)
    
    # Compute KNN for conditions
    print("Computing KNN for conditions...")
    n_conditions = len(condition_list)
    n_neighbors_conditions = min(10, n_conditions - 1)  # Ensure we don't exceed n_samples-1
    
    condition_distances, condition_indices = compute_knn_distances(
        condition_distance_matrix, 
        n_neighbors=n_neighbors_conditions, 
        metric='precomputed'
    )
    
    # Compute KNN for patients
    print("Computing KNN for patients...")
    n_patients = A_first_visit.shape[0]
    n_neighbors_patients = min(10, n_patients - 1)  # Ensure we don't exceed n_samples-1
    
    patient_distances, patient_indices = compute_knn_distances(
        patient_distance_matrix, 
        n_neighbors=n_neighbors_patients, 
        metric='precomputed'
    )
    
    return {
        'condition_distances': condition_distances,
        'condition_indices': condition_indices,
        'patient_distances': patient_distances,
        'patient_indices': patient_indices,
        'condition_distance_matrix': condition_distance_matrix,
        'patient_distance_matrix': patient_distance_matrix,
        'n_neighbors_conditions': n_neighbors_conditions,
        'n_neighbors_patients': n_neighbors_patients
    }

if __name__ == "__main__":
    start_time = time.time()
    
    print("Starting KNN distance matrix computation...")
    
    # Compute KNN matrices
    results = precompute_knn_matrices()
    
    # Save results
    print("Saving KNN distance matrices...")
    np.savez_compressed(
        "Data/condition_knn_distances.npz",
        distances=results['condition_distances'],
        indices=results['condition_indices'],
        full_distance_matrix=results['condition_distance_matrix'],
        n_neighbors=results['n_neighbors_conditions']
    )
    
    np.savez_compressed(
        "Data/patient_knn_distances.npz",
        distances=results['patient_distances'],
        indices=results['patient_indices'],
        full_distance_matrix=results['patient_distance_matrix'],
        n_neighbors=results['n_neighbors_patients']
    )
    
    # Print summary
    print(f"Condition KNN distances shape: {results['condition_distances'].shape}")
    print(f"Condition KNN indices shape: {results['condition_indices'].shape}")
    print(f"Patient KNN distances shape: {results['patient_distances'].shape}")
    print(f"Patient KNN indices shape: {results['patient_indices'].shape}")
    print(f"Number of neighbors for conditions: {results['n_neighbors_conditions']}")
    print(f"Number of neighbors for patients: {results['n_neighbors_patients']}")
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    
    print("KNN distance matrices saved to:")
    print("- Data/condition_knn_distances.npz")
    print("- Data/patient_knn_distances.npz")
