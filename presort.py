import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import combinations
from functools import partial
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from dataloader import load_data
import sys
sys.setrecursionlimit(100000) #don't like this but it's necessary for the dendrogram function

def similarity_to_distance(similarity_matrix):
    

    max_sim = similarity_matrix.max()
    distance_matrix = max_sim - similarity_matrix

    np.fill_diagonal(distance_matrix, 0.0)
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
    A_first_visit = A[:,:,0]
    X_cov_first_visit = X_cov[:,:,0]

    condition_kernel_matrix = np.load("Data/condition_kernel_matrix.npz")
    condition_kernel_matrix = condition_kernel_matrix['kernel_matrix']
    condition_kernel_matrix = similarity_to_distance(condition_kernel_matrix)

    print(f"Number of unique conditions: {len(condition_list)}")
    print(f"Number of patients (first visit only): {A_first_visit.shape[0]}")
    
    # Compute condition distances and perform hierarchical clustering
    condition_linkage = linkage(squareform(condition_kernel_matrix), method='ward')
    condition_order = np.array(dendrogram(condition_linkage, no_plot=True)['leaves'])
    
    # Compute patient distances using first visit covariate matrix with Tanimoto distance
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
        # Return condensed distance matrix for pdist compatibility
        return squareform(tanimoto_dist, checks=False)
    
    patient_distances = tanimoto_distance(X_cov_first_visit)
    patient_linkage = linkage(patient_distances, method='average')
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