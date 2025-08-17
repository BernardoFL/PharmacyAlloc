import numpy as np

# Caches for lazy loading
_patient_knn_cache = None
_condition_knn_cache = None

def load_condition_knn(lazy_load=False):
    """
    Load precomputed KNN distance matrix for conditions.
    
    Args:
        lazy_load (bool): If True, use cached data if available.
    
    Returns:
        dict: Dictionary containing KNN data.
    """
    global _condition_knn_cache
    if lazy_load and _condition_knn_cache is not None:
        return _condition_knn_cache
        
    data = np.load("Data/condition_knn_distances.npz")
    knn_data = {
        'distances': data['distances'],
        'indices': data['indices'],
        'full_distance_matrix': data['full_distance_matrix'],
        'n_neighbors': data['n_neighbors']
    }
    
    if lazy_load:
        _condition_knn_cache = knn_data
        
    return knn_data

def load_patient_knn(lazy_load=False):
    """
    Load precomputed KNN distance matrix for patients.
    
    Args:
        lazy_load (bool): If True, use cached data if available.
    
    Returns:
        dict: Dictionary containing KNN data.
    """
    global _patient_knn_cache
    if lazy_load and _patient_knn_cache is not None:
        return _patient_knn_cache
        
    data = np.load("Data/patient_knn_distances.npz")
    knn_data = {
        'distances': data['distances'],
        'indices': data['indices'],
        'full_distance_matrix': data['full_distance_matrix'],
        'n_neighbors': data['n_neighbors']
    }
    
    if lazy_load:
        _patient_knn_cache = knn_data
        
    return knn_data

def get_knn_neighbors(knn_data, item_idx, k=None):
    """
    Get K nearest neighbors for a specific item.
    
    Args:
        knn_data: Dictionary from load_condition_knn() or load_patient_knn()
        item_idx: Index of the item to get neighbors for
        k: Number of neighbors to return (if None, returns all available)
    
    Returns:
        tuple: (distances, indices) for the k nearest neighbors
    """
    if k is None:
        k = knn_data['n_neighbors']
    else:
        k = min(k, knn_data['n_neighbors'])
    
    distances = knn_data['distances'][item_idx, :k]
    indices = knn_data['indices'][item_idx, :k]
    
    return distances, indices

def get_knn_summary():
    """
    Get a summary of the precomputed KNN matrices.
    
    Returns:
        dict: Summary information about both KNN matrices
    """
    try:
        condition_knn = load_condition_knn()
        patient_knn = load_patient_knn()
        
        return {
            'conditions': {
                'shape': condition_knn['distances'].shape,
                'n_neighbors': condition_knn['n_neighbors'],
                'total_items': condition_knn['distances'].shape[0]
            },
            'patients': {
                'shape': patient_knn['distances'].shape,
                'n_neighbors': patient_knn['n_neighbors'],
                'total_items': patient_knn['distances'].shape[0]
            }
        }
    except FileNotFoundError as e:
        print(f"KNN files not found: {e}")
        return None
