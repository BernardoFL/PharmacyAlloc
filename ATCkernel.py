
def precompute_drug_kernel(drug_list):
    """
    Precompute an M x M drug kernel matrix using graphkit-learn's Weisfeiler-Lehman subtree kernel.
    
    Args:
        drug_list: List of Drug objects (each with an attribute 'atcs' that is a list of NetworkX graphs).
        n_iter: Number of iterations for the WL kernel.
        
    Returns:
        A torch.Tensor of shape (M, M) containing the precomputed kernel values.
    """
    # Use the first ATC graph from each drug as its representative.
    graphs = [drug.atcs[0] for drug in drug_list]
    
    # Set up the WL kernel with VertexHistogram as the base kernel.
    wl_kernel = WeisfeilerLehman()
    # Compute the kernel matrix (a numpy array).
    K = wl_kernel.compute(graphs)
    
    # Convert the numpy array to a PyTorch tensor.
    K_tensor = torch.tensor(K, dtype=torch.float32)
    return K_tensor