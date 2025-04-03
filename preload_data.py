import os
# Configure JAX to use CPU and disable GPU/TPU warnings
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import jax.numpy as jnp
import numpy as np
from dataloader import load_data
import logging
from datetime import datetime
import time

def setup_logging(log_dir='logs'):
    """Set up logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'data_preload_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def main():
    # Set up logging
    log_file = setup_logging()
    start_time = time.time()
    logging.info("Starting data preloading...")
    
    try:
        # Create data directory if it doesn't exist
        if not os.path.exists('Data/cached'):
            os.makedirs('Data/cached')
        
        # Load data
        logging.info("Loading data from source files...")
        A, X_cov, condition_list = load_data()
        
        # Save matrices
        logging.info("Saving matrices to cache...")
        np.savez('Data/cached/preprocessed_data.npz',
                 A=np.array(A),
                 X_cov=np.array(X_cov))
        
        # Save condition list separately (since it's not a numeric array)
        import pickle
        with open('Data/cached/condition_list.pkl', 'wb') as f:
            pickle.dump(condition_list, f)
        
        logging.info(f"Data shapes:")
        logging.info(f"A: {A.shape}")
        logging.info(f"X_cov: {X_cov.shape}")
        logging.info(f"Number of conditions: {len(condition_list)}")
        logging.info(f"Total processing time: {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()