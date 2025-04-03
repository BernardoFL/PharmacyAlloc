#!/usr/bin/env python3

import os
# Configure JAX to use CPU and disable GPU/TPU warnings
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import logging
import jax
jax.config.update('jax_default_device', jax.devices('cpu')[0])
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial  # Change this import
import numpy as np
import optax
from dataloader import load_data
import GPVarInf as GPVI
from datetime import datetime
import time

def setup_logging(log_dir='logs'):
    """Set up logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'model_run_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def run_model():
    # Add profiling
    start_time = time.time()
    
    # Load data (only once)
    logging.info("Loading data...")
    A, X_cov, condition_list = load_data()
    logging.info(f"Data loading took {time.time() - start_time:.2f}s")
    
    # Convert to JAX arrays once
    A = jnp.asarray(A)
    X_cov = jnp.asarray(X_cov)
    
    # Load and convert indices once
    patient_order = jnp.asarray(np.load("Data/patient_order.npy"))
    condition_order = jnp.asarray(np.load("Data/condition_order.npy"))
    
    # Use JAX's efficient indexing
    A_first_visit = A.at[:,:,0].get()
    A_sorted = A_first_visit.at[patient_order].get().at[:,condition_order].get()
    X_cov_sorted = X_cov.at[patient_order].get()
    
    # Print debug info
    logging.info(f"Matrix shapes after sorting:")
    logging.info(f"A: {A_sorted.shape}")
    logging.info(f"X_cov: {X_cov_sorted.shape}")
    logging.info(f"Total processing time: {time.time() - start_time:.2f}s")
    
    return A_sorted, X_cov_sorted, condition_list

@partial(jit, static_argnums=(1, 2))
def train_step(params, model, optimizer, opt_state, batch):
    """Single training step"""
    x_batch, y_batch = batch
    
    def loss_fn(params):
        elbo = model.elbo(params, x_batch, y_batch)
        return -elbo  # Minimize negative ELBO
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, -loss  # Return positive ELBO

@jit
def elbo(self, params, X: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute ELBO with given parameters"""
    # Extract parameters
    q_mu = params['q_mu']
    q_sqrt = params['q_sqrt']
    log_lengthscale = params['log_lengthscale']
    log_variance = params['log_variance']
    log_scale = params['log_scale']
    
    # Compute kernel matrices
    K_mm = self.compute_Kmm(log_lengthscale, log_variance, log_scale)
    K_nm = self.compute_Knm(X, log_lengthscale, log_variance, log_scale)
    
    # Compute variational distribution
    q_dist = tfd.MultivariateNormalTriL(
        loc=q_mu.flatten(),
        scale_tril=q_sqrt
    )
    
    # Compute KL divergence
    p_dist = tfd.MultivariateNormalTriL(
        loc=jnp.zeros(self.num_inducing),
        scale_tril=jnp.eye(self.num_inducing)
    )
    kl = tfd.kl_divergence(q_dist, p_dist)
    
    # Compute log likelihood
    f_mean = K_nm @ jnp.linalg.solve(K_mm, q_mu)
    log_lik = -0.5 * jnp.sum((y - f_mean.flatten()) ** 2)
    
    return log_lik - kl

@jit
def compute_Kmm(self, log_lengthscale, log_variance, log_scale):
    """Compute kernel matrix between inducing points"""
    lengthscale = jnp.exp(log_lengthscale)
    variance = jnp.exp(log_variance)
    scale = jnp.exp(log_scale)
    
    K_mm = self.compute_kernel_matrix(self.inducing_points, self.inducing_points)
    K_mm = variance * K_mm
    K_mm += jnp.eye(self.num_inducing) * 1e-6  # Add jitter for stability
    return scale * K_mm

@jit
def compute_Knm(self, X, log_lengthscale, log_variance, log_scale):
    """Compute kernel matrix between data points and inducing points"""
    lengthscale = jnp.exp(log_lengthscale)
    variance = jnp.exp(log_variance)
    scale = jnp.exp(log_scale)
    
    K_nm = self.compute_kernel_matrix(X, self.inducing_points)
    return variance * K_nm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the KNN-GP model')
    parser.add_argument('--num-inducing', type=int, default=50,
                      help='Number of inducing points')
    parser.add_argument('--k-neighbors', type=int, default=5,
                      help='Number of nearest neighbors for KNN-GP')
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='Learning rate for optimization')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Batch size for training')
    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting model run with arguments: {args}")

    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        A_sorted, X_cov_sorted, condition_list = run_model()
        
        # First let's work with time=1
        y = A_sorted.T  # Shape: (n_conditions, n_patients)
        
        # Create input features
        n_conditions = len(condition_list)
        n_patients = y.shape[1]
        
        # Create condition indices grid
        condition_indices = jnp.repeat(jnp.arange(n_conditions), n_patients)
        
        # Create 2D spatial coordinates grid
        x_coords = jnp.repeat(jnp.arange(n_conditions), n_patients)
        y_coords = jnp.tile(jnp.arange(n_patients), n_conditions)
        spatial_coords = jnp.stack([x_coords, y_coords], axis=1)
        
        # Create continuous features
        continuous_features = jnp.tile(X_cov_sorted[:, :2], (n_conditions, 1))
        
        # Combine into input tensor x
        x = jnp.concatenate([
            condition_indices[:, None],
            spatial_coords,
            continuous_features
        ], axis=1)
        y = y.reshape(-1)  # Flatten y
        
        # Create inducing points
        key = random.PRNGKey(0)
        inducing_indices = random.permutation(key, x.shape[0])[:args.num_inducing]
        inducing_points = x[inducing_indices]
        
        # Create model and optimizer
        logging.info("Creating KNN-GP model...")
        model = GPVI.VariationalGP(inducing_points, condition_list, args.num_inducing)
        optimizer = optax.adam(args.learning_rate)
        opt_state = optimizer.init(model.get_params())
        
        # Create batches
        num_batches = len(x) // args.batch_size
        if num_batches == 0:
            logging.warning("Batch size is larger than the dataset size. Reducing batch size.")
            args.batch_size = len(x)
            num_batches = 1

        # Training loop
        logging.info("Starting training...")
        for epoch in range(args.num_epochs):
            # Shuffle data
            key = random.fold_in(key, epoch)
            perm = random.permutation(key, len(x))
            x_shuffled = x[perm]
            y_shuffled = y[perm]
            
            total_elbo = 0.0
            for i in range(num_batches):
                batch_idx = slice(i * args.batch_size, (i + 1) * args.batch_size)
                batch = (x_shuffled[batch_idx], y_shuffled[batch_idx])
                
                params, opt_state, elbo = train_step(
                    model.get_params(), 
                    model, 
                    optimizer, 
                    opt_state, 
                    batch
                )
                model.set_params(params)
                total_elbo += elbo
                
            avg_elbo = total_elbo / num_batches
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}/{args.num_epochs}, Average ELBO: {avg_elbo:.4f}")
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'model_{timestamp}.npz'
        np.savez(model_path, 
                 params=model.get_params(),
                 inducing_points=inducing_points,
                 condition_list=condition_list)
        logging.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()