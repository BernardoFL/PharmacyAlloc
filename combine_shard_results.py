#!/usr/bin/env python

import os
import sys
import argparse
import logging
import json
import pickle
from datetime import datetime
import numpy as np

sys.path.append('./Source')
sys.path.append('./_dependency')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import ot  # Python Optimal Transport library
except Exception:
    ot = None


def setup_logging(log_dir: str = 'logs') -> str:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'combine_results_run_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")
    return log_file


def compute_wasserstein_barycenter(mcmc_samples, weights=None):
    """Compute a Wasserstein (or Euclidean fallback) barycenter across shards.

    Accepts a list of arrays with identical shapes: (num_draws, ...).
    If POT (ot) is unavailable, falls back to simple Euclidean mean across shards.
    """
    n_distributions = len(mcmc_samples)
    if n_distributions == 0:
        raise ValueError("No sample arrays provided to compute barycenter.")

    # Default uniform weights
    if weights is None:
        weights = np.ones(n_distributions, dtype=float) / float(n_distributions)

    # If POT isn't installed, fall back to simple (weighted) Euclidean mean
    if ot is None:
        stacked = np.stack(mcmc_samples, axis=0)  # (S, num_draws, ...)
        # Broadcast weights to leading dimension
        w = np.array(weights, dtype=float).reshape(-1, *([1] * (stacked.ndim - 1)))
        return np.sum(w * stacked, axis=0)

    # POT barycenter in flattened space
    all_samples = []
    all_weights = []
    for samples in mcmc_samples:
        flat_samples = samples.reshape(len(samples), -1)
        sample_weights = np.ones(len(samples)) / len(samples)
        all_samples.append(flat_samples)
        all_weights.append(sample_weights)

    barycenter = ot.lp.free_support_barycenter(all_samples, all_weights, weights)

    original_shape = mcmc_samples[0].shape
    if len(original_shape) > 2:
        barycenter = barycenter.reshape(-1, *original_shape[1:])
    return barycenter


def load_shard_samples(shard_dir: str):
    """Load per-shard MCMC samples and normalize keys for downstream use.

    - Supports legacy keys: 'f' (latent field), hyperparams like 'eta', 'ell', 'sigma_noise'.
    - Supports GMRF keys: 'Lambda' (latent field), 'beta_pat', 'tau', 'lambdas'.
    - If 'Lambda' exists, creates a compatibility alias 'f' pointing to the same array.
    - If metadata is present with shape, will expose a reshaped 'f_reshaped' for legacy consumers.
    """
    samples_file = os.path.join(shard_dir, 'mcmc_samples.npy')
    samples = np.load(samples_file, allow_pickle=True).item()

    # GMRF compatibility: use 'Lambda' as latent and alias to 'f'
    if 'Lambda' in samples and 'f' not in samples:
        samples['f'] = samples['Lambda']

    # Check for metadata and reshape 'f' if needed (legacy)
    if '_metadata' in samples and 'shape' in samples['_metadata'] and 'f' in samples:
        N, C = samples['_metadata']['shape']
        total_draws = samples['f'].shape[0]
        samples['f_reshaped'] = samples['f'].reshape(total_draws, N, C)

    return samples


def _first_available_draw_count(sample_dict):
    """Attempt to infer number of draws from any array-valued entry."""
    for v in sample_dict.values():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            return v.shape[0]
    return 0


def combine_from_dirs(shard_dirs):
    all_samples = [load_shard_samples(d) for d in shard_dirs]

    combined = {}

    # Decide which keys to combine: intersection of ndarray-valued keys across shards
    common_keys = set(all_samples[0].keys())
    for s in all_samples[1:]:
        common_keys &= set(s.keys())
    # Exclude non-array/meta keys
    candidate_keys = [k for k in common_keys if isinstance(all_samples[0][k], np.ndarray)]

    # Prioritize known hyper/latent keys if present
    prioritized = ['eta', 'ell', 'sigma_noise', 'beta_pat', 'tau', 'lambdas', 'f', 'Lambda']
    keys_to_process = [k for k in prioritized if k in candidate_keys]
    # Include any remaining array keys not yet processed
    keys_to_process += [k for k in candidate_keys if k not in keys_to_process]

    # Combine selected keys
    for key in keys_to_process:
        try:
            combined[key] = compute_wasserstein_barycenter([s[key] for s in all_samples])
        except Exception as exc:
            logging.warning(f"Failed barycenter for key '{key}' ({exc}); using simple mean.")
            stacked = np.stack([s[key] for s in all_samples], axis=0)
            combined[key] = np.mean(stacked, axis=0)

    # If we have latent field, compute probabilities
    latent_key = 'f' if 'f' in combined else ('Lambda' if 'Lambda' in combined else None)
    if latent_key is not None:
        logits = combined[latent_key]
        combined['p'] = 1.0 / (1.0 + np.exp(-logits))

    # Metadata
    combined['num_shards'] = len(shard_dirs)
    combined['shard_results_dirs'] = shard_dirs
    combined['combination_method'] = 'wasserstein_barycenter' if ot is not None else 'euclidean_mean_fallback'
    combined['total_samples'] = _first_available_draw_count(all_samples[0]) * len(shard_dirs)
    return combined


def save_combined_results(combined_samples, args, failed_shards=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'./Res/sharded_hierarchical_gp_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    np.save(f"{results_dir}/combined_post_samples.npy", combined_samples)

    metadata = {
        'num_shards': combined_samples['num_shards'],
        'shard_results_dirs': combined_samples['shard_results_dirs'],
        'args': vars(args) if hasattr(args, '__dict__') else {},
        'timestamp': timestamp,
        'failed_shards': failed_shards or []
    }
    with open(f"{results_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

    # Write a short text summary
    with open(os.path.join(results_dir, 'summary_report.txt'), 'w') as f:
        f.write("COMBINED SHARDED RUN SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total shards: {combined_samples['num_shards']}\n")
        f.write(f"Combination method: {combined_samples.get('combination_method', 'unknown')}\n")
        f.write("Keys in combined samples:\n")
        for k, v in combined_samples.items():
            if isinstance(v, np.ndarray):
                f.write(f"  - {k}: {v.shape}\n")
            elif isinstance(v, list):
                f.write(f"  - {k}: list(len={len(v)})\n")
            else:
                f.write(f"  - {k}: {type(v)}\n")

    logging.info(f"Combined results saved to {results_dir}")
    return results_dir


def main():
    parser = argparse.ArgumentParser(description='Combine shard results from a manifest or list of directories')
    parser.add_argument('--manifest', type=str, default=None, help='Path to manifest.json produced by run_shards.py')
    parser.add_argument('--dirs', type=str, nargs='*', help='Explicit list of shard result directories')
    parser.add_argument('--shard_ids', type=int, nargs='*', help='If provided, auto-select the latest results dir for each shard id (expects directories named hierarchical_gp_shard{ID:03d}_TIMESTAMP)')
    args = parser.parse_args()

    setup_logging()

    try:
        if args.manifest:
            with open(args.manifest, 'r') as f:
                manifest = json.load(f)
            shard_dirs = manifest.get('shard_results_dirs', [])
            failed_shards = manifest.get('failed_shards', [])
            logging.info(f"Loaded {len(shard_dirs)} shard dirs from manifest; failed shards: {failed_shards}")
        else:
            shard_dirs = args.dirs or []
            failed_shards = []
            logging.info(f"Using shard dirs from CLI: {len(shard_dirs)}")

        if args.shard_ids:
            # For each shard id, find the latest matching directory
            discovered = []
            res_root = './Res'
            for sid in args.shard_ids:
                candidates = []
                if os.path.isdir(res_root):
                    for name in os.listdir(res_root):
                        full = os.path.join(res_root, name)
                        if not os.path.isdir(full):
                            continue
                        # Match legacy hierarchical prefix or gmrf with shard suffix
                        legacy_ok = name.startswith(f"hierarchical_gp_shard{int(sid):03d}_")
                        gmrf_ok = name.startswith("gmrf_") and (f"shard_{int(sid)}" in name)
                        if legacy_ok or gmrf_ok:
                            candidates.append(full)
                if candidates:
                    latest = max(candidates, key=os.path.getmtime)
                    discovered.append(latest)
                    logging.info(f"Shard {sid}: selected {latest}")
                else:
                    logging.warning(f"Shard {sid}: no directories found for shard id {sid}")
            shard_dirs = discovered if discovered else shard_dirs

        if not shard_dirs:
            raise ValueError('No shard result directories provided')

        combined = combine_from_dirs(shard_dirs)
        out_dir = save_combined_results(combined, args, failed_shards)
        logging.info(f"Combination finished. Output: {out_dir}")
    except Exception as exc:
        logging.error(f"Error during combination: {exc}", exc_info=True)
        raise


if __name__ == '__main__':
    main()




