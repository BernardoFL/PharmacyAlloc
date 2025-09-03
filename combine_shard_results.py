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
    """Compute an empirical Wasserstein barycenter across shards.

    - For 1D parameters (after flattening), uses W2-quantile averaging (exact in 1D).
    - For multi-D parameters, uses POT sliced Wasserstein barycenter when available.
    - If POT is unavailable or an error occurs, falls back to weighted Euclidean mean across shards.

    Returns (barycenter_samples, method_str).
    """
    n_distributions = len(mcmc_samples)
    if n_distributions == 0:
        raise ValueError("No sample arrays provided to compute barycenter.")

    # Default uniform weights over distributions
    if weights is None:
        weights = np.ones(n_distributions, dtype=float) / float(n_distributions)

    # Normalize shapes and detect dimensionality
    flat_samples = [arr.reshape(arr.shape[0], -1) for arr in mcmc_samples]
    dims = [fs.shape[1] for fs in flat_samples]
    if not all(d == dims[0] for d in dims):
        raise ValueError("All sample arrays must have the same flattened dimensionality.")
    dim = dims[0]

    # Utility: weighted Euclidean mean across shards (returns same shape as samples)
    def euclidean_mean() -> tuple:
        stacked = np.stack(mcmc_samples, axis=0)  # (S, num_draws, ...)
        w = np.array(weights, dtype=float).reshape(-1, *([1] * (stacked.ndim - 1)))
        return np.sum(w * stacked, axis=0), 'euclidean_mean_fallback'

    # 1D exact barycenter via quantile averaging
    if dim == 1:
        try:
            # Choose common number of support points K
            num_points_per = [fs.shape[0] for fs in flat_samples]
            K = int(np.min(num_points_per))
            if K <= 0:
                return euclidean_mean()

            # Uniform quantile grid
            qs = np.linspace(0.0, 1.0, K, endpoint=False) + 0.5 / K

            # For each distribution, compute quantiles via sorting and interpolation
            quantiles = []
            for fs in flat_samples:
                xs = np.sort(fs[:, 0])  # (n,)
                # Empirical CDF inverse at qs using linear interpolation between order stats
                ranks = qs * (len(xs) - 1)
                lo = np.floor(ranks).astype(int)
                hi = np.ceil(ranks).astype(int)
                frac = ranks - lo
                q = (1 - frac) * xs[lo] + frac * xs[hi]
                quantiles.append(q)

            quantiles = np.stack(quantiles, axis=0)  # (S, K)
            w = np.array(weights, dtype=float).reshape(-1, 1)
            bary_1d = np.sum(w * quantiles, axis=0).reshape(K, 1)

            # Reshape back to original trailing shape if needed
            original_shape = mcmc_samples[0].shape
            if len(original_shape) > 2:
                bary_1d = bary_1d.reshape(-1, *original_shape[1:])
            return bary_1d, 'wasserstein_barycenter_1d'
        except Exception:
            logging.exception("1D barycenter failed; using Euclidean mean fallback.")
            return euclidean_mean()

    # Multi-D sliced Wasserstein barycenter using POT
    if ot is None:
        return euclidean_mean()

    try:
        # Common number of support points for barycenter
        num_points_per = [fs.shape[0] for fs in flat_samples]
        K = int(np.min(num_points_per))
        if K <= 0:
            return euclidean_mean()

        # Optionally subsample each distribution to K points (without replacement)
        rng = np.random.default_rng(123)
        Xs = []
        for fs in flat_samples:
            n = fs.shape[0]
            if n == K:
                Xs.append(fs)
            elif n > K:
                idx = rng.choice(n, size=K, replace=False)
                Xs.append(fs[idx])
            else:
                # if fewer, resample with replacement to K
                idx = rng.choice(n, size=K, replace=True)
                Xs.append(fs[idx])

        # Initialize barycenter support with first distribution
        X_init = Xs[0].copy()

        # Try sliced barycenter API
        if hasattr(ot, 'sliced') and hasattr(ot.sliced, 'sliced_wasserstein_barycenter'):
            Xb = ot.sliced.sliced_wasserstein_barycenter(Xs, np.asarray(weights, dtype=float), X_init, n_projections=128)
            bary = np.asarray(Xb, dtype=float)
            original_shape = mcmc_samples[0].shape
            if len(original_shape) > 2:
                bary = bary.reshape(-1, *original_shape[1:])
            return bary, 'sliced_wasserstein_barycenter'

        # Fallback: free-support barycenter if available
        if hasattr(ot, 'lp') and hasattr(ot.lp, 'free_support_barycenter'):
            # Uniform weights over support points
            point_weights = [np.ones(x.shape[0], dtype=float) / float(x.shape[0]) for x in Xs]
            # Provide initial support and barycenter weights (uniform)
            b_init = np.ones(K, dtype=float) / float(K)
            Xb = ot.lp.free_support_barycenter(Xs, point_weights, np.asarray(weights, dtype=float), X_init=X_init, b=b_init)
            bary = np.asarray(Xb, dtype=float)
            original_shape = mcmc_samples[0].shape
            if len(original_shape) > 2:
                bary = bary.reshape(-1, *original_shape[1:])
            return bary, 'free_support_wasserstein_barycenter'

        # If no suitable POT API, fall back
        return euclidean_mean()
    except Exception:
        logging.exception("Multi-D barycenter failed; using Euclidean mean fallback.")
        return euclidean_mean()


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

    # Decide which keys to combine: only allowlist and present in ALL shards
    allowlist = {'beta_cond', 'beta_pat', 'lambdas', 'tau'}
    common_keys = set(all_samples[0].keys())
    for s in all_samples[1:]:
        common_keys &= set(s.keys())
    candidate_keys = [k for k in common_keys if k in allowlist and isinstance(all_samples[0][k], np.ndarray)]
    # Preserve stable ordering
    ordered = ['beta_cond', 'beta_pat', 'lambdas', 'tau']
    keys_to_process = [k for k in ordered if k in candidate_keys]

    # Combine selected keys
    per_key_method = {}
    for key in keys_to_process:
        try:
            arrays = [s[key] for s in all_samples]

            # Align number of draws by trimming to min draws
            min_draws = int(min(a.shape[0] for a in arrays))
            arrays = [a[:min_draws] for a in arrays]

            # Handle vector parameters with per-coordinate 1D barycenters
            if arrays[0].ndim == 2 and arrays[0].shape[1] > 1:
                # Align vector length by trimming to min length across shards
                min_len = int(min(a.shape[1] for a in arrays))
                arrays = [a[:, :min_len] for a in arrays]

                # Compute per-coordinate barycenter independently
                cols = []
                methods = []
                for j in range(min_len):
                    one_d_arrays = [a[:, j] for a in arrays]
                    b_j, m_j = compute_wasserstein_barycenter([x.reshape(-1, 1) for x in one_d_arrays])
                    # b_j has shape (K,1); squeeze to (K,)
                    cols.append(b_j.reshape(-1))
                    methods.append(m_j)
                bary = np.stack(cols, axis=1)
                # Use 1d tag if all 1d
                method = 'per_coordinate_wasserstein_1d' if all(m.startswith('wasserstein') for m in methods) else 'per_coordinate_mixed'
            else:
                # Scalar case
                bary, method = compute_wasserstein_barycenter(arrays)
            combined[key] = bary
            per_key_method[key] = method
        except Exception as exc:
            logging.warning(f"Failed barycenter for key '{key}' ({exc}); using simple mean.")
            try:
                arrays = [s[key] for s in all_samples]
                # Try mean after aligning shapes if possible
                flat_dims = [a.reshape(a.shape[0], -1).shape[1] for a in arrays]
                if all(d == flat_dims[0] for d in flat_dims):
                    min_draws = int(min(a.shape[0] for a in arrays))
                    arrays = [a[:min_draws] for a in arrays]
                stacked = np.stack(arrays, axis=0)
                combined[key] = np.mean(stacked, axis=0)
            except Exception:
                logging.exception(f"Simple mean also failed for key '{key}'. Skipping this key.")
                continue
            per_key_method[key] = 'euclidean_mean_fallback'

    # If we have latent field, compute probabilities
    latent_key = 'f' if 'f' in combined else ('Lambda' if 'Lambda' in combined else None)
    if latent_key is not None:
        logits = combined[latent_key]
        combined['p'] = 1.0 / (1.0 + np.exp(-logits))

    # Metadata
    combined['num_shards'] = len(shard_dirs)
    combined['shard_results_dirs'] = shard_dirs
    # Record method at both global and per-key levels
    any_wasserstein = any(m.startswith('wasserstein') or 'barycenter' in m for m in per_key_method.values())
    combined['combination_method'] = 'wasserstein' if any_wasserstein else 'euclidean_mean_fallback'
    combined['combination_method_by_key'] = per_key_method
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




