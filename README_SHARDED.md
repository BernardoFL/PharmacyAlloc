# Sharded Hierarchical GP Model

This directory contains a sharded implementation of the hierarchical Gaussian Process model for large-scale patient data analysis. The sharded approach divides the data into manageable chunks and combines the results with proper likelihood correction.

## Overview

When dealing with large datasets, it's often computationally infeasible to run the full hierarchical GP model on all data at once. The sharded approach addresses this by:

1. **Dividing the data** into shards of configurable size (default: 1,000 patients)
2. **Running the model** on each shard independently
3. **Combining results** with likelihood elevated to the power of the number of shards
4. **Applying proper corrections** to maintain statistical validity

## Key Features

- **Automatic sharding**: Automatically determines the number of shards based on total dataset size
- **Likelihood correction**: Properly handles the likelihood elevation to power of number of shards
- **Robust error handling**: Continues processing even if some shards fail
- **Comprehensive logging**: Detailed logs for monitoring and debugging
- **Summary reports**: Automatic generation of summary reports for each run

## Files

- `run_model_sharded.py`: Main script for running sharded inference
- `test_sharded_model.py`: Test script to verify the implementation
- `README_SHARDED.md`: This documentation file

## Usage

### Basic Usage

```bash
# Run with default parameters (1,000 patients per shard, 500 samples per shard)
python run_model_sharded.py

# Run with custom parameters
python run_model_sharded.py --pnum 1000 --shard_size 500
```

### Command Line Arguments

- `--pnum`: Number of posterior samples per shard (default: 500)
- `--shard_size`: Number of patients per shard (default: 1,000)

### Example Commands

```bash
# Quick test with small parameters
python run_model_sharded.py --pnum 100 --shard_size 200

# Production run with large parameters
python run_model_sharded.py --pnum 2000 --shard_size 1000

# Single shard test (use very large shard size)
python run_model_sharded.py --pnum 500 --shard_size 10000
```

## How It Works

### 1. Data Sharding

The script first determines the total number of patients in the dataset and calculates the number of shards needed:

```python
total_patients = get_total_patients()
num_shards = (total_patients + shard_size - 1) // shard_size
```

### 2. Shard Execution

For each shard, the script:
- Calculates the starting index and size for the shard
- Runs `run_model.py` with appropriate parameters
- Captures the results directory

### 3. Result Combination

The results are combined using proper likelihood correction:

- **GP hyperparameters** (eta, ell, sigma_noise): Averaged across shards
- **Latent field f**: Combined with precision correction (variance divided by num_shards)
- **Probabilities p**: Combined via logit space correction

### 4. Likelihood Correction

When data is divided into m shards, the likelihood should be elevated to the power of m. This is implemented by:

- Multiplying the posterior precision by m
- Dividing the posterior variance by m
- Scaling sample deviations by sqrt(m)

## Output

The script creates a timestamped results directory in `./Res/` with:

- `combined_post_samples.npy`: Combined posterior samples with likelihood correction
- `metadata.pkl`: Metadata about the run (arguments, shard info, etc.)
- `summary_report.txt`: Human-readable summary of the run

### Sample Summary Report

```
SHARDED HIERARCHICAL GP MODEL RUN SUMMARY
==================================================

Timestamp: 2024-01-15 14:30:25
Total shards: 5
Shard size: 1000
Posterior samples per shard: 500
Total posterior samples: 2500
Likelihood correction factor: 5

Combined samples keys:
  eta: (2000,)
  ell: (2000,)
  sigma_noise: (2000,)
  f: (2000, 5000)
  p: (2000, 1000, 50)

Shard result directories:
  Shard 1: ./Res/hierarchical_gp_20240115_143025_1
  Shard 2: ./Res/hierarchical_gp_20240115_143025_2
  ...
```

## Testing

Run the test script to verify the implementation:

```bash
python test_sharded_model.py
```

The test script performs:
1. Small shard test with 100 patients per shard
2. Single shard test with very large shard size
3. Validation of result combination

## Error Handling

The script includes robust error handling:

- **Failed shards**: If some shards fail, the script continues with successful ones
- **Timeout protection**: Subprocess calls have timeout limits
- **Detailed logging**: Comprehensive logging for debugging
- **Graceful degradation**: Can proceed with partial results

## Performance Considerations

### Memory Usage

- Each shard runs independently, reducing peak memory usage
- Results are loaded and combined incrementally
- Large datasets can be processed without memory issues

### Computational Efficiency

- Shards can be run in parallel (future enhancement)
- Each shard uses the same number of MCMC samples
- Total computational time scales linearly with number of shards

### Storage

- Each shard creates its own results directory
- Combined results are stored separately
- Original shard results are preserved for inspection

## Limitations

1. **Sequential execution**: Shards are currently run sequentially (parallel execution is a future enhancement)
2. **Fixed shard size**: All shards have the same size (except possibly the last one)
3. **Memory for combination**: Loading all shard results for combination requires sufficient memory

## Future Enhancements

1. **Parallel execution**: Run shards in parallel using multiprocessing
2. **Dynamic shard sizing**: Adjust shard size based on available memory
3. **Incremental combination**: Combine results incrementally to reduce memory usage
4. **Distributed execution**: Support for distributed computing across multiple machines

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce shard size or number of posterior samples
2. **Timeout errors**: Increase timeout limits or reduce shard size
3. **Failed shards**: Check individual shard logs for specific errors
4. **Import errors**: Ensure all dependencies are installed in the correct environment

### Debugging

1. Check the main log file in `logs/sharded_model_run_*.log`
2. Examine individual shard logs in their respective directories
3. Review the summary report for overall statistics
4. Use the test script to verify basic functionality

## Mathematical Details

### Likelihood Correction

For a hierarchical GP model with data divided into m shards, the likelihood correction ensures:

1. **Posterior precision**: `Σ_post = Σ_prior + m * Σ_likelihood`
2. **Posterior mean**: Weighted average of shard-specific means
3. **Sample correction**: Scale deviations by `1/sqrt(m)`

This maintains the statistical validity of the combined posterior while allowing for efficient computation on large datasets.
