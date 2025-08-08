# PharmacyAlloc: Bayesian Inference for Medical Data using Ising Models

This repository implements a Bayesian inference framework for analyzing medical data using anisotropic Ising models. The system processes patient-drug-condition relationships and performs statistical inference using advanced sampling techniques.

## Project Overview

The project analyzes medical data where:
- **Patients** are represented as rows in a binary matrix
- **Conditions** are represented as columns in a binary matrix  
- **Binary values** indicate presence/absence of conditions for each patient
- **Ising model** captures interactions between conditions using anisotropic parameters

## Code Structure and Architecture

### Main Entry Point: `run_model.py`

The main script orchestrates the entire inference pipeline:

```python
python run_model.py --pnum 500 --batch_size 20
```

**Key Components:**
- **Data Loading**: Uses `dataloader.py` to load patient data (controlled by `batch_size`)
- **Model Definition**: Defines the Ising model using `Source/Models.py` (grid size determined automatically from data)
- **Inference Engine**: Uses Numpyro for MCMC sampling
- **Likelihood Computation**: Uses `Source/JAXFDBayes.py` for efficient computation

### Core Modules

#### 1. Data Layer (`dataloader.py`)

**Purpose**: Loads and preprocesses medical data from CSV files.

**Key Classes:**
- `Drug`: Represents drugs with ATC classifications and associated conditions
- `Condition`: Represents medical conditions and their associated drugs
- `Patient`: Represents patients with multiple visits and conditions

**Main Functions:**
- `load_patients_from_csv_files()`: Loads data from CSV files
- `construct_A_matrix()`: Creates binary patient-condition matrix
- `load_data()`: Main data loading interface

**Data Flow:**
```
CSV Files → Patient Objects → Binary Matrix A → JAX Arrays
```

#### 2. Model Layer (`Source/Models.py`)

**Purpose**: Defines the anisotropic Ising model for medical data.

**Key Classes:**
- `IsingAnisotropic`: Main model class implementing anisotropic 2D Ising model

**Model Parameters:**
- `gamma`: Horizontal interaction parameter (within rows/patients)
- `beta_c`: Vertical interaction parameters (within columns/conditions)

**Key Methods:**
- `stat_m()`, `stat_p()`: Compute statistics for likelihood
- `ratio_m()`, `ratio_p()`: Compute ratio terms for FDBayes
- `sample()`: Generate samples from the model

**Model Structure:**
```
IsingAnisotropic(n_patients, n_conditions)
├── Horizontal interactions (gamma parameter)
└── Vertical interactions (beta_c parameters per condition)
```

#### 3. Inference Layer (`Source/JAXFDBayes.py`)

**Purpose**: Provides JAX-optimized likelihood computation for Bayesian inference.

**Key Components:**
- `JAXFDBayes` class: Main interface for likelihood computation
- JIT-compiled functions for efficient computation

**Core Functions:**
- `compute_ising_statistics_m/p()`: Compute statistics efficiently
- `compute_ratio_m/p()`: Compute ratio terms
- `fd_bayes_loss()`: Compute FDBayes loss function

**Optimization Features:**
- JAX JIT compilation for speed
- Vectorized operations
- Memory-efficient computation

#### 4. Distribution Layer (`Source/NumpyroDistributions.py`)

**Purpose**: Provides Numpyro-compatible distributions for the Ising model.

**Key Classes:**
- `IsingAnisotropicDistribution`: Numpyro distribution for Ising model

**Features:**
- Compatible with Numpyro MCMC samplers
- Implements sampling and log-probability methods
- Integrates with FDBayes likelihood

#### 5. Core Dependencies (`_dependency/`)

**Purpose**: Provides low-level Ising model implementation.

**Key Files:**
- `ising.py`: Core Ising model implementation
- `util.py`: Utility functions
- `ergm-sample.r`: R-based sampling utilities

## Data Flow Architecture

```
1. Data Loading (dataloader.py)
   ↓
2. Data Preprocessing (run_model.py)
   ↓
3. Model Initialization (Source/Models.py)
   ↓
4. Likelihood Setup (Source/JAXFDBayes.py)
   ↓
5. MCMC Inference (Numpyro)
   ↓
6. Results Storage (./Res/)
```

## Model Specification

### Ising Model Parameters

The anisotropic Ising model uses:

- **γ (gamma)**: Controls horizontal interactions between conditions within patients
- **β_c (beta_c)**: Controls vertical interactions within each condition column

### Likelihood Function

The model uses FDBayes (Fisher Divergence Bayes) likelihood:

```
L(θ|X) ∝ exp(-FD_loss(θ, X))
```

Where `FD_loss` is computed using:
- `stat_m()`: Statistics for ratio_m computation
- `stat_p()`: Statistics for ratio_p computation
- `ratio_m()` and `ratio_p()`: Ratio terms for loss computation

### Prior Specifications

- **γ**: Gaussian-InverseGamma hyperprior
- **β_c**: Horseshoe prior for sparsity
- **Hyperparameters**: Hierarchical structure for regularization

## Usage Examples

### Basic Inference

```bash
# Run with default parameters
python run_model.py

# Run with custom parameters
python run_model.py --dnum 500 --pnum 1000 --batch_size 100
```

### Parameter Tuning

```bash
# Use more posterior samples
python run_model.py --pnum 2000

# Limit patient batch size
python run_model.py --batch_size 50
```

## Output Structure

Results are saved in timestamped directories under `./Res/`:

```
./Res/ising_YYYYMMDD_HHMMSS/
├── post_samples.npy      # Posterior samples
├── mcmc_samples.npy      # Raw MCMC samples
└── diagnostics/          # MCMC diagnostics
```

## Dependencies

### Core Dependencies
- **JAX**: For efficient computation and automatic differentiation
- **Numpyro**: For Bayesian inference and MCMC sampling
- **PyTorch**: For model implementation (legacy)
- **NumPy/SciPy**: For numerical operations
- **Pandas**: For data manipulation

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate polypharm

# Install dependencies
pip install -r requirements.txt
```

## File Organization

```
PharmacyAlloc/
├── run_model.py              # Main entry point
├── dataloader.py             # Data loading and preprocessing
├── Source/
│   ├── Models.py             # Ising model implementation
│   ├── JAXFDBayes.py         # JAX-optimized likelihood
│   └── NumpyroDistributions.py # Numpyro distributions
├── _dependency/
│   ├── ising.py              # Core Ising implementation
│   ├── util.py               # Utility functions
│   └── ergm-sample.r         # R-based sampling
├── Data/                     # Input data files
├── Res/                      # Output results
└── logs/                     # Log files
```

## Key Features

1. **Efficient Computation**: JAX JIT compilation for fast likelihood evaluation
2. **Scalable Inference**: Numpyro MCMC with NUTS sampler
3. **Flexible Data**: Supports various medical data formats
4. **Robust Priors**: Hierarchical priors for regularization
5. **Comprehensive Logging**: Detailed logging and diagnostics

## Contributing

When contributing to this project:

1. Follow the existing code structure
2. Add appropriate logging statements
3. Update documentation for new features
4. Test with different data sizes
5. Ensure JAX compatibility for performance-critical code

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

[Add citation information here] 