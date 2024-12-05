# In-Context Learning with Covariance Study

This repository extends the original in-context learning codebase to study how transformers perform with different covariance structures. Added features include:

## Key Changes

- Added support for random covariance matrices in data sampling
- Implemented scale-varying evaluation (c=1,4,9)
- Added configurations for fixed/random covariance training
- Extended evaluation framework to test context length generalization

## Configurations

Two main training scenarios:
- `fixed_cov_*.yaml`: Fixed identity covariance matrix
- `random_cov_*.yaml`: Random diagonal covariance matrices (λᵢ ~ Exp(1))

Each available in three context lengths: N=40,70,100

## Running Experiments

```bash
# Train fixed covariance model
python src/train.py --config configs/fixed_cov_N40.yaml

# Train random covariance model  
python src/train.py --config configs/random_cov_N40.yaml

# Evaluate models
python src/eval.py path/to/model/dir
```

## Key Parameters

- Model: GPT2 (256 embedding, 12 layers, 8 heads)
- Dimensions: d=20
- Context lengths: N={40,70,100}
- Covariance scales: c={1,4,9}
- Data: Mean-zero Gaussian features

## Results

The evaluation produces metrics for:
- Fixed covariance testing
- Random covariance testing (multiple scales)
- Context length generalization