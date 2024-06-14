
# iScore: A Machine Learning-Based Scoring Function for de novo Drug Discovery
Implementation of "iScore: A ML-Based Scoring Function for de novo Drug Discovery" by S.J. Mahdizadeh and L.A. Eriksson ([https://doi.org/10.1101/2024.04.02.587723](https://doi.org/10.1101/2024.04.02.587723)).

## Getting Started

1. **Prerequisites:**
   * Python 3.11.4
   * TensorFlow 2.14
   * scikit-learn 1.3.0
   * XGBoost 2.0.3
   * pandas 1.5.3

2. **Installation:**
   * Clone this repository: `git clone https://github.com/i-TripleD/iScore.git`
   * Change directory: `cd iScore`
   * Install required packages: `pip install -r requirements.txt`

## Usage

### Pre-Trained Models

Pre-trained iScore models and scaler are available in the "Models" directory. You can use these for immediate predictions. Refer to the paper for model details and performance metrics.

All scripts are executed from repository root. 

### Retraining Models

Before retraining, you'll need to generate the training data by running the following script. This assumes you have prepared your protein structures correctly, run dpocket, and sorted your pKd data.

#### Step 1: Prepare Training Data

```bash
# Generate Descroptors
python Training/Descriptors.py
```
- This script generates the necessary descriptors from your data. Ensure you have the required files in the correct locations as specified in the script.
- The un-prepared protein structure 1a1e is included as an example. Do not include this example protein when generating training data. Replace the 1a1e folder, pKd.csv and dpout_explicitp.txt for before retraining. 

#### Step 2: Retrain Models

Once your training data is ready, you can proceed to retrain the models:

```bash
# DNN Model
python Training/DNN_train.py

# XGBoost Model
python Training/XGB_train.py

# Random Forest Model
python Training/RF_train.py

# Hybrid Model
python Training/Hybrid_train.py

# Ultra-Fast Screening Model
python Training/UFS_train.py
```

**Note:** Training scripts are currently set up for 1*10-fold cross-validation. You can easily modify them to suit your specific cross-validation strategy.

### Benchmarking

Benchmark iScore models against the CASF2016 dataset:

```bash
# DNN Model
python Benchmark/DNN_benchmark.py

# XGBoost Model
python Benchmark/XGB_benchmark.py

# Random Forest Model
python Benchmark/RF_benchmark.py

# Hybrid Model
python Benchmark/Hybrid_benchmark.py
```

## Citation

If you find iScore useful in your research, please cite the following paper:

```
Mahdizadeh, S.J. and Eriksson, L.A. (2024). iScore: A ML-Based Scoring Function for de novo Drug Discovery. bioRxiv 2024.04.02.587723v1. doi: 10.1101/2024.04.02.587723.
```