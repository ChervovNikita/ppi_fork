# PPI_GNN
In order to replicate the results mentioned in paper, please follow the following steps:
  1. Download the Pan's human features file and place the files at ../human_features/processed/. The link is given in PPI_GNN/Human_features/README.md. For the S.      cerevisiae PPI dataset, download the input feature file and place it at ../S. cerevisiae/processed/. The link is given in PPI_GNN/S. cerevisiae/README.md.
  2. Next use the command: python train.py to train the model.


The steps to predicting protein interactions on a new dataset are:
  1. First, get the node features from protein sequences using the SeqVec method (seqvec_embedding.py) and then build the protein graph (proteins_to_graphs.py).
  2. Next, use the command "python data_prepare.py" to get input features for the model.
  3. Then, use the command "python train.py" to train the model.
  4. Use the command: "python test.py" to evaluate the trained model on unseen data (test set).

To create the ppi_env environment, run:
$ conda env create -f ppi_env.yml

# PPI Prediction with Graph Neural Networks

This repository contains code for predicting protein-protein interactions using Graph Neural Networks.

## Setup

1. Install dependencies:
   ```
   conda env create -f ppi_env.yml
   conda activate ppi_env
   ```

2. Setup Weights & Biases for experiment tracking:
   ```
   pip install wandb
   wandb login
   ```
   Or update your API key in the `.env` file.

## Hyperparameter Tuning

The repository includes a hyperparameter tuning framework with integration to Weights & Biases for experiment tracking.

To run hyperparameter tuning:

```bash
# For random search with default settings (GCNN with descriptors)
python hp_tuning.py

# For grid search on basic GCNN
python hp_tuning.py --model GCNN --search_type grid

# For random search with specified trials
python hp_tuning.py --model GCNN_with_descriptors --search_type random --trials 20
```

Results are saved to `../masif_features/hp_results/` and tracked online in Weights & Biases.

## Configuration

The hyperparameter configuration is defined in `config.py`. You can modify:
- Base configuration: learning rate, batch size, etc.
- Model-specific configurations
- Hyperparameter search spaces

For online WandB tracking, ensure you have a valid API key in the `.env` file.
