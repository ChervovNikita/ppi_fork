import os
import random

# Set a shared random seed for reproducibility
SEED = 42

# Base model configurations
BASE_CONFIG = {
    "seed": SEED,
    "batch_size": 1,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "early_stopping_patience": 6,
    "milestones": [1, 5],  # For MultiStepLR scheduler
    "gamma": 0.5,  # Learning rate decay factor
    "num_features_pro": 1024,
    "output_dim": 128
}

# Specific model configurations
MODEL_CONFIGS = {
    "GCNN": {
        **BASE_CONFIG,
        "model_type": "GCNN",
        "use_descriptors": False,
        "dropout": 0.2
    },
    
    "GCNN_with_descriptors": {
        **BASE_CONFIG,
        "model_type": "GCNN",
        "use_descriptors": True,
        "transformer_dim": 31,  # 32-1 as in the models.py
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.2,
        "dim_feedforward": 128
    }
}

# Define hyperparameter search space for tuning
HP_SEARCH_SPACE = {
    "GCNN": {
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "dropout": [0.1, 0.2, 0.3, 0.5],
        "output_dim": [64, 128, 256]
    },
    
    "GCNN_with_descriptors": {
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "transformer_dim": [15, 23, 31, 47],
        "nhead": [2, 4, 8],
        "num_layers": [1, 2, 3, 4],
        "dropout": [0.1, 0.2, 0.3, 0.5],
        "dim_feedforward": [64, 128, 256, 512]
    }
}

def get_config(model_name="GCNN_with_descriptors"):
    """Get the configuration for a specific model.
    
    Args:
        model_name (str): Name of the model configuration to retrieve
        
    Returns:
        dict: Configuration dictionary for the specified model
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found in configurations. Available models: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[model_name]

def get_search_space(model_name="GCNN_with_descriptors"):
    """Get the hyperparameter search space for a specific model.
    
    Args:
        model_name (str): Name of the model to get search space for
        
    Returns:
        dict: Search space dictionary for the specified model
    """
    if model_name not in HP_SEARCH_SPACE:
        raise ValueError(f"Model {model_name} not found in search spaces. Available models: {list(HP_SEARCH_SPACE.keys())}")
    
    return HP_SEARCH_SPACE[model_name]

def update_model_config(model_name, **kwargs):
    """Update the configuration for a specific model.
    
    Args:
        model_name (str): Name of the model configuration to update
        **kwargs: Key-value pairs to update in the configuration
        
    Returns:
        dict: Updated configuration dictionary
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found in configurations. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name].copy()
    config.update(kwargs)
    return config 