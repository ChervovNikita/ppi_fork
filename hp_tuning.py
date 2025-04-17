import torch
import numpy as np
import os
import argparse
from itertools import product
from tqdm import tqdm
import json
from datetime import datetime
import wandb
from dotenv import load_dotenv
import gc
import traceback

# Load environment variables from .env file
load_dotenv()

# Force wandb to use online mode
os.environ["WANDB_MODE"] = "online"

from config import get_config, get_search_space, update_model_config, SEED
from models import GCNN, AttGNN, GCNN_with_descriptors
from data_prepare import trainloader, testloader
from metrics import get_accuracy, get_mse
import torch.nn as nn

# Set torch cuda memory management
if torch.cuda.is_available():
    # Configure PyTorch to be more aggressive with memory cleanup
    torch.cuda.empty_cache()
    # Set memory allocation strategy to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set all seeds for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train(model, device, trainloader, optimizer, scheduler, loss_func, epoch, num_epochs):
    print(f'Training on {len(trainloader)} samples.....')
    model.train()
    predictions_tr = torch.Tensor()
    labels_tr = torch.Tensor()
    total_loss = 0
    total_count = 0
    loop = tqdm(trainloader, total=len(trainloader), desc=f'Epoch {epoch}/{num_epochs}')
    for count,(prot_1, prot_2, label, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped) in enumerate(loop):
        prot_1 = prot_1.to(device)
        prot_2 = prot_2.to(device)
        mas1_straight = mas1_straight.to(device)
        mas1_flipped = mas1_flipped.to(device)
        mas2_straight = mas2_straight.to(device)
        mas2_flipped = mas2_flipped.to(device)
        optimizer.zero_grad()
        output = model(prot_1, prot_2, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped)
        predictions_tr = torch.cat((predictions_tr, output.cpu()), 0)
        labels_tr = torch.cat((labels_tr, label.view(-1,1).cpu()), 0)
        loss = loss_func(output, label.view(-1,1).float().to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_count += len(prot_1.x) if hasattr(prot_1, 'x') else 1
    scheduler.step()
    labels_tr = labels_tr.detach().numpy()
    predictions_tr = torch.sigmoid(torch.tensor(predictions_tr)).numpy()
    acc_tr = get_accuracy(labels_tr, predictions_tr, 0.5)
    print(f'Epoch [{epoch}/{num_epochs}] - train_loss: {total_loss / total_count} - train_accuracy: {acc_tr}')
    return total_loss / total_count, acc_tr

def predict(model, device, loader, epoch=0, num_epochs=0):
    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    with torch.no_grad():
        loop = tqdm(loader, total=len(loader), desc=f'Validation')
        for prot_1, prot_2, label, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped in loop:
            prot_1 = prot_1.to(device)
            prot_2 = prot_2.to(device)
            mas1_straight = mas1_straight.to(device)
            mas1_flipped = mas1_flipped.to(device)
            mas2_straight = mas2_straight.to(device)
            mas2_flipped = mas2_flipped.to(device)
            output = model(prot_1, prot_2, mas1_straight, mas1_flipped, mas2_straight, mas2_flipped)
            predictions = torch.cat((predictions, output.cpu()), 0)
            labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
    labels = labels.numpy()
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    return labels.flatten(), predictions.flatten()

def create_model(config):
    """Create model based on configuration."""
    if config["model_type"] == "GCNN":
        if config["use_descriptors"]:
            model = GCNN_with_descriptors(
                n_output=1,
                num_features_pro=config["num_features_pro"],
                output_dim=config["output_dim"],
                dropout=config["dropout"],
                transformer_dim=config["transformer_dim"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dim_feedforward=config["dim_feedforward"]
            )            
        else:
            model = GCNN(
                n_output=1, 
                num_features_pro=config["num_features_pro"],
                output_dim=config["output_dim"],
                dropout=config["dropout"]
            )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    return model

def cleanup_memory():
    """Force garbage collection and clear CUDA cache to free up memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_trial(config, trial_id):
    """Run a single hyperparameter trial."""
    print(f"\nRunning trial {trial_id}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize wandb for this trial
    try:
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "ppi-hyperparameter-tuning"),
            name=f"trial_{trial_id}_{config['model_type']}",
            config=config,
            group=config['model_type'],
            reinit=True,
            dir=os.environ.get("WANDB_LOG_DIR", "../masif_features/wandb_logs"),
            notes=os.environ.get("WANDB_NOTES", "Hyperparameter tuning")
        )
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        print("Continuing without wandb logging...")
        wandb_run = None
    
    # Create model and optimizer
    try:
        # Clean up memory before creating a new model
        cleanup_memory()
        
        model = create_model(config)
        model.to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=config["milestones"], 
            gamma=config["gamma"]
        )
        
        loss_func = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_val_accuracy = 0
        best_epoch = 0
        epochs_no_improve = 0
        early_stop_patience = 2  # Set to 2 epochs without improvement
        min_acceptable_accuracy = 0.6  # Minimum acceptable accuracy threshold
        
        results = {
            "trial_id": trial_id,
            "config": config,
            "epochs": [],
            "best_val_loss": float('inf'),
            "best_val_accuracy": 0,
            "best_epoch": 0,
            "status": "completed"
        }
        
        for epoch in range(config["num_epochs"]):
            try:
                # Train
                train_loss, train_acc = train(
                    model, device, trainloader, optimizer, scheduler, loss_func,
                    epoch+1, config["num_epochs"]
                )
                
                # Validate
                y_true, y_pred = predict(model, device, testloader)
                val_loss = get_mse(y_true, y_pred)
                val_accuracy = get_accuracy(y_true, y_pred, 0.5)
                
                print(f'Epoch [{epoch+1}/{config["num_epochs"]}] - val_loss: {val_loss} - val_accuracy: {val_accuracy}')
                
                # Log metrics to wandb if available
                if wandb_run:
                    wandb.log({
                        "epoch": epoch+1,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy
                    })
                
                # Save epoch results
                results["epochs"].append({
                    "epoch": epoch+1,
                    "train_loss": float(train_loss),  # Convert to native Python type
                    "train_accuracy": float(train_acc),  # Convert to native Python type
                    "val_loss": float(val_loss),  # Convert to native Python type
                    "val_accuracy": float(val_accuracy)  # Convert to native Python type
                })
                
                # Check for best validation performance
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch
                    # Optionally save model
                    model_save_path = f"../masif_features/models/trial_{trial_id}_best_acc.pth"
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
                    
                    # Also log best model to wandb using Artifact
                    if wandb_run:
                        wandb.run.summary["best_val_accuracy"] = best_val_accuracy
                        wandb.run.summary["best_epoch"] = best_epoch
                        
                        # Create and log a model artifact instead of using wandb.save
                        model_artifact = wandb.Artifact(
                            name=f"model-trial-{trial_id}", 
                            type="model",
                            description=f"Best model for trial {trial_id} with accuracy {best_val_accuracy:.4f}"
                        )
                        model_artifact.add_file(model_save_path)
                        wandb.log_artifact(model_artifact)
                
                # Accuracy threshold stopping check - stop if accuracy is too low
                if val_accuracy < min_acceptable_accuracy:
                    print(f'Stopping early: validation accuracy {val_accuracy:.4f} below threshold {min_acceptable_accuracy}')
                    if wandb_run:
                        wandb.run.summary["stopped_early"] = True
                        wandb.run.summary["stopped_at_epoch"] = epoch+1
                        wandb.run.summary["stopped_reason"] = f"Low accuracy below {min_acceptable_accuracy}"
                    break
                    
                # Early stopping check (validation loss not improving)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epoch > 5 and epochs_no_improve >= early_stop_patience:
                    print(f'Early stopping after {early_stop_patience} epochs without improvement!')
                    if wandb_run:
                        wandb.run.summary["stopped_early"] = True
                        wandb.run.summary["stopped_at_epoch"] = epoch+1
                        wandb.run.summary["stopped_reason"] = f"No improvement for {early_stop_patience} epochs"
                    break
                    
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory error during training!")
                print("This configuration requires too much GPU memory.")
                results["status"] = "failed"
                results["failure_reason"] = "CUDA out of memory during training"
                if wandb_run:
                    wandb.run.summary["status"] = "failed"
                    wandb.run.summary["failure_reason"] = "CUDA out of memory"
                break
            except Exception as e:
                print(f"Error during training: {e}")
                traceback.print_exc()
                results["status"] = "failed"
                results["failure_reason"] = str(e)
                if wandb_run:
                    wandb.run.summary["status"] = "failed"
                    wandb.run.summary["failure_reason"] = str(e)
                break
        
        # Update results with best metrics
        results["best_val_loss"] = float(best_val_loss)  # Convert to native Python type
        results["best_val_accuracy"] = float(best_val_accuracy)  # Convert to native Python type
        results["best_epoch"] = int(best_epoch)  # Convert to native Python type
        results["completed_epochs"] = int(epoch + 1)  # Convert to native Python type
        
        # Add a final summary to wandb
        if wandb_run:
            wandb.run.summary["completed_epochs"] = int(epoch + 1)
            wandb.run.summary["best_val_loss"] = float(best_val_loss)
            wandb.run.summary["best_val_accuracy"] = float(best_val_accuracy)
            wandb.run.summary["status"] = results["status"]
            
            # Finish the wandb run
            wandb.finish()
        
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory error during model creation!")
        print("This configuration requires too much GPU memory.")
        results = {
            "trial_id": trial_id,
            "config": config,
            "status": "failed",
            "failure_reason": "CUDA out of memory during model creation",
            "epochs": []
        }
        if wandb_run:
            wandb.run.summary["status"] = "failed"
            wandb.run.summary["failure_reason"] = "CUDA out of memory during model creation"
            wandb.finish()
    except Exception as e:
        print(f"Error in trial: {e}")
        traceback.print_exc()
        results = {
            "trial_id": trial_id,
            "config": config,
            "status": "failed",
            "failure_reason": str(e),
            "epochs": []
        }
        if wandb_run:
            wandb.run.summary["status"] = "failed"
            wandb.run.summary["failure_reason"] = str(e)
            wandb.finish()
    
    # Clean up memory after trial
    try:
        del model, optimizer, scheduler
    except:
        pass
    cleanup_memory()
    
    return results

def grid_search(model_name):
    """Perform grid search over hyperparameters."""
    base_config = get_config(model_name)
    search_space = get_search_space(model_name)
    
    # Create all combinations of hyperparameters
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(product(*values))
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../masif_features/hp_results/{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting hyperparameter tuning for {model_name}")
    print(f"Total combinations to try: {len(combinations)}")
    
    # Initialize wandb
    try:
        print("Logging in to Weights & Biases in online mode...")
        wandb.login(key=os.environ.get("WANDB_API_KEY", None))
        print("Successfully logged in to Weights & Biases")
    except Exception as e:
        print(f"Warning: WandB login failed: {e}")
        print("Make sure you have a valid API key in your .env file")
        return
    
    for i, combination in enumerate(combinations):
        # Update config with this combination
        trial_config = base_config.copy()
        for j, key in enumerate(keys):
            # Convert NumPy types to native Python types
            if isinstance(combination[j], np.integer):
                trial_config[key] = int(combination[j])
            elif isinstance(combination[j], np.floating):
                trial_config[key] = float(combination[j])
            else:
                trial_config[key] = combination[j]
        
        # Run trial
        trial_result = run_trial(trial_config, i+1)
        results.append(trial_result)
        
        # Save interim results
        with open(f"{results_dir}/trial_{i+1}.json", 'w') as f:
            json.dump(trial_result, f, indent=2, cls=NumpyEncoder)
        
        # Save all results so far
        with open(f"{results_dir}/all_results.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Find best configuration
    best_trial = max(results, key=lambda x: x["best_val_accuracy"])
    print(f"\nBest trial: {best_trial['trial_id']}")
    print(f"Best validation accuracy: {best_trial['best_val_accuracy']}")
    print(f"Configuration: {json.dumps(best_trial['config'], indent=2, cls=NumpyEncoder)}")
    
    # Save best configuration separately
    with open(f"{results_dir}/best_config.json", 'w') as f:
        json.dump(best_trial, f, indent=2, cls=NumpyEncoder)
    
    return best_trial

def random_search(model_name, num_trials=10):
    """Perform random search over hyperparameters."""
    base_config = get_config(model_name)
    search_space = get_search_space(model_name)
    
    # Modify search space to reduce memory usage for transformer dimensions
    if "dim_feedforward" in search_space:
        # Limit maximum dimension size to reduce memory usage
        search_space["dim_feedforward"] = [d for d in search_space["dim_feedforward"] if d <= 256]
    
    if "transformer_dim" in search_space:
        # Prefer smaller transformer dimensions
        search_space["transformer_dim"] = [d for d in search_space["transformer_dim"] if d <= 31]
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../masif_features/hp_results/{model_name}_random_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting random hyperparameter search for {model_name}")
    print(f"Total trials: {num_trials}")
    print(f"Search space (adjusted for memory): {json.dumps(search_space, indent=2, cls=NumpyEncoder)}")
    
    # Initialize wandb
    try:
        print("Logging in to Weights & Biases in online mode...")
        wandb.login(key=os.environ.get("WANDB_API_KEY", None))
        print("Successfully logged in to Weights & Biases")
    except Exception as e:
        print(f"Warning: WandB login failed: {e}")
        print("Make sure you have a valid API key in your .env file")
        return
    
    for i in range(num_trials):
        # Create random configuration
        trial_config = base_config.copy()
        for key, values in search_space.items():
            value = np.random.choice(values)
            # Convert NumPy types to native Python types
            if isinstance(value, np.integer):
                trial_config[key] = int(value)
            elif isinstance(value, np.floating):
                trial_config[key] = float(value)
            else:
                trial_config[key] = value
        
        # Run trial
        trial_result = run_trial(trial_config, i+1)
        results.append(trial_result)
        
        # Save interim results
        with open(f"{results_dir}/trial_{i+1}.json", 'w') as f:
            json.dump(trial_result, f, indent=2, cls=NumpyEncoder)
        
        # Save all results so far
        with open(f"{results_dir}/all_results.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        # Early exit if we've had too many failures
        completed_trials = sum(1 for r in results if r.get("status") == "completed")
        failed_trials = sum(1 for r in results if r.get("status") == "failed")
        
        # If more than 70% of trials have failed, it might indicate issues with memory or config
        if i >= 3 and failed_trials > (i * 0.7):
            print(f"Warning: {failed_trials}/{i+1} trials have failed. Consider adjusting parameters to reduce memory usage.")
    
    # Find best configuration from completed trials
    completed_results = [r for r in results if r.get("status") == "completed"]
    if completed_results:
        best_trial = max(completed_results, key=lambda x: x["best_val_accuracy"])
        print(f"\nBest trial: {best_trial['trial_id']}")
        print(f"Best validation accuracy: {best_trial['best_val_accuracy']}")
        print(f"Configuration: {json.dumps(best_trial['config'], indent=2, cls=NumpyEncoder)}")
        
        # Save best configuration separately
        with open(f"{results_dir}/best_config.json", 'w') as f:
            json.dump(best_trial, f, indent=2, cls=NumpyEncoder)
    else:
        
        print("\nNo trials completed successfully. Consider adjusting your search space to use less memory.")
        best_trial = None
    
    return best_trial

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for PPI models')
    parser.add_argument('--model', type=str, default='GCNN_with_descriptors',
                        choices=['GCNN', 'GCNN_with_descriptors'],
                        help='Model type for hyperparameter tuning')
    parser.add_argument('--search_type', type=str, default='random',
                        choices=['grid', 'random'],
                        help='Search strategy (grid or random)')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials for random search')
    args = parser.parse_args()
    
    if args.search_type == 'grid' 
        best_trial = grid_search(args.model)
    else:
        best_trial = random_search(args.model, args.trials)
    
    print("\nHyperparameter tuning completed!")
    print(f"Best configuration saved to ../masif_features/hp_results/")

if __name__ == "__main__":
    main()
