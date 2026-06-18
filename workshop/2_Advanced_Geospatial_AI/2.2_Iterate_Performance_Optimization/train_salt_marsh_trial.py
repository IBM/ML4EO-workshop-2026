#!/usr/bin/env python3

# © Copyright IBM Corporation 2026
# SPDX-License-Identifier: Apache-2.0

"""
Trial script for iterate HPO over the salt marsh downstream segmentation task.

iterate passes sampled and static parameters as environment variables:
  ITERATE_PARAM_*        – Any hyperparameter (e.g., LR, BATCH_SIZE, WEIGHT_DECAY)
  ITERATE_PARAM_CONFIG   – base terratorch config file (static)
  ITERATE_PARAM_EPOCHS   – number of training epochs per trial (static)
  ITERATE_TRIAL_NUMBER   – integer trial index
  ITERATE_OUT_FILE       – path where metrics must be written (name: value)

The script automatically handles any hyperparameters defined in hpo_config.yaml:
  1. Reads ALL parameters from ITERATE_PARAM_* environment variables dynamically.
  2. Maps them to appropriate terratorch CLI arguments (e.g., lr -> --optimizer.init_args.lr).
  3. Invokes the terratorch CLI via subprocess with all hyperparameter overrides.
  4. Parses Lightning's CSV log to extract the best validation loss.
  5. Writes the metric to ITERATE_OUT_FILE so iterate can read it.

To add new hyperparameters:
  - Simply add them to the 'hpo' section in hpo_config.yaml
  - The script will automatically detect and apply them
  - For custom mappings, update the param_mappings dict in build_terratorch_args()
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

# Script directory – used to resolve config paths regardless of CWD
SCRIPT_DIR = Path(__file__).resolve().parent


def get_params():
    """
    Read trial parameters from ITERATE_PARAM_* environment variables.
    
    Returns a dictionary of all hyperparameters found, plus config_path and epochs.
    Automatically discovers all ITERATE_PARAM_* variables and converts them to
    appropriate types.
    """
    params = {}
    
    # Scan all environment variables for ITERATE_PARAM_* prefix
    for key, value in os.environ.items():
        if key.startswith("ITERATE_PARAM_"):
            param_name = key.replace("ITERATE_PARAM_", "").lower()
            
            # Skip special parameters that aren't hyperparameters
            if param_name in ["config", "epochs"]:
                continue
            
            # Try to infer type and convert
            try:
                # Try integer first
                if "." not in value and "e" not in value.lower():
                    params[param_name] = int(value)
                else:
                    # Otherwise treat as float
                    params[param_name] = float(value)
            except ValueError:
                # If conversion fails, keep as string
                params[param_name] = value
    
    # Handle special static parameters
    config = os.environ.get(
        "ITERATE_PARAM_CONFIG",
        "config_salt_marsh.yaml",
    )
    # Resolve config path: prefer ITERATE_NB_DIR (set by the notebook) so the
    # config file is found in the notebook's directory even when the trial
    # script runs from a different location (e.g. /home/sagemaker-user/).
    config_path = Path(config)
    if not config_path.is_absolute():
        base = Path(os.environ.get("ITERATE_NB_DIR", str(SCRIPT_DIR)))
        config_path = base / config_path

    epochs = int(os.environ.get("ITERATE_PARAM_EPOCHS", "5"))
    
    return params, config_path, epochs


def find_metrics_csv(root: Path):
    """Return the most recently modified metrics.csv under *root*."""
    candidates = sorted(
        root.rglob("metrics.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def best_val_loss(metrics_csv: Path) -> float:
    """Parse Lightning's metrics.csv and return the minimum val/loss."""
    values = []
    with metrics_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell = row.get("val/loss", "").strip()
            if cell:
                try:
                    values.append(float(cell))
                except ValueError:
                    pass
    if not values:
        raise RuntimeError(f"No 'val/loss' rows found in {metrics_csv}")
    return min(values)


def build_terratorch_args(params: dict) -> list:
    """
    Build terratorch CLI arguments from hyperparameters.
    
    Maps parameter names to their corresponding terratorch CLI argument paths.
    Common mappings:
      - lr -> --optimizer.init_args.lr
      - batch_size -> --data.init_args.batch_size
      - weight_decay -> --optimizer.init_args.weight_decay
      - dropout -> --model.init_args.dropout
      
    For unknown parameters, attempts to map them intelligently based on name.
    """
    args = []
    
    # Define known parameter mappings
    param_mappings = {
        "lr": "--optimizer.init_args.lr",
        "batch_size": "--data.init_args.batch_size",
        "weight_decay": "--optimizer.init_args.weight_decay",
        "num_workers": "--data.init_args.num_workers",
        # Add more mappings as needed
    }
    
    for param_name, param_value in params.items():
        # Use known mapping if available
        if param_name in param_mappings:
            cli_arg = param_mappings[param_name]
        else:
            # For unknown parameters, try to infer the path
            # Default to optimizer.init_args for most training hyperparameters
            cli_arg = f"--optimizer.init_args.{param_name}"
            print(f"[INFO] Unknown parameter '{param_name}', mapping to {cli_arg}")
        
        args.extend([cli_arg, str(param_value)])
    
    return args


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Read parameters from ITERATE_PARAM_* environment variables
    # ------------------------------------------------------------------ #
    params, config_path, epochs = get_params()

    trial_num = os.environ.get("ITERATE_TRIAL_NUMBER", "?")
    out_file  = os.environ.get("ITERATE_OUT_FILE")

    # Print all hyperparameters
    params_str = "  ".join([f"{k}={v}" for k, v in params.items()])
    print(f"[trial {trial_num}] {params_str}  epochs={epochs}  config={config_path}")

    # ------------------------------------------------------------------ #
    # 2. Run terratorch fit with jsonargparse overrides for all hyperparameters.
    #    The dotted keys (e.g., `optimizer.init_args.lr`) follow
    #    Lightning-CLI jsonargparse conventions and work with any optimizer block.
    # ------------------------------------------------------------------ #
    log_dir = SCRIPT_DIR / f"hpo_trial_{trial_num}"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Write metrics.csv to a fully known path by passing an explicit CSVLogger.
    # Lightning's default root_dir handling can vary; this is deterministic.
    # CSVLogger(save_dir=log_dir, name="", version=0) writes to:
    #   {save_dir}/version_0/metrics.csv
    metrics_csv_path = log_dir / "version_0" / "metrics.csv"

    logger_json = (
        '{"class_path":"lightning.pytorch.loggers.CSVLogger",'
        '"init_args":{"save_dir":"' + str(log_dir) + '",'
        '"name":"","version":0}}'
    )

    # Build base command
    cmd = [
        sys.executable, "-m", "terratorch",
        "fit",
        "-c", str(config_path),
    ]
    
    # Add dynamically generated hyperparameter arguments
    cmd.extend(build_terratorch_args(params))
    
    # Add trainer configuration
    cmd.extend([
        "--trainer.max_epochs", str(epochs),
        "--trainer.default_root_dir", str(log_dir),
        "--trainer.logger", logger_json,
    ])
    
    result = subprocess.run(cmd)


    if result.returncode != 0:
        print(f"[trial {trial_num}] terratorch failed (exit {result.returncode})",
              file=sys.stderr)
        sys.exit(result.returncode)

    # ------------------------------------------------------------------ #
    # 3. Extract best validation loss from Lightning's auto-generated CSV
    # ------------------------------------------------------------------ #
    # Prefer the explicit logger path; fall back to rglob search.
    metrics_csv = metrics_csv_path if metrics_csv_path.exists() else find_metrics_csv(log_dir)
    if metrics_csv is None:
        print(f"[trial {trial_num}] ERROR: no metrics.csv found under {log_dir}",
              file=sys.stderr)
        sys.exit(1)

    val_loss = best_val_loss(metrics_csv)
    print(f"[trial {trial_num}] best val_loss = {val_loss:.6f}")

    # ------------------------------------------------------------------ #
    # 4. Write metric to ITERATE_OUT_FILE (iterate reads this for Optuna)
    # ------------------------------------------------------------------ #
    if out_file:
        with open(out_file, "w") as fh:
            fh.write(f"val_loss: {val_loss}\n")
        print(f"[trial {trial_num}] metrics written to {out_file}")
    else:
        # Fallback: print in the expected format if env var is not set
        print(f"val_loss: {val_loss}")

if __name__ == "__main__":
    main()
