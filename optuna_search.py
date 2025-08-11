import optuna
import subprocess
import os
import sys

def objective(trial):
    # Suggest hyperparameters
    d_model = trial.suggest_categorical('d_model', [384, 512, 768, 1024])
    d_ff = trial.suggest_categorical('d_ff', [4 * d_model, 3072, 4096, 5120])
    num_layers = trial.suggest_int('num_layers', 4, 16, step=4)
    num_heads = trial.suggest_categorical('num_heads', [8, 12, 16])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    base_lr = trial.suggest_loguniform('base_lr', 1e-5, 5e-4)
    min_lr = trial.suggest_loguniform('min_lr', 1e-6, 1e-4)
    accumulation_steps = trial.suggest_categorical('accumulation_steps', [4, 8, 16])

    # Build command to run train.py with these parameters
    cmd = [
        sys.executable, 'train.py',
        '--dataset', 'owt',
        '--d_model', str(d_model),
        '--d_ff', str(d_ff),
        '--num_layers', str(num_layers),
        '--num_heads', str(num_heads),
        '--batch_size', str(batch_size),
        '--base_lr', str(base_lr),
        '--min_lr', str(min_lr),
        '--accumulation_steps', str(accumulation_steps),
        '--num_steps', '200',  # Use a small number for quick search
        '--profile'
    ]
    # Run the training script and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        output = result.stdout + '\n' + result.stderr
        # Parse the last reported loss from output
        for line in reversed(output.splitlines()):
            if 'loss=' in line:
                try:
                    loss_str = line.split('loss=')[1].split(',')[0]
                    return float(loss_str)
                except Exception:
                    continue
        # If no loss found, return a high value
        return 1e6
    except subprocess.TimeoutExpired:
        return 1e6

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print('Best trial:')
    print(study.best_trial)
    print('Best params:')
    print(study.best_params)

if __name__ == '__main__':
    main()
