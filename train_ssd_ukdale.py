"""
train_ssd_ukdale.py
===================
Trains the standard Mamba-3 SISO SSD model (no Liquid Neural Network) on the
UK-DALE dataset for Non-Intrusive Load Monitoring (NILM).

This script is the pure-SSD baseline counterpart to train_ssd_lnn.py.
The only difference is use_liquid=False — the LiquidGate is disabled, so the
SSM uses fixed per-head decay (standard Mamba-3 SISO behaviour).

Model and SSD algorithm are imported directly from train_ssd_lnn.py to avoid
code duplication. Data loading and metrics use the shared Source Code modules,
matching the structure of train_tcn_lnn.py.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics, save_model

# ── Reuse model architecture and data helpers from train_ssd_lnn.py ──
sys.path.insert(0, os.path.dirname(__file__))
from train_ssd_lnn import (
    SSDLNNConfig, Mamba3LNNRegressor, get_device,
    APPLIANCES, THRESHOLDS,
    load_data, prepare_appliance_data,
)


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_ssd_model(data_dict, model_params, train_params, save_dir='models'):
    """Train the Mamba-3 SISO SSD model (use_liquid=False) on UK-DALE NILM data.

    Args:
        data_dict:    dict from load_and_preprocess_ukdale()
        model_params: dict with SSDLNNConfig fields (use_liquid forced False)
        train_params: dict with lr, epochs, patience
        save_dir:     directory to save checkpoints and plots

    Returns:
        (model, history, best_model_path)
    """
    os.makedirs(save_dir, exist_ok=True)

    train_loader = data_dict['train_loader']
    val_loader   = data_dict['val_loader']

    # Force use_liquid=False — this is the pure SSD baseline
    model_params = {**model_params, 'use_liquid': False}

    cfg = SSDLNNConfig(
        d_model     = model_params.get('d_model',      64),
        n_layer     = model_params.get('n_layer',       2),
        d_state     = model_params.get('d_state',      16),
        expand      = model_params.get('expand',        2),
        headdim     = model_params.get('headdim',      16),
        chunk_size  = model_params.get('chunk_size',   32),
        use_liquid  = False,
        tau_min     = model_params.get('tau_min',     0.1),
        liquid_scale= model_params.get('liquid_scale',1.0),
    )

    device = get_device()
    model  = Mamba3LNNRegressor(
        cfg,
        input_dim  = data_dict['input_size'],
        output_dim = data_dict['output_size'],
        device     = device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStarting SSD training on {device} | params: {n_params:,}")

    lr       = train_params.get('lr',       0.001)
    epochs   = train_params.get('epochs',   80)
    patience = train_params.get('patience', 20)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss   = float('inf')
    best_state      = None
    best_model_path = None
    counter         = 0

    for epoch in range(epochs):
        # ── Training phase ──
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # ── Validation phase ──
        model.eval()
        val_loss    = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                val_loss += loss.item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        y_scaler = data_dict.get('y_scaler')
        if y_scaler is not None:
            all_targets = y_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
            all_outputs = y_scaler.inverse_transform(all_outputs.reshape(-1, 1)).flatten()
        threshold = data_dict.get('threshold', 10.0)
        metrics = calculate_nilm_metrics(all_targets, all_outputs, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
            f"Val MAE: {metrics['mae']:.2f}, Val SAE: {metrics['sae']:.4f}, "
            f"Val F1: {metrics['f1']:.4f}"
        )

        # ── Early stopping + checkpoint ──
        if avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            counter         = 0
            best_state      = {k: v.clone() for k, v in model.state_dict().items()}
            best_model_path = os.path.join(save_dir, 'ssd_model_best.pth')
            save_model(model, model_params, train_params, metrics, best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    # ── Test evaluation (best checkpoint) ──
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = data_dict.get('test_loader')
    test_metrics = {}
    if test_loader is not None:
        model.eval()
        test_targets = []
        test_outputs = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_targets.append(targets.cpu().numpy())
                test_outputs.append(outputs.cpu().numpy())

        test_targets = np.concatenate(test_targets)
        test_outputs = np.concatenate(test_outputs)
        y_scaler = data_dict.get('y_scaler')
        if y_scaler is not None:
            test_targets = y_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()
            test_outputs = y_scaler.inverse_transform(test_outputs.reshape(-1, 1)).flatten()
        threshold = data_dict.get('threshold', 10.0)
        test_metrics = calculate_nilm_metrics(test_targets, test_outputs, threshold=threshold)
        print(
            f"Test  MAE: {test_metrics['mae']:.4f}  "
            f"SAE: {test_metrics['sae']:.4f}  "
            f"F1: {test_metrics['f1']:.4f}  "
            f"P: {test_metrics['precision']:.4f}  "
            f"R: {test_metrics['recall']:.4f}"
        )

    final_path = os.path.join(save_dir, 'ssd_model_final.pth')
    save_model(model, model_params, train_params, metrics, final_path)

    # ── Training history plot ──
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 2, 2)
    val_mae = [m['mae'] for m in history['val_metrics']]
    plt.plot(val_mae, label='Validation MAE')
    plt.title('Validation MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ssd_training_history.png'))
    plt.close()

    # ── Save history JSON ──
    with open(os.path.join(save_dir, 'ssd_history.json'), 'w') as f:
        json.dump(
            {
                'train_loss':  [float(v) for v in history['train_loss']],
                'val_loss':    [float(v) for v in history['val_loss']],
                'val_metrics': [{k: float(v) for k, v in m.items()} for m in history['val_metrics']],
                'test_metrics': {k: float(v) for k, v in test_metrics.items()} if test_metrics else {},
            },
            f, indent=4,
        )

    return model, history, best_model_path, test_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Train on All Appliances
# ──────────────────────────────────────────────────────────────────────────────

def train_ssd_all_appliances(house_number=1, window_size=100, save_dir='models/ssd'):
    """Train SSD on all appliances in the specified UK-DALE house.

    Args:
        house_number: House number in the UK-DALE dataset
        window_size:  Input sequence length (must be divisible by chunk_size=32)
        save_dir:     Directory to save the models

    Returns:
        (results dict, base_save_dir)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(base_save, exist_ok=True)

    raw_data   = load_data()
    appliances = {i: name for i, name in enumerate(APPLIANCES)}

    print(f"Training SSD models for {len(appliances)} appliances:")
    for idx, name in appliances.items():
        print(f"  Index {idx}: {name}")

    config = {
        'window_size': window_size,
        'timestamp':   timestamp,
        'appliances':  appliances,
    }
    with open(os.path.join(base_save, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    model_params = {
        'input_size':   1,
        'output_size':  1,
        'd_model':      64,
        'n_layer':      2,
        'd_state':      16,
        'expand':       2,
        'headdim':      16,
        'chunk_size':   25,
        'use_liquid':   False,
    }
    train_params = {'lr': 0.001, 'epochs': 80, 'patience': 20}

    results = {}

    for appliance_idx, appliance_name in appliances.items():
        print(f"\n{'-'*50}")
        print(f"Training SSD model for {appliance_name} (index {appliance_idx})")
        print(f"{'-'*50}\n")

        appliance_dir = os.path.join(base_save, appliance_name)
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            data_dict = prepare_appliance_data(raw_data, appliance_name, window_size=window_size)
            data_dict['threshold'] = THRESHOLDS[appliance_name]

            model, history, best_model_path, test_metrics = train_ssd_model(
                data_dict, model_params, train_params,
                save_dir=appliance_dir,
            )

            results[appliance_name] = {
                'model_path':      best_model_path,
                'appliance_index': appliance_idx,
                'final_metrics':   history['val_metrics'][-1] if history['val_metrics'] else None,
                'test_metrics':    test_metrics,
                'history':         history,
            }
            print(f"Successfully trained SSD model for {appliance_name}")

        except Exception as e:
            print(f"Error training SSD model for {appliance_name}: {str(e)}")

    # ── Summary JSON ──
    summary = {
        'timestamp':    timestamp,
        'house_number': house_number,
        'results': {
            name: {
                'model_path':    info['model_path'],
                'final_metrics': {k: float(v) for k, v in info['final_metrics'].items()}
                                  if info['final_metrics'] else None,
                'test_metrics':  {k: float(v) for k, v in info['test_metrics'].items()}
                                  if info.get('test_metrics') else None,
            }
            for name, info in results.items()
        },
    }
    with open(os.path.join(base_save, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # ── Combined val metrics plot (appliances × metrics grid) ──
    trained = {
        name: info for name, info in results.items()
        if info.get('history') and info['history']['val_metrics']
    }
    if trained:
        n_apps = len(trained)
        fig, axes = plt.subplots(n_apps, 3, figsize=(15, 4 * n_apps))
        if n_apps == 1:
            axes = [axes]
        fig.suptitle(
            f'Val Metrics per Epoch — SSD (80 epochs)',
            fontsize=13, fontweight='bold',
        )

        for row, (app_name, info) in enumerate(trained.items()):
            vm       = info['history']['val_metrics']
            epochs_x = range(1, len(vm) + 1)
            mae_vals = [m['mae'] for m in vm]
            sae_vals = [m['sae'] for m in vm]
            f1_vals  = [m['f1']  for m in vm]

            for col, (vals, ylabel, title) in enumerate([
                (mae_vals, 'MAE (W)',  f'{app_name} — MAE (W)'),
                (sae_vals, 'SAE',      f'{app_name} — SAE'),
                (f1_vals,  'F1',       f'{app_name} — F1'),
            ]):
                ax = axes[row][col]
                ax.plot(epochs_x, vals, color='#888888', linewidth=1.5)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Epoch', fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(base_save, 'ssd_val_metrics.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close()
        print(f"Val metrics plot saved to {base_save}/ssd_val_metrics.png")

    print("\nSSD training completed for all appliances!")
    print(f"Results saved to {base_save}")

    return results, base_save


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, save_dir = train_ssd_all_appliances(house_number=5)
