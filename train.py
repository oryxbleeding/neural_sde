# train.py – finale Version (fehlerfrei, deterministisch, CUDA-kompatibel)

import os
# Setze CUBLAS-Workspace für deterministisches Verhalten auf CUDA
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import yaml
import torch
import torchsde
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
import seaborn as sns
import random
import numpy as np

from data_loader import load_config, get_empirical_sample
from model import MortalitySDE
from utils import mmd_loss

def set_all_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

def train():
    cfg = load_config()
    Path(cfg["training"]["checkpoints"]).mkdir(parents=True, exist_ok=True)

    set_all_seeds(42)

    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training läuft auf: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    real_lifespans = torch.tensor(
        get_empirical_sample(cfg), dtype=torch.float32, device=device
    )

    model = MortalitySDE(
        hidden_size=cfg["model"]["hidden_size"],
        sigma_max=cfg["model"].get("sigma_max", 1.0),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    dt = cfg["model"]["dt"]
    t_max = cfg["model"]["t_max"]
    ts = torch.linspace(0, t_max, int(t_max / dt) + 1, device=device)

    mmd_threshold = 0.0015
    losses = []
    early_stopped = False

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], label="MMD Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD Loss")
    ax.set_title("Training Monitoring")
    ax.grid(True)
    ax.legend()

    kde_ages = []
    kde_labels = []

    for epoch in trange(epochs, desc="Training"):
        model.train()

        y0 = 0.95 + 0.1 * torch.rand(batch_size, 1, device=device)
        paths = torchsde.sdeint(model, y0, ts, method="euler", dt=dt)

        soft_rates = torch.sigmoid(-10 * paths.squeeze(-1))
        weighted_time = (ts[:, None] * soft_rates).sum(dim=0)
        normalization = soft_rates.sum(dim=0) + 1e-8
        simulated_ages = weighted_time / normalization

        idx = torch.randint(0, real_lifespans.shape[0], (batch_size,), device=device)
        real_batch = real_lifespans[idx]

        loss = mmd_loss(real_batch, simulated_ages, sigma=cfg["training"]["mmd_kernel_bw"])

        if torch.isnan(loss) or loss.item() > 10.0:
            print(f"\n⚠️ Warnung: Loss explodiert bei Epoch {epoch}: {loss.item():.4f}")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if loss.item() < mmd_threshold:
            print(f"\n🏁 Ziel erreicht bei Epoch {epoch+1}: MMD Loss = {loss.item():.6f}")
            torch.save(model.state_dict(), Path(cfg["training"]["checkpoints"]) / "model_final.pt")
            plt.ioff()
            line.set_data(range(len(losses)), losses)
            ax.relim()
            ax.autoscale_view()
            plt.savefig(Path(cfg["training"]["checkpoints"]) / "training_loss.png")
            early_stopped = True
            break

        if (epoch + 1) % 500 == 0:
            line.set_data(range(len(losses)), losses)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)
            print(f"Epoch {epoch+1:5d} – MMD Loss: {loss.item():.6f}")

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                y0_small = 0.95 + 0.1 * torch.rand(1000, 1, device=device)
                paths_small = torchsde.sdeint(model, y0_small, ts, method="euler", dt=dt)
                soft_rates_small = torch.sigmoid(-10 * paths_small.squeeze(-1))
                weighted_time_small = (ts[:, None] * soft_rates_small).sum(dim=0)
                normalization_small = soft_rates_small.sum(dim=0) + 1e-8
                sim_ages_small = (weighted_time_small / normalization_small).cpu().numpy()

                kde_ages.append(sim_ages_small)
                kde_labels.append(f"Epoche {epoch+1}")

    if not early_stopped:
        plt.ioff()
        line.set_data(range(len(losses)), losses)
        ax.relim()
        ax.autoscale_view()
        plt.savefig(Path(cfg["training"]["checkpoints"]) / "training_loss.png")
        torch.save(model.state_dict(), Path(cfg["training"]["checkpoints"]) / "model_final.pt")
        print("\n✅ Training abgeschlossen und Modell gespeichert.")

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("coolwarm", len(kde_ages))

    for i, (ages, label) in enumerate(zip(kde_ages, kde_labels)):
        sns.kdeplot(ages, bw_adjust=1.2, label=label, color=colors[i], alpha=0.6)

    real_lifespans_np = real_lifespans.cpu().numpy()
    sns.kdeplot(real_lifespans_np, bw_adjust=1.2, label="Reale Todesalter", color="black", linestyle="--", linewidth=2)

    plt.title("Entwicklung simulierte Todesalter während des Trainings", fontsize=14)
    plt.xlabel("Alter", fontsize=12)
    plt.ylabel("Dichte", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(cfg["training"]["checkpoints"]) / "training_kde_evolution.png")
    plt.show()

if __name__ == "__main__":
    train()