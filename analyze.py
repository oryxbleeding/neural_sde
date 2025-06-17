# analyze.py – finale Version (mit Momenten, Plots, Tabellen, Fehlerbehandlung)

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torchsde
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats

from data_loader import load_config, get_empirical_sample
from model import MortalitySDE
from utils import mmd_loss

def analyze():
    cfg = load_config()
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Analyse-Device: {device}")

    real_lifespans = torch.tensor(
        get_empirical_sample(cfg), dtype=torch.float32, device=device
    )

    model = MortalitySDE(
        hidden_size=cfg["model"]["hidden_size"],
        sigma_max=cfg["model"].get("sigma_max", 1.0),
    ).to(device)
    model.load_state_dict(torch.load(Path(cfg["training"]["checkpoints"]) / "model_final.pt", map_location=device))
    model.eval()

    batch_size = cfg["analysis"]["n_simulated"]
    dt = cfg["model"]["dt"]
    t_max = cfg["model"]["t_max"]
    ts = torch.linspace(0, t_max, int(t_max / dt) + 1, device=device)

    with torch.no_grad():
        y0 = 0.95 + 0.1 * torch.rand(batch_size, 1, device=device)
        paths = torchsde.sdeint(model, y0, ts, method="euler", dt=dt)

        soft_rates = torch.sigmoid(-10 * paths.squeeze(-1))
        weighted_time = (ts[:, None] * soft_rates).sum(dim=0)
        normalization = soft_rates.sum(dim=0) + 1e-8
        simulated_ages = (weighted_time / normalization).cpu().numpy()

        real_sample = real_lifespans[
            torch.randint(0, real_lifespans.shape[0], (batch_size,), device=device)
        ].cpu().numpy()

    # Statistische Kennzahlen
    mmd_value = mmd_loss(torch.tensor(real_sample), torch.tensor(simulated_ages)).item()
    ks_stat, _ = stats.ks_2samp(real_sample, simulated_ages)
    cvm_stat = stats.cramervonmises_2samp(real_sample, simulated_ages).statistic

    print(f"MMD = {mmd_value:.6f} |  KS = {ks_stat:.4f} |  CvM = {cvm_stat:.4f}")

    print("Momente  (Real,  Sim)")
    print(f"Mean : {np.mean(real_sample):.5f} {np.mean(simulated_ages):.5f}")
    print(f"Var  : {np.var(real_sample):.5f} {np.var(simulated_ages):.5f}")
    print(f"Skew : {stats.skew(real_sample):.5f} {stats.skew(simulated_ages):.5f}")
    print(f"Kurt : {stats.kurtosis(real_sample):.5f} {stats.kurtosis(simulated_ages):.5f}")

    # KDE-Plot
    plt.figure(figsize=(12, 6))
    sns.kdeplot(real_sample, bw_adjust=1.2, label="Reale Todesalter", fill=True)
    sns.kdeplot(simulated_ages, bw_adjust=1.2, label="Simulierte Todesalter", fill=True)

    plt.title("Vergleich reale vs. simulierte Todesalter (Quartal-SDE)", fontsize=14)
    plt.xlabel("Alter", fontsize=12)
    plt.ylabel("Dichte", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = Path(cfg["training"]["checkpoints"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "compare_kde.png")
    plt.show()

if __name__ == "__main__":
    analyze()