import torch
from torch.utils.data import DataLoader

NOISE = [0.01, 0.05, 0.1, 0.25]


@torch.no_grad()
def noise_robustness(
    model,
    dataset,
    noise_levels=NOISE,
    samples_per_image=10000, # vou usar pra pegar a Probabilidade via Monte Carlo
    batch_size=128,
    device="cuda",
):
    model.eval().to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = {r: [] for r in noise_levels}

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        B = x.size(0)

        for r in noise_levels:
            correct = torch.zeros(B, device=device)

            remaining = samples_per_image

            while remaining > 0:
                n = min(batch_size, remaining)

                z = torch.randn(
                    (n, *x.shape),
                    device=device,
                )

                z = z / (
                    z.flatten(2).norm(dim=2)
                    .view(n, B, 1, 1, 1)
                    + 1e-12
                )

                noisy = (
                    x.unsqueeze(0)
                    + r * z
                ).view(n * B, *x.shape[1:])

                pred = model(noisy).argmax(1)
                pred = pred.view(n, B)

                correct += (pred == y.unsqueeze(0)).float().sum(0)

                remaining -= n

            probs = correct / samples_per_image
            results[r].append(probs.cpu())

    for r in noise_levels:
        results[r] = torch.cat(results[r])

    summary = {
        r: results[r].mean().item()
        for r in noise_levels
    }

    return summary, results








# CARREGAR MODELO
import numpy as np

params = list(np.load("./modelo/modelo_final_FL_DP.npy", allow_pickle=True))

# Quantas camadas carregou
print(f"Número de camadas: {len(params)}")

# Shape e stats de cada camada
for i, layer in enumerate(params):
    print(f"  Camada {i}: shape={layer.shape}, dtype={layer.dtype}, "
          f"min={layer.min():.4f}, max={layer.max():.4f}, mean={layer.mean():.4f}")