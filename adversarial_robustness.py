import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms as T
from flwr_datasets import FederatedDataset

from PIL import Image

import numpy as np
import train
import parameters_federated


NOISE = [0.01, 0.05, 0.1, 0.25]

device = "cuda" if torch.cuda.is_available() else "cpu"


# Transformar o Dataset do flower em tupla do Pytorch
class TupleDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.transform(self.ds[i]["image"])
        return img, self.ds[i]["label"]


# Carrega o Dataset
def get_test_dataset():
    fds = FederatedDataset(dataset="ylecun/mnist", partitioners={})
    split = fds.load_split("test")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (parameters_federated.MEAN,),
            (parameters_federated.STD,)
        )
    ])

    return TupleDataset(split, transform)



# CARREGAR MODELO SALVO EM ARQUIVO
def load_model(num_classes, path="./modelos/modelo_final_FL_DP.npy",device='cpu'):
    model = train.Net(num_classes)
    params = list(np.load(path, allow_pickle=True))

    state_dict = model.state_dict()
    for i,key in enumerate(state_dict.keys()):
        state_dict[key] = torch.from_numpy(params[i]).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    print("Modelo Carregado!\n")
    return model
    



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


model = load_model(
    num_classes=parameters_federated.NUM_CLASSES, 
    path="./modelos/modelo_final_FL_DP.npy", 
    device=device
)

test_dataset = get_test_dataset()

summary, results = noise_robustness(
    model=model,
    dataset=test_dataset,
    device=device
)

print("Resultados:", summary)