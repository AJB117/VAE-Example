from torchvision.utils import save_image, make_grid
import torch
import argparse
import torchvision as tv
import matplotlib.pyplot as plt

from dataclasses import dataclass
from tqdm import tqdm
from torchvision import transforms
from models import DDPM, DummyEpsModel
from typing import Tuple


@dataclass
class Args:
    batch_size: int
    epochs: int
    lr: float
    case_study: bool
    num_images: int
    save_name: str
    n_steps: int
    betas: Tuple[float, float]
    viz_trajectory: bool


def viz_trajectory(
    model: DDPM,
    device: torch.device = torch.device("cpu"),
) -> None:
    model.eval()
    print(model)

    with torch.no_grad():
        _, trajectory = model.sample(1, (1, 28, 28), device, return_trajectory=True)

        for i, x_i in enumerate(tqdm(trajectory)):
            plt.imshow(x_i.view(28, 28).cpu(), cmap="gray")
            plt.savefig(f"./trajectory/trajectory_{i}.png")
            plt.close()


def case_study(
    model: DDPM,
    num_images: int,
) -> None:
    model.eval()
    print(model)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        x_i = model.sample(num_images, (1, 28, 28), device)

        # make a grid of images using plt
        fig, ax = plt.subplots(1, num_images, figsize=(20, 20))
        for i in range(num_images):
            ax[i].imshow(x_i[i].view(28, 28).cpu(), cmap="gray")
            ax[i].axis("off")
        plt.savefig(f"./ddpm_outputs_{num_images}.png")
        plt.close()


# Credit to https://github.com/cloneofsimo/minDiffusion/blob/master/superminddpm.py for this implementation of DDPM
def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDPM(eps_model=DummyEpsModel(1), betas=args.betas, n_T=args.n_steps)
    model.to(device)

    if args.viz_trajectory:
        model.load_state_dict(torch.load("./ddpm_mnist.pth"))
        viz_trajectory(model, device)
        return

    if args.case_study:
        model.load_state_dict(torch.load("./ddpm_mnist.pth"))
        case_study(model, args.num_images)
        return

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = tv.datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=20
    )
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    for i in range(args.epochs):
        model.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = model(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        model.eval()
        with torch.no_grad():
            xh = model.sample(16, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ddpm_sample_{i}.png")

            torch.save(model.state_dict(), "./ddpm_mnist.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_name", type=str)

    parser.add_argument("--n_steps", default=100, type=int)
    parser.add_argument("--betas", default=[1e-4, 0.02], nargs="+", type=float)

    # visualizaiton of images
    parser.add_argument("--case_study", action="store_true")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--viz_trajectory", action="store_true")

    args = parser.parse_args()
    args = Args(**vars(args))
    main(args)
