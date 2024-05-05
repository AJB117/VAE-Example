import torch
import argparse
import torchvision as tv
import matplotlib.pyplot as plt
import torchvision
import random
import numpy as np

from dataclasses import dataclass
from torch import Tensor
from torchvision import transforms
from models import VAE_MLP, VAE_CNN
from typing import Tuple, Union
from sklearn.preprocessing import MinMaxScaler
from matplotlib import offsetbox


@dataclass
class Args:
    batch_size: int
    epochs: int
    lr: float
    latent_dim: int
    hidden_layer_sizes: list
    case_study: bool
    num_images: int
    model_type: str
    latents_to_sample: int
    save_name: str
    viz_latent: str


def loss_fn(
    recon_x: Tensor, x: Tensor, mu: Tensor, var: Tensor
) -> Tuple[Tensor, Tensor]:
    recon_loss = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 784), x.view(-1, 784), reduction="sum"
    )

    kl_div = -0.5 * torch.sum(1 + torch.log(var**2) - mu**2 - var**2)

    return recon_loss, kl_div


# From https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
def plot_embedding(X, images, labels, title):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)

    targets = images.targets[: len(labels)]
    for label in labels:
        ax.scatter(
            *X[targets == label].T,
            marker=f"${label}$",
            s=60,
            color=plt.cm.Dark2(label),
            alpha=0.425,
            zorder=2,
        )

    shown_images = np.array([[1.0, 1.0]])
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)

        if np.min(dist) < 4e-3:
            continue

        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(images[i][0].squeeze(0), cmap=plt.cm.gray_r), X[i]
        )

        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

        ax.set_title(title)
        ax.axis("off")


def viz_latent_space(
    model: Union[VAE_MLP, VAE_CNN],
    test_data: torchvision.datasets.MNIST,
    device: torch.device = torch.device("cpu"),
    model_type: str = "mlp",
) -> None:
    model.eval()
    print(model)

    random_indices = [random.randint(0, len(test_data) - 1) for _ in range(500)]
    images = test_data.data[random_indices]

    with torch.no_grad():
        if model_type == "cnn":
            x = images.unsqueeze(0).unsqueeze(0).float()
        else:
            x = images.view(-1, 784).float()

        x = x.to(device)

        mean, var = model.encoder(x)
        z = model.reparameterize(mean, var)

        _, axarr = plt.subplots(1, 2)

        z = z.cpu().numpy()
        y = test_data.targets[random_indices].numpy()
        plot_embedding(z, test_data, y, "Latent embedding of the digits")

        plt.show()


def case_study(
    model: Union[VAE_MLP, VAE_CNN],
    num_images: int,
    test_data: torchvision.datasets.MNIST,
    model_type: str,
    device: torch.device = torch.device("cpu"),
) -> None:
    model.eval()
    print(model)

    random_indices = [random.randint(0, len(test_data) - 1) for _ in range(num_images)]
    images = test_data[random_indices]

    with torch.no_grad():
        for image in images:
            if model_type == "cnn":
                x = image.unsqueeze(0).unsqueeze(0).float()
            else:
                x = image.view(-1, 784).float()

            x = x.to(device)

            x_pred, _, _ = model(x)
            x_pred = x_pred.view(-1, 1, 28, 28)

            _, axarr = plt.subplots(1, 2)

            axarr[0].imshow(x.view(28, 28).cpu(), cmap="gray")
            axarr[0].set_title("Original")

            axarr[1].imshow(x_pred.view(28, 28).cpu(), cmap="gray")
            axarr[1].set_title("Reconstructed")

            plt.show()


def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = tv.datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    test = tv.datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    if args.case_study or args.viz_latent:
        if args.model_type == "cnn":
            model = VAE_CNN(
                latent_dim=args.latent_dim, latents_to_sample=args.latents_to_sample
            )
        else:
            model = VAE_MLP(
                in_dim=784,
                latent_dim=args.latent_dim,
                hidden_layer_sizes=args.hidden_layer_sizes,
                latents_to_sample=args.latents_to_sample,
            )

        model.load_state_dict(
            torch.load(f"vae_{args.save_name}.pth", map_location=device)
        )
        model = model.to(device)

        if args.case_study:
            case_study(model, args.num_images, test.data, args.model_type, device)
        elif args.viz_latent:
            viz_latent_space(model, train, device, args.model_type)

        return

    n = train.data.shape[1] ** 2

    if args.model_type == "cnn":
        model = VAE_CNN(
            latent_dim=args.latent_dim, latents_to_sample=args.latents_to_sample
        )
    else:
        model = VAE_MLP(
            in_dim=n,
            latent_dim=args.latent_dim,
            hidden_layer_sizes=args.hidden_layer_sizes,
            latents_to_sample=args.latents_to_sample,
        )

    model = model.to(device)
    print(model)

    save_name = args.save_name if args.save_name else args.model_type

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "max",
        factor=0.5,
        patience=10,
        threshold=0.001,
        cooldown=0,
        min_lr=0.0001,
        verbose=True,
    )

    recon_losses_train, kl_divs_train = [], []
    recon_losses_eval, kl_divs_eval = [], []

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            # for batch in tqdm(train_loader):
            batch[0] = batch[0].to(device)
            if args.model_type == "cnn":
                x = batch[0].float()
            else:
                x = batch[0].view(-1, 784)

            optimizer.zero_grad()
            x_pred, mean, var = model(x)
            recon_loss, kl_div = loss_fn(x_pred, x, mean, var)

            loss = recon_loss + kl_div
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        recon_losses_train.append(recon_loss.item() / len(train_loader.dataset))
        kl_divs_train.append(kl_div.item() / len(train_loader.dataset))

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch[0] = batch[0].to(device)
                if args.model_type == "cnn":
                    x = batch[0].float()
                else:
                    x = batch[0].view(-1, 784)

                x_pred, mean, var = model(x)
                recon_loss, kl_div = loss_fn(x_pred, x, mean, var)

                loss = recon_loss + kl_div
                val_loss += loss.item()

        scheduler.step(val_loss)

        recon_losses_eval.append(recon_loss.item() / len(val_loader.dataset))
        kl_divs_eval.append(kl_div.item() / len(val_loader.dataset))

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving model with new best loss ", val_loss)

            torch.save(model.state_dict(), f"vae_{save_name}.pth")

        print(f"Epoch {epoch}: Train loss {train_loss}, Val loss {val_loss}")

    plt.plot(recon_losses_train, label="Reconstruction Loss Train")
    plt.plot(kl_divs_train, label="KL Divergence Train")
    plt.plot(recon_losses_eval, label="Reconstruction Loss Eval")
    plt.plot(kl_divs_eval, label="KL Divergence Eval")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss types over time")
    plt.savefig(f"loss_curves_{save_name}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--model_type", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--latents_to_sample", type=int, default=1)
    parser.add_argument("--save_name", type=str)

    # for MLP
    parser.add_argument(
        "--hidden_layer_sizes", type=int, nargs="+", default=[212, 192, 128]
    )

    # visualizaiton of images
    parser.add_argument("--case_study", action="store_true")
    parser.add_argument("--num_images", type=int, default=10)

    # visualization of the latent code
    parser.add_argument("--viz_latent", action="store_true")

    args = parser.parse_args()
    args = Args(**vars(args))
    main(args)
