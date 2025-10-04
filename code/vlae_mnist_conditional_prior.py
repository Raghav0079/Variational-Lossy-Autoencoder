# vlae_mnist_conditional_prior.py
"""
VLAE for MNIST with:
 - conditional top-down prior p(z1 | z2)
 - KL annealing (linear warmup over kl_anneal_epochs)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# ---------------------------
# Config / Hyperparameters
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr = 1e-3
epochs = 30
latent_dims = [32, 16]  # [z1_dim, z2_dim]
hidden_dim = 256
save_dir = "vlae_mnist_cond_prior_out"
os.makedirs(save_dir, exist_ok=True)
log_interval = 100

# KL annealing schedule
kl_anneal_epochs = 10  # linear warmup over first N epochs (set 0 to disable)
# total anneal steps (iterations)
# will be computed after train_loader is created

# ---------------------------
# Data
# ---------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

total_train_iters = kl_anneal_epochs * len(train_loader) if kl_anneal_epochs > 0 else 1

# ---------------------------
# Utils: gaussian KL
# ---------------------------
def kl_gaussian(mu_q, logvar_q, mu_p=None, logvar_p=None):
    """
    KL( N(mu_q, var_q) || N(mu_p, var_p) )
    If mu_p/logvar_p is None, assumes standard normal (0, I).
    Returns sum over all elements (batch summed).
    """
    if mu_p is None:
        # KL to standard normal
        return -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())
    else:
        # general formula
        # KL = 0.5 * sum( log(var_p/var_q) + (var_q + (mu_q-mu_p)^2)/var_p - 1 )
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        # avoid tiny values
        ratio = var_q / (var_p + 1e-8)
        term = ((mu_q - mu_p).pow(2) + var_q) / (var_p + 1e-8)
        kl = 0.5 * torch.sum(torch.log(var_p + 1e-8) - torch.log(var_q + 1e-8) + term - 1.0)
        return kl

# ---------------------------
# Modules
# ---------------------------
class ConvEncoder(nn.Module):
    def __init__(self, z_dims=[32,16], hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 14 -> 7
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 64 * 7 * 7
        self.shared = nn.Linear(conv_out, hidden_dim)
        self.mu1 = nn.Linear(hidden_dim, z_dims[0])
        self.logvar1 = nn.Linear(hidden_dim, z_dims[0])
        self.mu2 = nn.Linear(hidden_dim, z_dims[1])
        self.logvar2 = nn.Linear(hidden_dim, z_dims[1])

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(self.shared(h))
        mu1, logvar1 = self.mu1(h), self.logvar1(h)
        mu2, logvar2 = self.mu2(h), self.logvar2(h)
        return (mu1, logvar1), (mu2, logvar2)

class PriorNet(nn.Module):
    """
    Maps z2 -> parameters of p(z1 | z2): mu_p(z2), logvar_p(z2)
    """
    def __init__(self, z2_dim=16, z1_dim=32, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z2_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, z1_dim)
        self.logvar = nn.Linear(hidden_dim, z1_dim)

    def forward(self, z2):
        h = self.net(z2)
        mu_p = self.mu(h)
        logvar_p = self.logvar(h)
        return mu_p, logvar_p

class LadderDecoder(nn.Module):
    def __init__(self, z_dims=[32,16], hidden_dim=256):
        super().__init__()
        fused_dim = z_dims[0] + z_dims[1]
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64 * 7 * 7),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # 14 -> 28
        )

    def forward(self, z1, z2):
        fused = torch.cat([z1, z2], dim=1)
        h = self.fc(fused)
        x_logits = self.deconv(h)
        x_recon = torch.sigmoid(x_logits)
        return x_recon

class VLAE(nn.Module):
    def __init__(self, z_dims=[32,16], hidden_dim=256):
        super().__init__()
        self.encoder = ConvEncoder(z_dims, hidden_dim)
        self.prior_net = PriorNet(z2_dim=z_dims[1], z1_dim=z_dims[0], hidden_dim=128)
        self.decoder = LadderDecoder(z_dims, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        (mu1, logvar1), (mu2, logvar2) = self.encoder(x)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        x_recon = self.decoder(z1, z2)
        latents = {
            "q_z1": (mu1, logvar1),
            "q_z2": (mu2, logvar2),
            "z_samples": (z1, z2)
        }
        return x_recon, latents

    def prior_given_z2(self, z2):
        # returns mu_p(z2), logvar_p(z2)
        return self.prior_net(z2)

    def loss_function(self, x, x_recon, latents, kl_weight=1.0):
        # Reconstruction:
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

        # KL z2 to standard normal
        mu2, logvar2 = latents["q_z2"]
        kld_z2 = kl_gaussian(mu2, logvar2, None, None)

        # KL z1 to conditional prior p(z1|z2)
        mu1, logvar1 = latents["q_z1"]
        z2 = latents["z_samples"][1]
        mu1_p, logvar1_p = self.prior_given_z2(z2)
        kld_z1 = kl_gaussian(mu1, logvar1, mu1_p, logvar1_p)

        kld = kld_z2 + kld_z1

        loss = recon_loss + kl_weight * kld
        return loss, recon_loss, kld, kld_z1, kld_z2

# ---------------------------
# Utilities: save reconstructions and sampling conditioned
# ---------------------------
def save_reconstructions(x, x_recon, epoch, tag="train"):
    n = min(8, x.size(0))
    comparison = torch.cat([x[:n], x_recon[:n]])
    fname = os.path.join(save_dir, f"{tag}_recon_epoch{epoch}.png")
    utils.save_image(comparison.cpu(), fname, nrow=n)
    print(f"[I/O] Saved reconstructions to {fname}")

def sample_and_save_conditioned(model, epoch, nrow=8):
    model.eval()
    with torch.no_grad():
        # sample z2 ~ N(0, I)
        z2 = torch.randn(nrow, latent_dims[1], device=device)
        # get p(z1|z2) params and sample z1 ~ p(z1|z2)
        mu1_p, logvar1_p = model.prior_given_z2(z2)
        std1_p = torch.exp(0.5 * logvar1_p)
        eps = torch.randn_like(std1_p)
        z1 = mu1_p + eps * std1_p
        imgs = model.decoder(z1, z2)
        fname = os.path.join(save_dir, f"sample_conditioned_epoch{epoch}.png")
        utils.save_image(imgs.cpu(), fname, nrow=nrow)
        print(f"[I/O] Saved conditioned samples to {fname}")

# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(model, optimizer, epoch, global_step):
    model.train()
    running_loss = running_recon = running_kld = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # compute kl_weight linear anneal
        if kl_anneal_epochs > 0:
            kl_weight = float(min(1.0, (global_step + 1) / float(total_train_iters)))
        else:
            kl_weight = 1.0

        optimizer.zero_grad()
        x_recon, latents = model(data)
        loss, recon_loss, kld, kld_z1, kld_z2 = model.loss_function(data, x_recon, latents, kl_weight=kl_weight)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_recon += recon_loss.item()
        running_kld += kld.item()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]"
                  f" loss/px: {loss.item()/len(data):.4f} recon/px: {recon_loss.item()/len(data):.4f}"
                  f" kld_z1: {kld_z1.item()/len(data):.4f} kld_z2: {kld_z2.item()/len(data):.4f}"
                  f" kl_w: {kl_weight:.4f}")

        global_step += 1
    print(f"Epoch {epoch} Train Avg Loss per dataset: {running_loss / len(train_loader.dataset):.6f}")
    return global_step

def evaluate(model, epoch):
    model.eval()
    test_loss = test_recon = test_kld = 0.0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            x_recon, latents = model(data)
            loss, recon_loss, kld, kld_z1, kld_z2 = model.loss_function(data, x_recon, latents, kl_weight=1.0)
            test_loss += loss.item()
            test_recon += recon_loss.item()
            test_kld += kld.item()
            if batch_idx == 0:
                save_reconstructions(data, x_recon, epoch, tag="test")
    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss per data: {test_loss:.6f}, Recon: {test_recon/len(test_loader.dataset):.6f}, KLD: {test_kld/len(test_loader.dataset):.6f}")

# ---------------------------
# Run training
# ---------------------------
def main():
    model = VLAE(z_dims=latent_dims, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_test_loss = float("inf")

    for epoch in range(1, epochs + 1):
        global_step = train_one_epoch(model, optimizer, epoch, global_step)
        evaluate(model, epoch)
        sample_and_save_conditioned(model, epoch)

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict()
        }
        ckpt_path = os.path.join(save_dir, f"vlae_condprior_epoch{epoch}.pt")
        torch.save(ckpt, ckpt_path)

    print("Training finished.")

if __name__ == "__main__":
    main()
