"""
DiT Training Script — Flow Matching

This script trains the Diffusion Transformer (DiT) model on the MNIST dataset
using the flow matching objective instead of DDPM noise prediction.

Key differences from DDPM training (train.py):
- Timestep t is continuous in [0, 1] instead of discrete in {0, ..., T-1}
- Forward process is linear interpolation instead of noise scheduling
- Model predicts velocity (ε - x_0) instead of noise (ε)
- Loss is MSE between predicted and target velocity
- Timestep is scaled by 1000 before passing to the model's sinusoidal embedding

Training loop:
1. Sample a batch of clean images from the dataset
2. Sample t ~ Uniform(0, 1) for each image
3. Interpolate: x_t = (1-t)*x_0 + t*ε
4. Use the DiT model to predict velocity v_θ(x_t, t)
5. Compute MSE loss between v_θ and target velocity (ε - x_0)
6. Backpropagate and update model weights

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2022)
"""

import os
import time

import torch
import wandb
from dataset import MNIST
from dit import DiT
from flow_matching import forward_interpolate
from torch import nn
from torch.utils.data import DataLoader

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

EPOCH = 500
BATCH_SIZE = 300
LEARNING_RATE = 1e-3

# Scale factor for the time embedding input.
# The DiT model's sinusoidal time embedding was designed for inputs in [0, 1000].
# We scale continuous t ∈ [0, 1] by this factor so the embedding frequencies
# cover a similar range.
T_SCALE = 1000

wandb.init(
    entity="nv-welcome",
    project="gym",
    name=f"DiT_101_flow_matching_{time.strftime('%Y%m%d_%H%M%S')}",
    config={
        "method": "flow_matching",
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "t_scale": T_SCALE,
        "loss": "MSE",
    },
)


# ==============================================================================
# DATASET
# ==============================================================================

dataset = MNIST()
print(f"Dataset size: {len(dataset)} images")

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=10,
    persistent_workers=True,
)


# ==============================================================================
# MODEL
# ==============================================================================

model = DiT(
    img_size=28,
    patch_size=4,
    channel=1,
    emb_size=64,
    label_num=10,
    dit_num=3,
    head=4,
).to(DEVICE)

try:
    model.load_state_dict(torch.load("model_fm.pth"))
    print("Loaded existing flow matching checkpoint")
except FileNotFoundError:
    print("No checkpoint found, starting from scratch")
except Exception as e:
    print(f"Could not load checkpoint: {e}")


# ==============================================================================
# OPTIMIZER AND LOSS
# ==============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# MSE loss is standard for flow matching velocity prediction
loss_fn = nn.MSELoss()


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

model.train()
iter_count = 0

print(f"Starting flow matching training for {EPOCH} epochs...")
print(f"Batch size: {BATCH_SIZE}, Batches per epoch: {len(dataloader)}")

for epoch in range(EPOCH):
    for imgs, labels in dataloader:
        # ==== Step 1: Prepare the data ====

        # Normalize pixel values from [0, 1] to [-1, 1]
        x_0 = imgs * 2 - 1  # Shape: (batch, 1, 28, 28)

        # Sample continuous timesteps t ~ Uniform(0, 1)
        t = torch.rand(size=(imgs.size(0),))  # Shape: (batch,)

        y = labels  # Shape: (batch,)

        # ==== Step 2: Forward interpolation ====

        # x_t = (1-t)*x_0 + t*ε, velocity = ε - x_0
        x_t, velocity = forward_interpolate(x_0, t)

        # ==== Step 3: Model prediction ====

        # Scale t to [0, 1000] for the sinusoidal time embedding
        t_scaled = (t * T_SCALE).to(DEVICE)

        pred_velocity = model(x_t.to(DEVICE), t_scaled, y.to(DEVICE))

        # ==== Step 4: Compute loss ====

        loss = loss_fn(pred_velocity, velocity.to(DEVICE))

        # ==== Step 5: Backpropagation and optimization ====

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ==== Step 6: Logging and checkpointing ====

        wandb.log({"loss": loss.item(), "epoch": epoch}, step=iter_count)

        if iter_count % 1000 == 0:
            print(f"Epoch: {epoch}, Iter: {iter_count}, Loss: {loss.item():.6f}")

            torch.save(model.state_dict(), ".model_fm.pth")
            os.replace(".model_fm.pth", "model_fm.pth")

        iter_count += 1

wandb.finish()
print("Training complete!")
