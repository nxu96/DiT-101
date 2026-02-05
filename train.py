"""
DiT Training Script

This script trains the Diffusion Transformer (DiT) model on the MNIST dataset.
The training objective is to predict the noise that was added to an image,
which is the standard DDPM (Denoising Diffusion Probabilistic Models) approach.

Training loop:
1. Sample a batch of clean images from the dataset
2. Sample random timesteps for each image
3. Add noise to images according to the forward diffusion process
4. Use the DiT model to predict the noise
5. Compute loss between predicted and actual noise
6. Backpropagate and update model weights

Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import T  # Total diffusion timesteps
from dataset import MNIST  # MNIST dataset wrapper
from diffusion import forward_add_noise  # Forward diffusion process
from dit import DiT  # Diffusion Transformer model

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Device selection: use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Training hyperparameters
EPOCH = 500  # Number of training epochs
BATCH_SIZE = 300  # Number of images per batch
LEARNING_RATE = 1e-3  # Adam optimizer learning rate


# ==============================================================================
# DATASET
# ==============================================================================

# Load MNIST training dataset
# Each sample is a tuple of (image_tensor, label)
# Image shape: (1, 28, 28), pixel range: [0, 1]
dataset = MNIST()
print(f"Dataset size: {len(dataset)} images")

# Create DataLoader for batching and shuffling
# - shuffle=True: Randomize order each epoch for better training
# - num_workers=10: Use 10 parallel workers for data loading
# - persistent_workers=True: Keep workers alive between epochs (faster)
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

# Initialize the DiT model with MNIST-specific parameters:
# - img_size=28: MNIST images are 28x28 pixels
# - patch_size=4: Split into 4x4 patches (7x7 = 49 patches total)
# - channel=1: Grayscale images (1 channel)
# - emb_size=64: 64-dimensional token embeddings
# - label_num=10: 10 classes (digits 0-9)
# - dit_num=3: 3 DiT transformer blocks
# - head=4: 4 attention heads per block
model = DiT(
    img_size=28,
    patch_size=4,
    channel=1,
    emb_size=64,
    label_num=10,
    dit_num=3,
    head=4,
).to(DEVICE)

# Try to load a previously saved model checkpoint
# This allows resuming training from where we left off
try:
    model.load_state_dict(torch.load("model.pth"))
    print("Loaded existing model checkpoint")
except FileNotFoundError:
    print("No checkpoint found, starting from scratch")
except Exception as e:
    print(f"Could not load checkpoint: {e}")


# ==============================================================================
# OPTIMIZER AND LOSS
# ==============================================================================

# Adam optimizer with learning rate of 1e-3
# Adam is a good default choice for deep learning
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# L1 Loss (Mean Absolute Error)
# Measures average absolute difference between predicted and actual noise
# Alternative: nn.MSELoss() (Mean Squared Error) is also commonly used
loss_fn = nn.L1Loss()


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

# Set model to training mode (enables dropout, batch norm training behavior)
model.train()

# Iteration counter for logging and checkpointing
iter_count = 0

print(f"Starting training for {EPOCH} epochs...")
print(f"Batch size: {BATCH_SIZE}, Batches per epoch: {len(dataloader)}")

for epoch in range(EPOCH):
    for imgs, labels in dataloader:
        # ==== Step 1: Prepare the data ====

        # Normalize pixel values from [0, 1] to [-1, 1]
        # This matches the range of Gaussian noise (mean=0, std=1)
        # and is standard practice in diffusion models
        x = imgs * 2 - 1  # Shape: (batch, 1, 28, 28)

        # Sample random timesteps for each image in the batch
        # Each image gets a different random timestep in [0, T-1]
        t = torch.randint(0, T, (imgs.size(0),))  # Shape: (batch,)

        # Class labels for conditional generation
        y = labels  # Shape: (batch,)

        # ==== Step 2: Forward diffusion (add noise) ====

        # Add noise to images according to the forward diffusion process
        # Returns: x (noisy images), noise (the noise that was added)
        x, noise = forward_add_noise(x, t)

        # ==== Step 3: Model prediction ====

        # Move data to device (GPU/CPU)
        # The model predicts what noise was added to the image
        pred_noise = model(x.to(DEVICE), t.to(DEVICE), y.to(DEVICE))

        # ==== Step 4: Compute loss ====

        # Compare predicted noise with actual noise using L1 loss
        loss = loss_fn(pred_noise, noise.to(DEVICE))

        # ==== Step 5: Backpropagation and optimization ====

        # Clear gradients from previous iteration
        optimizer.zero_grad()

        # Compute gradients via backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # ==== Step 6: Logging and checkpointing ====

        # Log progress and save checkpoint every 1000 iterations
        if iter_count % 1000 == 0:
            print(f"Epoch: {epoch}, Iter: {iter_count}, Loss: {loss.item():.6f}")

            # Save model checkpoint atomically:
            # 1. Save to temporary file first
            # 2. Then rename to final filename
            # This prevents corruption if interrupted during save
            torch.save(model.state_dict(), ".model.pth")
            os.replace(".model.pth", "model.pth")

        iter_count += 1

print("Training complete!")
