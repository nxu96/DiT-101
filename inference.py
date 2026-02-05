"""
DiT Inference Script

This script performs inference (image generation) using a trained DiT model.
It implements the reverse diffusion process (denoising) to generate images
from pure Gaussian noise.

The reverse process works by:
1. Starting with pure random noise
2. Iteratively denoising from timestep T-1 down to 0
3. At each step, the model predicts the noise component
4. We remove part of the predicted noise to get a cleaner image

Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
import matplotlib.pyplot as plt

from config import T  # Total diffusion timesteps
from diffusion import alphas, alphas_cumprod, variance  # Diffusion parameters
from dit import DiT  # Diffusion Transformer model


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Device selection: use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ==============================================================================
# REVERSE DIFFUSION (DENOISING) PROCESS
# ==============================================================================


def backward_denoise(model, x, y):
    """
    Perform the reverse diffusion process to generate images from noise.

    Starting from pure noise (x_T), iteratively denoise to get the final image (x_0).
    At each timestep t, we:
    1. Use the model to predict the noise in x_t
    2. Compute the mean of x_{t-1} using the DDPM formula
    3. Add a small amount of random noise (except at t=0)

    The DDPM reverse process formula:
        μ_θ(x_t, t) = (1/√α_t) * (x_t - (1-α_t)/√(1-α̅_t) * ε_θ(x_t, t))
        x_{t-1} = μ_θ(x_t, t) + σ_t * z,  where z ~ N(0, I)

    Args:
        model: Trained DiT model
        x: Initial noise tensor, shape (batch, channel, height, width)
        y: Class labels for conditional generation, shape (batch,)

    Returns:
        steps: List of tensors showing the denoising progression
               First element is the initial noise, last is the final image
    """
    # Store intermediate results for visualization
    steps = [x.clone()]

    # Move diffusion parameters to the correct device
    # We use 'global' to modify the module-level variables
    global alphas, alphas_cumprod, variance
    alphas_device = alphas.to(DEVICE)
    alphas_cumprod_device = alphas_cumprod.to(DEVICE)
    variance_device = variance.to(DEVICE)

    # Move input tensors to device
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    # Set model to evaluation mode (disables dropout, uses running stats for batchnorm)
    model.eval()

    # Disable gradient computation for inference (saves memory and speeds up)
    with torch.no_grad():
        # Iterate backwards from T-1 to 0
        for time in range(T - 1, -1, -1):
            # Create a tensor of the current timestep for the entire batch
            # Shape: (batch,), all values equal to 'time'
            t = torch.full((x.size(0),), time).to(DEVICE)

            # ==== Step 1: Predict the noise at timestep t ====
            # The model takes noisy image, timestep, and class label
            # and outputs its prediction of what noise was added
            noise = model(x, t, y)

            # ==== Step 2: Compute the mean of x_{t-1} ====
            # Using the DDPM reverse process formula:
            # μ = (1/√α_t) * (x_t - (1-α_t)/√(1-α̅_t) * ε_θ)

            # Shape for broadcasting: (batch, 1, 1, 1)
            shape = (x.size(0), 1, 1, 1)

            # Get alpha values for current timestep and reshape for broadcasting
            alpha_t = alphas_device[t].view(*shape)
            alpha_cumprod_t = alphas_cumprod_device[t].view(*shape)
            variance_t = variance_device[t].view(*shape)

            # Compute the mean of the posterior distribution
            # Formula: μ = (1/√α_t) * (x_t - (1-α_t)/√(1-α̅_t) * predicted_noise)
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise
            )

            # ==== Step 3: Sample x_{t-1} ====
            if time != 0:
                # For t > 0: add random noise scaled by the variance
                # x_{t-1} = μ + σ_t * z, where z ~ N(0, I)
                x = mean + torch.randn_like(x) * torch.sqrt(variance_t)
            else:
                # For t = 0: no noise added (final step)
                x = mean

            # Clamp values to [-1, 1] to prevent numerical instability
            x = torch.clamp(x, -1.0, 1.0).detach()

            # Store this step for visualization
            steps.append(x.clone())

    return steps


# ==============================================================================
# LOAD TRAINED MODEL
# ==============================================================================

# Initialize the DiT model with the same architecture used during training
model = DiT(
    img_size=28,
    patch_size=4,
    channel=1,
    emb_size=64,
    label_num=10,
    dit_num=3,
    head=4,
).to(DEVICE)

# Load the trained weights
model.load_state_dict(torch.load("model.pth"))
print("Model loaded successfully")


# ==============================================================================
# GENERATE IMAGES
# ==============================================================================

# Number of images to generate (one for each digit 0-9)
batch_size = 10

# Start with pure random noise
# Shape: (batch_size, 1, 28, 28) - same as MNIST images
x = torch.randn(size=(batch_size, 1, 28, 28))

# Class labels: generate one of each digit (0, 1, 2, ..., 9)
y = torch.arange(start=0, end=10, dtype=torch.long)

print(f"Generating {batch_size} images...")
print(f"Class labels: {y.tolist()}")

# Run the reverse diffusion process to denoise
steps = backward_denoise(model, x, y)

print(f"Denoising complete! Generated {len(steps)} intermediate steps")


# ==============================================================================
# VISUALIZE RESULTS
# ==============================================================================

# Number of intermediate steps to show in the visualization
num_imgs = 20

# Create a figure with batch_size rows and num_imgs columns
# Each row shows the denoising progression for one image
plt.figure(figsize=(15, 15))

for b in range(batch_size):
    for i in range(num_imgs):
        # Calculate which step to show (evenly spaced through the process)
        # +1 because steps[0] is pure noise, steps[T] is final image
        idx = int(T / num_imgs) * (i + 1)

        # Get the image at this step and move to CPU
        # Convert from [-1, 1] back to [0, 1] for visualization
        final_img = (steps[idx][b].to("cpu") + 1) / 2

        # Reshape from (C, H, W) to (H, W, C) for matplotlib
        final_img = final_img.permute(1, 2, 0)

        # Plot in a grid
        plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
        plt.imshow(final_img, cmap="gray")
        plt.axis("off")

plt.suptitle("Denoising Process: Noise → Generated Digit", fontsize=14)
plt.tight_layout()
plt.savefig("inference.png", dpi=150, bbox_inches="tight")
plt.show()

print("Visualization saved to inference.png")
