"""
Diffusion Process Implementation

This module implements the forward diffusion process for DDPM (Denoising Diffusion
Probabilistic Models). The forward process gradually adds Gaussian noise to images
over T timesteps, and computes the parameters needed for both forward and reverse
(denoising) processes.

Key equations from the DDPM paper:
- Forward process: q(x_t | x_0) = N(x_t; sqrt(α̅_t) * x_0, (1 - α̅_t) * I)
- This allows us to sample x_t directly from x_0 without iterating through all steps

Notation:
- β_t (beta): Variance schedule, controls noise added at each step
- α_t (alpha): 1 - β_t
- α̅_t (alpha_bar/cumprod): Product of all α from step 1 to t

Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
from config import T  # Total number of diffusion timesteps (e.g., 1000)

# ==============================================================================
# DIFFUSION SCHEDULE PARAMETERS
# ==============================================================================
# These parameters define the noise schedule and are precomputed for efficiency.
# They are used in both the forward (adding noise) and reverse (denoising) processes.

# Beta schedule: linear increase from 0.0001 to 0.02 over T steps
# β_t controls how much noise is added at each step
# Small values at start (preserve structure) -> larger values at end (more noise)
betas = torch.linspace(0.0001, 0.02, T)  # Shape: (T,)

# Alpha: the "keep" ratio at each step (how much of the signal to retain)
# α_t = 1 - β_t
alphas = 1 - betas  # Shape: (T,)

# Alpha cumulative product: product of all alphas from step 0 to t
# α̅_t = α_1 * α_2 * ... * α_t
# This tells us how much of the original signal remains at step t
# Example: [α_1, α_1*α_2, α_1*α_2*α_3, ...]
alphas_cumprod = torch.cumprod(alphas, dim=-1)  # Shape: (T,)

# Alpha cumulative product for previous timestep (t-1)
# Prepend 1.0 for t=0 case (no noise added yet, full signal)
# Example: [1, α_1, α_1*α_2, α_1*α_2*α_3, ...]
# This is used in the reverse/denoising process
alphas_cumprod_prev = torch.cat(
    (torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1
)  # Shape: (T,)

# Posterior variance: used in the reverse (denoising) process
# This is the variance of q(x_{t-1} | x_t, x_0)
# Formula: σ²_t = β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
variance = (
    (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
)  # Shape: (T,)


# ==============================================================================
# FORWARD DIFFUSION PROCESS
# ==============================================================================


def forward_add_noise(x, t):
    """
    Add noise to images according to the forward diffusion process.

    The key insight of DDPM is that we can sample x_t directly from x_0
    without iterating through all intermediate steps:

        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε

    where ε ~ N(0, I) is standard Gaussian noise.

    Args:
        x: Clean images (or partially noised images)
           Shape: (batch, channel, height, width)
           Expected range: [-1, 1] (normalized)
        t: Timesteps for each image in the batch
           Shape: (batch,)
           Each value is an integer in [0, T-1]

    Returns:
        x_noisy: Images with noise added according to timestep t
                 Shape: (batch, channel, height, width)
        noise: The Gaussian noise that was added (used as training target)
               Shape: (batch, channel, height, width)
    """
    # Sample random Gaussian noise with same shape as input
    noise = torch.randn_like(x)  # Shape: (batch, channel, height, width)
    # NOTE: The values of noise are iid. So the noise of every pixel in
    # every image is independently sampled from a standard normal distribution.

    # Get α̅_t for each image's timestep and reshape for broadcasting
    # alphas_cumprod[t] extracts the values for each batch item
    # .view() reshapes from (batch,) to (batch, 1, 1, 1) for broadcasting
    batch_alphas_cumprod = alphas_cumprod[t].view(x.size(0), 1, 1, 1)

    # Apply the forward diffusion formula:
    # x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
    #
    # - sqrt(α̅_t) * x_0: Scaled original image (signal component)
    # - sqrt(1 - α̅_t) * ε: Scaled noise (noise component)
    #
    # At t=0: α̅_t ≈ 1, so x_t ≈ x_0 (almost no noise)
    # At t=T-1: α̅_t ≈ 0, so x_t ≈ ε (almost pure noise)
    # NOTE: x_noisy is actually a linear combo of the original image and the noise.
    # It is not a simple addition of the two.
    x_noisy = (
        torch.sqrt(batch_alphas_cumprod) * x
        + torch.sqrt(1 - batch_alphas_cumprod) * noise
    )

    return x_noisy, noise


# ==============================================================================
# TEST / VISUALIZATION
# ==============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataset import MNIST

    # Load MNIST dataset
    dataset = MNIST()

    # Stack 2 images into a batch: (2, 1, 28, 28)
    x = torch.stack((dataset[0][0], dataset[1][0]), dim=0)

    # Display original images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image 1")
    plt.imshow(x[0].permute(1, 2, 0))  # (C, H, W) -> (H, W, C) for matplotlib
    plt.subplot(1, 2, 2)
    plt.title("Original Image 2")
    plt.imshow(x[1].permute(1, 2, 0))
    plt.show()

    # Sample random timesteps for each image
    t = torch.randint(0, T, size=(x.size(0),))
    print(f"Timesteps: {t}")

    # Normalize pixel values from [0, 1] to [-1, 1]
    # This matches the range of Gaussian noise (mean=0)
    # and is standard practice in diffusion models
    x = x * 2 - 1

    # Apply forward diffusion (add noise)
    x_noisy, noise = forward_add_noise(x, t)
    print(f"Noisy image shape: {x_noisy.shape}")
    print(f"Noise shape: {noise.shape}")

    # Display noisy images
    # Convert back from [-1, 1] to [0, 1] for visualization: (x + 1) / 2
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(f"Noisy Image 1 (t={t[0].item()})")
    plt.imshow(((x_noisy[0] + 1) / 2).permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.title(f"Noisy Image 2 (t={t[1].item()})")
    plt.imshow(((x_noisy[1] + 1) / 2).permute(1, 2, 0))
    plt.show()
