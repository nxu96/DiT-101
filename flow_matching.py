"""
Flow Matching Process Implementation

This module implements the forward interpolation process for Flow Matching
(Lipman et al., 2022). Instead of DDPM's noise schedule with betas and alphas,
flow matching defines a straight-line (optimal transport) path from data to noise.

Key equations:
- Forward path:  x_t = (1 - t) * x_0 + t * ε,  where t ∈ [0, 1]
- Velocity:      v   = dx_t/dt = ε - x_0        (constant along each path)

The model is trained to predict this velocity field v_θ(x_t, t) ≈ ε - x_0.
At inference, we solve the ODE dx/dt = v_θ(x, t) backwards from t=1 to t=0.

Compared to DDPM:
- No beta schedule, no alpha cumprod — just linear interpolation
- Continuous t ∈ [0, 1] instead of discrete t ∈ {0, ..., T-1}
- Model predicts velocity instead of noise
- Sampling via ODE solver instead of reverse Markov chain

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2022)
"""

import torch


# ==============================================================================
# FORWARD INTERPOLATION PROCESS
# ==============================================================================


def forward_interpolate(x_0, t):
    """
    Interpolate between clean images and noise along a straight-line path.

    The flow matching forward process defines a linear path from data to noise:
        x_t = (1 - t) * x_0 + t * ε

    where ε ~ N(0, I) is standard Gaussian noise.

    At t=0: x_t = x_0 (clean image)
    At t=1: x_t = ε   (pure noise)

    The velocity (time derivative) of this path is constant:
        v = dx_t/dt = ε - x_0

    This velocity is the training target for the model.

    Args:
        x_0: Clean images
             Shape: (batch, channel, height, width)
             Expected range: [-1, 1] (normalized)
        t: Interpolation timesteps for each image in the batch
           Shape: (batch,)
           Each value is a float in [0, 1]

    Returns:
        x_t: Interpolated images at time t
             Shape: (batch, channel, height, width)
        velocity: Target velocity field v = ε - x_0 (used as training target)
                  Shape: (batch, channel, height, width)
    """
    eps = torch.randn_like(x_0)

    # Reshape t from (batch,) to (batch, 1, 1, 1) for broadcasting
    t = t.view(-1, 1, 1, 1)

    # Linear interpolation: x_t = (1 - t) * x_0 + t * ε
    x_t = (1 - t) * x_0 + t * eps

    # Velocity is the derivative of the path: v = ε - x_0
    velocity = eps - x_0

    return x_t, velocity


# ==============================================================================
# TEST / VISUALIZATION
# ==============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataset import MNIST

    dataset = MNIST()

    # Stack 2 images into a batch: (2, 1, 28, 28)
    x = torch.stack((dataset[0][0], dataset[1][0]), dim=0)

    # Normalize pixel values from [0, 1] to [-1, 1]
    x = x * 2 - 1

    # Sample random continuous timesteps in [0, 1]
    t = torch.rand(size=(x.size(0),))
    print(f"Timesteps: {t}")

    # Apply forward interpolation
    x_t, velocity = forward_interpolate(x, t)
    print(f"Interpolated image shape: {x_t.shape}")
    print(f"Velocity shape: {velocity.shape}")

    # Display interpolated images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title(f"Interpolated Image 1 (t={t[0].item():.2f})")
    plt.imshow(((x_t[0] + 1) / 2).permute(1, 2, 0), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title(f"Interpolated Image 2 (t={t[1].item():.2f})")
    plt.imshow(((x_t[1] + 1) / 2).permute(1, 2, 0), cmap="gray")
    # Save the plot
    plt.savefig("flow_matching.png")