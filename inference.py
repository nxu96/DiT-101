"""
DiT Inference Script

This script performs inference (image generation) using a trained DiT model.
It implements two reverse diffusion samplers:

1. DDPM (Ho et al., 2020) - The original stochastic sampler, requires all T steps.
2. DDIM (Song et al., 2020) - A faster, deterministic sampler that can skip steps.
   Uses the SAME trained model with no retraining needed.

Both samplers start from pure Gaussian noise and iteratively denoise to produce images.
"""

import matplotlib.pyplot as plt
import torch
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


def backward_denoise_ddpm(model, x, y):
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
# DDIM REVERSE DIFFUSION (DETERMINISTIC / ACCELERATED DENOISING)
# ==============================================================================


def backward_denoise_ddim(model, x, y, ddim_steps=50, eta=0.0):
    """
    Perform the DDIM reverse process to generate images from noise.

    DDIM (Song et al., 2020) derives a non-Markovian reverse process that
    shares the same forward marginals q(x_t | x_0) as DDPM. This allows:
    - Skipping timesteps for much faster sampling (e.g. 50 steps vs 1000)
    - Deterministic generation (eta=0) for reproducible outputs
    - Tunable stochasticity via eta (eta=1 recovers DDPM)

    DDIM reverse step:
        1. Predict x_0:  x̂_0 = (x_t - √(1-α̅_t) · ε_θ) / √α̅_t
        2. Direction:     d   = √(1 - α̅_{t-1} - σ²) · ε_θ
        3. Noise:         σ   = η · √((1-α̅_{t-1})/(1-α̅_t)) · √(1 - α̅_t/α̅_{t-1})
        4. Sample:  x_{t-1}  = √α̅_{t-1} · x̂_0  +  d  +  σ · z

    Args:
        model: Trained DiT model (same model used for DDPM, no retraining)
        x: Initial noise tensor, shape (batch, channel, height, width)
        y: Class labels for conditional generation, shape (batch,)
        ddim_steps: Number of denoising steps (can be much less than T)
        eta: Stochasticity parameter. 0 = deterministic, 1 = DDPM-like

    Returns:
        steps: List of tensors showing the denoising progression
    """
    steps = [x.clone()]

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    alphas_cumprod_device = alphas_cumprod.to(DEVICE)

    # Build a subsequence of T timesteps evenly spaced, e.g. [999, 979, ..., 0]
    # This is the key to DDIM's speed: we only visit ddim_steps out of T total
    timestep_seq = torch.linspace(T - 1, 0, ddim_steps + 1).long()

    model.eval()
    with torch.no_grad():
        for i in range(ddim_steps):
            t_cur = timestep_seq[i]  # current timestep  (e.g. 999)
            t_prev = timestep_seq[i + 1]  # previous timestep (e.g. 979)

            t_batch = torch.full((x.size(0),), t_cur, device=x.device)

            # Model predicts the noise component ε_θ(x_t, t)
            pred_noise = model(x, t_batch, y)

            alpha_bar_t = alphas_cumprod_device[t_cur].view(1, 1, 1, 1)
            alpha_bar_prev = (
                alphas_cumprod_device[t_prev].view(1, 1, 1, 1)
                if t_prev >= 0
                else torch.ones(1, 1, 1, 1, device=DEVICE)
            )

            # Step 1: predict x_0 from x_t and predicted noise
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(
                alpha_bar_t
            )
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Step 2: compute DDIM variance (controlled by eta)
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )

            # Step 3: direction pointing to x_t
            direction = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise

            # Step 4: combine into x_{t-1}
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction
            if eta > 0 and t_prev > 0:
                x = x + sigma * torch.randn_like(x)

            x = torch.clamp(x, -1.0, 1.0).detach()
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

# Choose sampler: "ddpm" (1000 steps, stochastic) or "ddim" (fast, deterministic)
SAMPLER = "ddpm"
DDIM_STEPS = 50  # Only used when SAMPLER="ddim"; try 20, 50, or 100

if SAMPLER == "ddim":
    print(f"Using DDIM sampler with {DDIM_STEPS} steps (eta=0, deterministic)")
    steps = backward_denoise_ddim(model, x, y, ddim_steps=DDIM_STEPS, eta=0.0)
else:
    print(f"Using DDPM sampler with {T} steps")
    steps = backward_denoise_ddpm(model, x, y)

num_steps = len(steps) - 1
print(f"Denoising complete! {num_steps} denoising steps")


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
        # Evenly spaced indices through the denoising trajectory
        idx = int(num_steps / num_imgs) * (i + 1)
        idx = min(idx, num_steps)

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
plt.savefig(f"inference_{SAMPLER}.png", dpi=150, bbox_inches="tight")

print(f"Visualization saved to inference_{SAMPLER}.png")
